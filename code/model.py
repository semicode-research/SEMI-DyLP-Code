import torch
import torch.nn as nn
from torch.nn import init
from combiner import Combiner
from edge_updater import Edge_updater_nn
from node_updater import TLSTM
from scipy.sparse import lil_matrix, find
import numpy as np
from numpy.random import choice
from decayer import Decayer
from attention import Attention, EnhancedGATAttention
import time
import random


class DyGNN(nn.Module):
    def __init__(self, num_embeddings, embedding_dims, edge_output_size, device, w, is_att=False, transfer=False, nor=0,
                 if_no_time=0, threhold=None, threhold_2nd=None, second_order=False, if_updated=0, drop_p=0,
                 num_negative=5, act='tanh',
                 if_propagation=1, decay_method='exp', weight=None, relation_size=None, bias=True,
                 gat_heads=4, second_order_max_nodes=5, gat_dropout=0.1, use_enhanced_gat=True,
                 profile_runtime=False, second_order_bridge_max_nodes=0, second_order_mode='per_bridge',
                 hard_negative_ratio=0.5, negative_pool_size=0, negative_sampling_strategy='legacy',
                 persistence_weight=0.0, persistence_tau=0.0,
                 state_lookup_mode='legacy', negative_rep_timing='legacy', graph_mode='directed'):
        super(DyGNN, self).__init__()
        self.embedding_dims = embedding_dims
        self.num_embeddings = num_embeddings
        self.nor = nor
        self.device = device
        self.second_order_max_nodes = second_order_max_nodes
        self.second_order_bridge_max_nodes = second_order_bridge_max_nodes
        self.second_order_mode = second_order_mode
        self.transfer = transfer
        self.if_propagation = if_propagation
        self.if_no_time = if_no_time
        self.second_order = second_order
        self.combiner = Combiner(embedding_dims, embedding_dims, act).to(device)
        self.decay_method = decay_method
        self.if_updated = if_updated

        self.threhold = threhold
        self.threhold_2nd = threhold_2nd

        self.use_enhanced_gat = use_enhanced_gat
        self.num_negative = num_negative
        self.hard_negative_ratio = float(np.clip(hard_negative_ratio, 0.0, 1.0))
        self.negative_pool_size = int(negative_pool_size)
        self.negative_sampling_strategy = negative_sampling_strategy
        self.persistence_weight = float(persistence_weight)
        self.persistence_tau = float(persistence_tau)
        self.state_lookup_mode = state_lookup_mode
        self.negative_rep_timing = negative_rep_timing
        self.graph_mode = str(graph_mode).strip().lower()
        if self.graph_mode not in {'directed', 'undirected'}:
            raise ValueError('graph_mode must be directed or undirected')
        self.is_undirected = self.graph_mode == 'undirected'
        self.profile_runtime = profile_runtime
        self.profile_stats = {}
        self.profile_count_keys = set()

        print(f'1st Order Threshold: {threhold}, 2nd Order Threshold: {threhold_2nd}')
        print(f'Using Enhanced GAT: {use_enhanced_gat}, Heads: {gat_heads}, Dropout: {gat_dropout}')
        pool_text = self.negative_pool_size if self.negative_pool_size > 0 else 'history-only'
        print(f'Model Config: Dim={embedding_dims}, NegSamples={num_negative}, '
              f'NegStrategy={self.negative_sampling_strategy}, '
              f'HardNegRatio={self.hard_negative_ratio}, NegPool={pool_text}, '
              f'PersistenceWeight={self.persistence_weight}, PersistenceTau={self.persistence_tau}, '
              f'StateLookup={self.state_lookup_mode}, NegRepTiming={self.negative_rep_timing}, '
              f'GraphMode={self.graph_mode}')

        if act == 'tanh':
            self.act = nn.Tanh().to(device)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid().to(device)
        else:
            self.act = nn.ReLU().to(device)

        self.decayer = Decayer(device, w, decay_method)
        self.edge_updater_head = Edge_updater_nn(embedding_dims, edge_output_size, act, relation_size).to(device)
        self.edge_updater_tail = Edge_updater_nn(embedding_dims, edge_output_size, act, relation_size).to(device)

        if if_no_time:
            self.node_updater_head = nn.LSTMCell(edge_output_size, embedding_dims, bias).to(device)
            self.node_updater_tail = nn.LSTMCell(edge_output_size, embedding_dims, bias).to(device)
        else:
            self.node_updater_head = TLSTM(edge_output_size, embedding_dims).to(device)
            self.node_updater_tail = TLSTM(edge_output_size, embedding_dims).to(device)

        self.tran_head_edge_head = nn.Linear(edge_output_size, embedding_dims, bias).to(device)
        self.tran_head_edge_tail = nn.Linear(edge_output_size, embedding_dims, bias).to(device)
        self.tran_tail_edge_head = nn.Linear(edge_output_size, embedding_dims, bias).to(device)
        self.tran_tail_edge_tail = nn.Linear(edge_output_size, embedding_dims, bias).to(device)

        self.is_att = is_att
        if self.is_att:
            if use_enhanced_gat:
                self.attention = EnhancedGATAttention(embedding_dims=embedding_dims, num_heads=gat_heads,
                                                      dropout=gat_dropout, alpha=0.2, use_residual=True,
                                                      use_layer_norm=True).to(device)
                self.head_attention = EnhancedGATAttention(embedding_dims=embedding_dims, num_heads=gat_heads,
                                                           dropout=gat_dropout, alpha=0.2).to(device)
                self.tail_attention = EnhancedGATAttention(embedding_dims=embedding_dims, num_heads=gat_heads,
                                                           dropout=gat_dropout, alpha=0.2).to(device)
            else:
                self.attention = Attention(embedding_dims, num_heads=gat_heads, dropout=gat_dropout).to(device)

        if self.is_att and use_enhanced_gat:
            self.attention_fusion = nn.Sequential(nn.Linear(embedding_dims * 2, embedding_dims), nn.ReLU(),
                                                  nn.Dropout(gat_dropout),
                                                  nn.Linear(embedding_dims, embedding_dims)).to(device)
            self.temporal_attention_scale = nn.Parameter(torch.tensor(1.0))

        self.recent_timestamp = torch.zeros((num_embeddings, 1), dtype=torch.float, requires_grad=False).to(device)

        self.adj_out = [dict() for _ in range(num_embeddings)]
        self.adj_in = [dict() for _ in range(num_embeddings)]
        self._adj_out_arrays = [None for _ in range(num_embeddings)]
        self._adj_in_arrays = [None for _ in range(num_embeddings)]
        self._adj_out_version = [0 for _ in range(num_embeddings)]
        self._adj_in_version = [0 for _ in range(num_embeddings)]
        self._degree_array = np.zeros(num_embeddings, dtype=np.float32)

        if weight is None:
            self.cell_head = nn.Embedding(num_embeddings, embedding_dims).to(device)
            self.cell_tail = nn.Embedding(num_embeddings, embedding_dims).to(device)
            self.hidden_head = nn.Embedding(num_embeddings, embedding_dims).to(device)
            self.hidden_tail = nn.Embedding(num_embeddings, embedding_dims).to(device)
            self.node_representations = nn.Embedding(num_embeddings, embedding_dims).to(device)
        else:
            self.cell_head = nn.Embedding(num_embeddings, embedding_dims, _weight=weight).to(device)
            self.cell_tail = nn.Embedding(num_embeddings, embedding_dims, _weight=weight).to(device)
            self.hidden_head = nn.Embedding(num_embeddings, embedding_dims, _weight=weight).to(device)
            self.hidden_tail = nn.Embedding(num_embeddings, embedding_dims, _weight=weight).to(device)
            self.node_representations = nn.Embedding(num_embeddings, embedding_dims, _weight=weight).to(device)

        self.cell_head.weight.requires_grad = False
        self.cell_tail.weight.requires_grad = False
        self.hidden_head.weight.requires_grad = False
        self.hidden_tail.weight.requires_grad = False
        self.node_representations.weight.requires_grad = False

        if transfer:
            self.transfer2head = nn.Linear(embedding_dims, embedding_dims, False).to(device)
            self.transfer2tail = nn.Linear(embedding_dims, embedding_dims, False).to(device)
            if drop_p >= 0:
                self.dropout = nn.Dropout(p=drop_p).to(device)

        self.cell_head_copy = nn.Embedding.from_pretrained(self.cell_head.weight.clone()).to(device)
        self.cell_tail_copy = nn.Embedding.from_pretrained(self.cell_tail.weight.clone()).to(device)
        self.hidden_head_copy = nn.Embedding.from_pretrained(self.hidden_head.weight.clone()).to(device)
        self.hidden_tail_copy = nn.Embedding.from_pretrained(self.hidden_tail.weight.clone()).to(device)
        self.node_representations_copy = nn.Embedding.from_pretrained(self.node_representations.weight.clone()).to(
            device)
        self._temporal_dist_cache = None
        self._degree_cache = None
        self._wasserstein_score_cache = None
        self._neighbors_cache = None
        self._current_t_value = None
        self._last_score_priors = None

    def reset_profile_stats(self):
        self.profile_stats = {}
        self.profile_count_keys = set()

    def add_profile_time(self, name, elapsed):
        if self.profile_runtime:
            self.profile_stats[name] = self.profile_stats.get(name, 0.0) + elapsed

    def add_profile_count(self, name, count):
        if self.profile_runtime:
            self.profile_stats[name] = self.profile_stats.get(name, 0.0) + count
            self.profile_count_keys.add(name)

    def select_second_order_bridges(self, bridge_scores, max_nodes):
        if max_nodes is None or max_nodes <= 0:
            return None
        if bridge_scores is None or bridge_scores.numel() <= max_nodes:
            return None
        _, indices = torch.topk(bridge_scores.view(-1), k=max_nodes)
        return set(int(i) for i in indices.detach().cpu().tolist())

    def _time_value(self, current_t):
        cached_time = getattr(self, '_current_t_value', None)
        if cached_time is not None:
            return cached_time
        if isinstance(current_t, torch.Tensor):
            return float(current_t.detach().cpu().item())
        return float(current_t)

    def canonical_endpoints(self, first, second):
        """Return the unique representation of an unordered event."""
        first = int(first)
        second = int(second)
        if self.is_undirected and first > second:
            return second, first
        return first, second

    def project_for_scoring(self, representations, role):
        """Project one representation (directed/legacy helper).

        Undirected pair scores use ``score_undirected_pairs``. Averaging the
        two projections here is retained only for compatibility with external
        code that requests a single projected representation.
        """
        if not self.transfer:
            return representations
        if self.is_undirected:
            # Retain both learned role projections, but aggregate them
            # commutatively.  This preserves model capacity while ensuring
            # that s(u, v) == s(v, u).
            return 0.5 * (
                self.transfer2head(representations)
                + self.transfer2tail(representations)
            )
        if role == 'head':
            return self.transfer2head(representations)
        if role == 'tail':
            return self.transfer2tail(representations)
        raise ValueError("role must be 'head' or 'tail'")

    def project_undirected_views(self, representations, apply_dropout=False):
        """Project one endpoint once and return its two latent score views."""
        if not self.is_undirected:
            raise ValueError("project_undirected_views is only valid in undirected mode")
        if apply_dropout:
            representations = self.dropout(representations)
        if self.transfer:
            head_view = self.transfer2head(representations)
            tail_view = self.transfer2tail(representations)
        else:
            head_view = tail_view = representations
        if self.nor:
            head_view = nn.functional.normalize(head_view, dim=-1)
            tail_view = nn.functional.normalize(tail_view, dim=-1)
        return head_view, tail_view

    def score_undirected_views(self, left_head, left_tail, right_head, right_tail):
        """Score two already-projected endpoints without repeated projection."""
        score = 0.5 * (
            torch.sum(left_head * right_tail, dim=-1)
            + torch.sum(left_tail * right_head, dim=-1)
        )
        return score / np.sqrt(float(self.embedding_dims))

    def score_undirected_pairs(self, left, right, apply_dropout=False):
        """Compute an expressive endpoint-swap-invariant pair score.

        Instead of taking a dot product after averaging the two projections,
        cross the two projections and average the two orientations:

          0.5 * (<P_h left, P_t right> + <P_t left, P_h right>).

        This preserves score(left, right) == score(right, left), while giving
        the two projection matrices distinct and useful gradients.
        """
        left_head, left_tail = self.project_undirected_views(left, apply_dropout)
        right_head, right_tail = self.project_undirected_views(right, apply_dropout)
        return self.score_undirected_views(
            left_head, left_tail, right_head, right_tail
        )

    @staticmethod
    def fast_wasserstein_1d(source_dist, candidate_dist):
        source = np.sort(np.asarray(source_dist, dtype=np.float32))
        candidate = np.sort(np.asarray(candidate_dist, dtype=np.float32))
        if source.size == 0 or candidate.size == 0:
            return 0.0
        all_values = np.concatenate((source, candidate))
        all_values.sort()
        deltas = np.diff(all_values)
        if deltas.size == 0:
            return 0.0
        source_cdf = np.searchsorted(source, all_values[:-1], side='right') / source.size
        candidate_cdf = np.searchsorted(candidate, all_values[:-1], side='right') / candidate.size
        return float(np.sum(np.abs(source_cdf - candidate_cdf) * deltas))

    def get_hard_negative_samples(self, node, candidates, node_reps, num_samples=5):
        if len(candidates) == 0:
            return []
        if len(candidates) <= num_samples:
            return list(candidates)
        candidate_list = list(candidates)
        if len(candidate_list) > 1000:
            candidate_list = random.sample(candidate_list, 1000)

        node_rep = node_reps(torch.as_tensor([node], dtype=torch.long, device=self.device))
        candidate_tensors = node_reps(torch.as_tensor(candidate_list, dtype=torch.long, device=self.device))

        scores = torch.mm(candidate_tensors, node_rep.t()).squeeze()
        k = min(num_samples, len(candidate_list))
        _, indices = torch.topk(scores, k)

        selected_candidates = [candidate_list[i] for i in indices.cpu().numpy()]
        return selected_candidates

    def sample_negative_nodes(self, anchor_node, candidate_nodes, forbidden_nodes, node_reps,
                              num_samples=None):
        if num_samples is None:
            num_samples = self.num_negative
        forbidden_nodes = set(forbidden_nodes)
        strategy = self.negative_sampling_strategy
        if strategy == 'random':
            return self.sample_random_nodes(num_samples, forbidden_nodes)

        candidate_set = set(int(node) for node in candidate_nodes) - forbidden_nodes

        if self.negative_pool_size > 0 and len(candidate_set) < self.negative_pool_size:
            needed = self.negative_pool_size - len(candidate_set)
            candidate_set.update(self.sample_random_nodes(needed, forbidden_nodes | candidate_set))

        if strategy == 'legacy':
            if not candidate_set:
                return self.sample_random_nodes(num_samples, forbidden_nodes)
            candidate_list = list(candidate_set)
            if np.random.random() > (1.0 - self.hard_negative_ratio):
                selected = self.get_hard_negative_samples(
                    anchor_node, candidate_list, node_reps, min(num_samples, len(candidate_list))
                )
            elif len(candidate_list) >= num_samples:
                selected = random.sample(candidate_list, num_samples)
            else:
                selected = candidate_list
            if len(selected) < num_samples:
                selected.extend(self.sample_random_nodes(
                    num_samples - len(selected), forbidden_nodes | set(selected)
                ))
            return selected[:num_samples]

        if strategy == 'history':
            selected = []
            if len(candidate_set) >= num_samples:
                selected.extend(random.sample(list(candidate_set), num_samples))
            else:
                selected.extend(list(candidate_set))
                selected.extend(self.sample_random_nodes(
                    num_samples - len(selected), forbidden_nodes | set(selected)
                ))
            return selected[:num_samples]

        selected = []
        if strategy == 'hard':
            num_hard = num_samples
        else:
            num_hard = min(num_samples, int(round(num_samples * self.hard_negative_ratio)))

        if candidate_set and num_hard > 0:
            selected.extend(self.get_hard_negative_samples(
                anchor_node, list(candidate_set), node_reps, min(num_hard, len(candidate_set))
            ))

        if len(selected) < num_samples:
            remaining = num_samples - len(selected)
            selected.extend(self.sample_random_nodes(remaining, forbidden_nodes | set(selected)))

        if len(selected) > num_samples:
            selected = selected[:num_samples]
        return selected

    def sample_random_nodes(self, num_samples, forbidden_nodes=None):
        if forbidden_nodes is None:
            forbidden_nodes = set()
        else:
            forbidden_nodes = set(forbidden_nodes)
        if len(forbidden_nodes) >= self.num_embeddings:
            return list(choice(range(self.num_embeddings), size=num_samples))
        samples = []
        selected = set()
        while len(samples) < num_samples:
            candidate = random.randrange(self.num_embeddings)
            if candidate not in forbidden_nodes and candidate not in selected:
                samples.append(candidate)
                selected.add(candidate)
            if len(forbidden_nodes) + len(selected) >= self.num_embeddings:
                break
        while len(samples) < num_samples:
            candidate = random.randrange(self.num_embeddings)
            if candidate not in forbidden_nodes:
                samples.append(candidate)
        return samples

    def get_persistence_prior(self, head, tail, current_t):
        if self.persistence_weight <= 0:
            return 0.0
        last_t = self.adj_out[int(head)].get(int(tail))
        if last_t is None:
            return 0.0
        if current_t is None:
            return 1.0
        if self.persistence_tau > 0:
            delta_t = max(self._time_value(current_t) - float(last_t), 0.0)
            return float(np.exp(-delta_t / self.persistence_tau))
        return 1.0

    def reset_time(self):
        self.recent_timestamp = torch.zeros((self.num_embeddings, 1), dtype=torch.float, requires_grad=False).to(
            self.device)
        self.adj_out = [dict() for _ in range(self.num_embeddings)]
        self.adj_in = [dict() for _ in range(self.num_embeddings)]
        self._adj_out_arrays = [None for _ in range(self.num_embeddings)]
        self._adj_in_arrays = [None for _ in range(self.num_embeddings)]
        self._adj_out_version = [0 for _ in range(self.num_embeddings)]
        self._adj_in_version = [0 for _ in range(self.num_embeddings)]
        self._degree_array = np.zeros(self.num_embeddings, dtype=np.float32)
        self._temporal_dist_cache = None
        self._degree_cache = None
        self._wasserstein_score_cache = None
        self._neighbors_cache = None
        self._current_t_value = None

    def reset_reps(self):
        self.cell_head = nn.Embedding.from_pretrained(self.cell_head_copy.weight.clone()).to(self.device)
        self.cell_tail = nn.Embedding.from_pretrained(self.cell_tail_copy.weight.clone()).to(self.device)
        self.hidden_head = nn.Embedding.from_pretrained(self.hidden_head_copy.weight.clone()).to(self.device)
        self.hidden_tail = nn.Embedding.from_pretrained(self.hidden_tail_copy.weight.clone()).to(self.device)
        self.node_representations = nn.Embedding.from_pretrained(self.node_representations_copy.weight.clone()).to(
            self.device)

    def compute_temporal_distribution(self, node, current_t):
        cache = getattr(self, '_temporal_dist_cache', None)
        cache_key = None
        if cache is not None:
            cache_key = (int(node), self._time_value(current_t), self.threhold)
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        head_neighbors, tail_neighbors, head_timestamps, tail_timestamps = self.get_neighbors(
            node, current_t, self.threhold
        )
        if len(head_neighbors) == 0:
            all_neighbors = tail_neighbors.astype(np.int64, copy=False)
        elif len(tail_neighbors) == 0:
            all_neighbors = head_neighbors.astype(np.int64, copy=False)
        else:
            all_neighbors = np.concatenate((head_neighbors, tail_neighbors)).astype(np.int64, copy=False)

        if len(all_neighbors) == 0:
            dist = np.array([0.0], dtype=np.float32)
            if cache is not None:
                cache[cache_key] = dist
            return dist

        dist = self._degree_array[all_neighbors].astype(np.float32, copy=False)
        if cache is not None:
            cache[cache_key] = dist
        return dist

    def compute_wasserstein_score(self, source_node, candidate_nodes, current_t):
        source_dist = self.compute_temporal_distribution(source_node, current_t)
        scores = []
        score_cache = getattr(self, '_wasserstein_score_cache', None)
        t_key = self._time_value(current_t)
        for candidate in candidate_nodes:
            cache_key = None
            if score_cache is not None:
                cache_key = (int(source_node), int(candidate), t_key)
                cached = score_cache.get(cache_key)
                if cached is not None:
                    scores.append(cached)
                    continue
            candidate_dist = self.compute_temporal_distribution(candidate, current_t)
            try:
                distance = self.fast_wasserstein_1d(source_dist, candidate_dist)
                similarity = 1.0 / (1.0 + distance)
            except:
                similarity = 0.5
            if score_cache is not None:
                score_cache[cache_key] = similarity
            scores.append(similarity)

        return np.array(scores)

    def compute_mh_score(self, candidate_nodes, initial_node=None):
        """Compute the MH acceptance term in Eq. (17) over the candidate state space.

        The Markov chain is constructed on the second-order candidate set. With the
        symmetric proposal used in the paper, the proposal terms cancel and the
        acceptance probability for a proposed candidate w is:

            min(1, deg(w^(i-1)) / deg(w)).

        If an initial state is not explicitly provided inside the candidate set, the
        first candidate is used as w^(0), keeping the chain within N_temp^(2)(u, t).
        """
        num_candidates = len(candidate_nodes)
        if num_candidates == 0:
            return np.array([])
        candidate_list = [int(candidate) for candidate in candidate_nodes]
        candidates = np.asarray(candidate_list, dtype=np.int64)
        degrees = self._degree_array[candidates].astype(np.float32, copy=False)
        degrees = np.maximum(degrees, 1.0)

        initial_index = 0
        if initial_node is not None:
            try:
                initial_index = candidate_list.index(int(initial_node))
            except ValueError:
                initial_index = 0

        mh_scores = np.empty(num_candidates, dtype=np.float32)
        current_degree = float(degrees[initial_index])
        for idx, proposal_degree in enumerate(degrees):
            acceptance_prob = min(1.0, current_degree / float(proposal_degree))
            mh_scores[idx] = acceptance_prob
            if random.random() <= acceptance_prob:
                current_degree = float(proposal_degree)
        return mh_scores

    def update_interaction_cache(self, head, tail, t):
        t_val = self._time_value(t)
        if self.is_undirected:
            head, tail = self.canonical_endpoints(head, tail)
            if head == tail:
                is_new = tail not in self.adj_out[head]
                self.adj_out[head][tail] = t_val
                self.adj_in[head][tail] = t_val
                if is_new:
                    self._degree_array[head] += 1.0
                self._adj_out_version[head] += 1
                self._adj_in_version[head] += 1
                self._adj_out_arrays[head] = None
                self._adj_in_arrays[head] = None
                return

            head_new = tail not in self.adj_out[head]
            tail_new = head not in self.adj_out[tail]
            # Mirror the physical contact in both legacy adjacency views.  No
            # second temporal event is created and the event is still counted
            # exactly once by the training loop.
            self.adj_out[head][tail] = t_val
            self.adj_out[tail][head] = t_val
            self.adj_in[head][tail] = t_val
            self.adj_in[tail][head] = t_val
            if head_new:
                self._degree_array[head] += 1.0
            if tail_new:
                self._degree_array[tail] += 1.0
            for node in (head, tail):
                self._adj_out_version[node] += 1
                self._adj_in_version[node] += 1
                self._adj_out_arrays[node] = None
                self._adj_in_arrays[node] = None
            return

        out_new = tail not in self.adj_out[head]
        in_new = head not in self.adj_in[tail]
        self.adj_out[head][tail] = t_val
        self.adj_in[tail][head] = t_val
        if out_new:
            self._degree_array[head] += 1.0
        if in_new:
            self._degree_array[tail] += 1.0
        self._adj_out_version[head] += 1
        self._adj_in_version[tail] += 1
        self._adj_out_arrays[head] = None
        self._adj_in_arrays[tail] = None

    def get_node_degree(self, node):
        node = int(node)
        degree_cache = getattr(self, '_degree_cache', None)
        if degree_cache is not None:
            cached = degree_cache.get(node)
            if cached is not None:
                return cached
        degree = float(self._degree_array[node])
        if degree_cache is not None:
            degree_cache[node] = degree
        return degree

    def compute_second_order_scores(self, bridge_node, origin_node, candidate_nodes, current_t):
        wasserstein_scores = self.compute_wasserstein_score(origin_node, candidate_nodes, current_t)
        mh_scores = self.compute_mh_score(candidate_nodes)
        return 0.7 * wasserstein_scores + 0.3 * mh_scores

    def select_optimal_second_order_nodes(self, bridge_node, origin_node, candidate_nodes, current_t, max_nodes=5):
        if len(candidate_nodes) <= max_nodes:
            return candidate_nodes
        profile_start = time.time() if self.profile_runtime else None
        combined_scores = self.compute_second_order_scores(bridge_node, origin_node, candidate_nodes, current_t)
        top_indices = np.argpartition(combined_scores, -max_nodes)[-max_nodes:]
        selected_nodes = [candidate_nodes[i] for i in top_indices]
        if profile_start is not None:
            self.add_profile_time('second_order_select', time.time() - profile_start)

        return selected_nodes

    def select_union_records(self, records, max_nodes):
        if len(records) <= max_nodes:
            return list(records.values())
        items = list(records.values())
        scores = np.fromiter((item['score'] for item in items), dtype=np.float32)
        top_indices = np.argpartition(scores, -max_nodes)[-max_nodes:]
        return [items[i] for i in top_indices]

    def add_union_candidate_records(self, records, bridge_node, origin_node, neighbors, timestamps, current_t,
                                    bridge_index):
        if len(neighbors) == 0:
            return
        candidates = [int(n) for n in neighbors]
        scores = self.compute_second_order_scores(bridge_node, origin_node, candidates, current_t)
        for idx, candidate in enumerate(candidates):
            score = float(scores[idx])
            old_record = records.get(candidate)
            if old_record is None or score > old_record['score']:
                records[candidate] = {
                    'node': candidate,
                    'timestamp': float(timestamps[idx]),
                    'bridge_index': bridge_index,
                    'score': score,
                }

    def collect_union_candidate_records(self, records, bridge_node, neighbors, timestamps, bridge_index):
        for idx, candidate in enumerate(neighbors):
            candidate = int(candidate)
            timestamp = float(timestamps[idx])
            old_record = records.get(candidate)
            if old_record is None or timestamp > old_record['timestamp']:
                records[candidate] = {
                    'node': candidate,
                    'timestamp': timestamp,
                    'bridge_node': int(bridge_node),
                    'bridge_index': bridge_index,
                }

    def score_union_candidate_records(self, records, origin_node, current_t):
        if len(records) == 0:
            return {}
        items = list(records.values())
        candidates = [item['node'] for item in items]
        wasserstein_scores = self.compute_wasserstein_score(origin_node, candidates, current_t)
        mh_scores = self.compute_mh_score(candidates)
        combined_scores = 0.7 * wasserstein_scores + 0.3 * mh_scores
        for idx, item in enumerate(items):
            item['score'] = float(combined_scores[idx])
        return {item['node']: item for item in items}

    def filter_second_order_neighbors(self, bridge_node, origin_node, neighbors, timestamps, current_t):
        if len(neighbors) <= self.second_order_max_nodes:
            return neighbors, timestamps
        selected_neighbors = set(self.select_optimal_second_order_nodes(
            bridge_node=bridge_node, origin_node=origin_node, candidate_nodes=neighbors,
            current_t=current_t, max_nodes=self.second_order_max_nodes
        ))
        filtered_neighbors = []
        filtered_timestamps = []
        for idx, neighbor in enumerate(neighbors):
            if neighbor in selected_neighbors:
                filtered_neighbors.append(neighbor)
                filtered_timestamps.append(timestamps[idx])
        return filtered_neighbors, filtered_timestamps

    def filter_second_order_neighbors_by_direction(self, bridge_node, origin_node, head_neighbors, head_timestamps,
                                                   tail_neighbors, tail_timestamps, current_t):
        all_candidates = list(dict.fromkeys([int(n) for n in head_neighbors] + [int(n) for n in tail_neighbors]))
        self.add_profile_count('second_order_candidate_nodes', len(all_candidates))
        if len(all_candidates) <= self.second_order_max_nodes:
            self.add_profile_count('second_order_selected_nodes', len(all_candidates))
            return list(head_neighbors), list(head_timestamps), list(tail_neighbors), list(tail_timestamps)

        selected_neighbors = set(self.select_optimal_second_order_nodes(
            bridge_node=bridge_node,
            origin_node=origin_node,
            candidate_nodes=all_candidates,
            current_t=current_t,
            max_nodes=self.second_order_max_nodes
        ))
        self.add_profile_count('second_order_selected_nodes', len(selected_neighbors))

        filtered_head_neighbors = []
        filtered_head_timestamps = []
        for idx, neighbor in enumerate(head_neighbors):
            if int(neighbor) in selected_neighbors:
                filtered_head_neighbors.append(int(neighbor))
                filtered_head_timestamps.append(head_timestamps[idx])

        filtered_tail_neighbors = []
        filtered_tail_timestamps = []
        for idx, neighbor in enumerate(tail_neighbors):
            if int(neighbor) in selected_neighbors:
                filtered_tail_neighbors.append(int(neighbor))
                filtered_tail_timestamps.append(tail_timestamps[idx])

        return filtered_head_neighbors, filtered_head_timestamps, filtered_tail_neighbors, filtered_tail_timestamps

    def get_enhanced_att_score(self, node, neighbors, node2rep, attention_type='default', temporal_weights=None,
                               center_node_rep=None):
        if len(neighbors) == 0:
            return torch.zeros(0, 1).to(self.device)
        if len(neighbors) == 1:
            return torch.ones(1, 1, device=self.device)

        nei_reps = self.get_rep(neighbors, 'node_rep', node2rep)

        if center_node_rep is not None:
            node_rep = center_node_rep
        else:
            node_rep = self.get_rep([node], 'node_rep', node2rep)

        if self.use_enhanced_gat and hasattr(self, 'head_attention'):
            if attention_type == 'head':
                attention_scores = self.head_attention(node_rep, nei_reps, temporal_weights)
            elif attention_type == 'tail':
                attention_scores = self.tail_attention(node_rep, nei_reps, temporal_weights)
            else:
                attention_scores = self.attention(node_rep, nei_reps, temporal_weights)
        else:
            attention_scores = self.attention(node_rep, nei_reps)

        return attention_scores

    def apply_union_second_order_records(self, records, direction, origin_node, current_t, bridge_edge_infos,
                                         node_type, node2cell_head, node2hidden_head, node2cell_tail,
                                         node2hidden_tail, node2rep):
        if len(records) == 0:
            return

        candidates = [record['node'] for record in records]
        timestamps = [record['timestamp'] for record in records]
        bridge_indices = [record['bridge_index'] for record in records]
        selected_edge_infos = bridge_edge_infos[bridge_indices]

        timestamp_tensor = torch.as_tensor(timestamps, dtype=torch.float32, device=self.device).view(-1, 1)
        delta_ts = current_t - timestamp_tensor
        transed_delta_ts = self.decayer(delta_ts)

        if direction == 'head':
            if node_type == 'head':
                nei_edge_info = self.tran_head_edge_head(selected_edge_infos)
            else:
                nei_edge_info = self.tran_tail_edge_head(selected_edge_infos)
            nei_cell = self.get_rep(candidates, 'cell_head', node2cell_head)
        else:
            if node_type == 'head':
                nei_edge_info = self.tran_head_edge_tail(selected_edge_infos)
            else:
                nei_edge_info = self.tran_tail_edge_tail(selected_edge_infos)
            nei_cell = self.get_rep(candidates, 'cell_tail', node2cell_tail)

        if not self.if_no_time:
            nei_edge_info = nei_edge_info * transed_delta_ts

        if self.is_att:
            temporal_weights = None
            if self.use_enhanced_gat and not self.if_no_time:
                temporal_weights = transed_delta_ts * self.temporal_attention_scale
            att_score = self.get_enhanced_att_score(
                origin_node, candidates, node2rep,
                attention_type=direction,
                temporal_weights=temporal_weights
            )
            nei_edge_info = nei_edge_info * att_score

        nei_cell = nei_cell + nei_edge_info
        nei_hidden = self.act(nei_cell)

        if direction == 'head':
            if self.is_undirected:
                nei_rep = self.combiner(nei_hidden, nei_hidden)
            else:
                other_hidden = self.get_rep(candidates, 'hidden_tail', node2hidden_tail)
                nei_rep = self.combiner(nei_hidden, other_hidden)
            for idx, nei in enumerate(candidates):
                node2cell_head[nei] = nei_cell[idx].view(-1, self.embedding_dims)
                node2hidden_head[nei] = nei_hidden[idx].view(-1, self.embedding_dims)
                if self.is_undirected:
                    node2cell_tail[nei] = nei_cell[idx].view(-1, self.embedding_dims)
                    node2hidden_tail[nei] = nei_hidden[idx].view(-1, self.embedding_dims)
                node2rep[nei] = nei_rep[idx].view(-1, self.embedding_dims)
        else:
            other_hidden = self.get_rep(candidates, 'hidden_head', node2hidden_head)
            nei_rep = self.combiner(other_hidden, nei_hidden)
            for idx, nei in enumerate(candidates):
                node2cell_tail[nei] = nei_cell[idx].view(-1, self.embedding_dims)
                node2hidden_tail[nei] = nei_hidden[idx].view(-1, self.embedding_dims)
                node2rep[nei] = nei_rep[idx].view(-1, self.embedding_dims)

    def second_order_union_propagation(self, bridge_nodes, origin_node, current_t, bridge_edge_infos,
                                       node_type, node2cell_head, node2hidden_head, node2cell_tail,
                                       node2hidden_tail, node2rep):
        if len(bridge_nodes) == 0:
            return
        profile_start = time.time() if self.profile_runtime else None
        head_records = {}
        tail_records = {}

        for bridge_index, bridge_node in enumerate(bridge_nodes):
            head_neighbors, tail_neighbors, head_timestamps, tail_timestamps = self.get_neighbors(
                bridge_node, current_t, self.threhold_2nd
            )
            if len(head_neighbors) > 0:
                self.collect_union_candidate_records(
                    head_records, bridge_node, head_neighbors, head_timestamps, bridge_index
                )
            if len(tail_neighbors) > 0:
                self.collect_union_candidate_records(
                    tail_records, bridge_node, tail_neighbors, tail_timestamps, bridge_index
                )

        head_records = self.score_union_candidate_records(head_records, origin_node, current_t)
        tail_records = self.score_union_candidate_records(tail_records, origin_node, current_t)
        selected_head_records = self.select_union_records(head_records, self.second_order_max_nodes)
        selected_tail_records = self.select_union_records(tail_records, self.second_order_max_nodes)
        self.add_profile_count('second_order_union_head_candidates', len(head_records))
        self.add_profile_count('second_order_union_tail_candidates', len(tail_records))
        self.add_profile_count('second_order_union_used',
                               len(selected_head_records) + len(selected_tail_records))

        update_start = time.time() if self.profile_runtime else None
        self.apply_union_second_order_records(
            selected_head_records, 'head', origin_node, current_t, bridge_edge_infos,
            node_type, node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep
        )
        self.apply_union_second_order_records(
            selected_tail_records, 'tail', origin_node, current_t, bridge_edge_infos,
            node_type, node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep
        )
        if update_start is not None:
            self.add_profile_time('second_order_update', time.time() - update_start)
        if profile_start is not None:
            self.add_profile_time('second_order_total', time.time() - profile_start)

    def collect_second_order_records_for_bridges(self, bridge_nodes, origin_node, current_t):
        head_records = []
        tail_records = []
        for bridge_index, bridge_node in enumerate(bridge_nodes):
            head_neighbors, tail_neighbors, head_timestamps, tail_timestamps = self.get_neighbors(
                bridge_node, current_t, self.threhold_2nd
            )
            head_neighbors = list(head_neighbors)
            head_timestamps = list(head_timestamps)
            tail_neighbors = list(tail_neighbors)
            tail_timestamps = list(tail_timestamps)

            if len(head_neighbors) > 0 or len(tail_neighbors) > 0:
                head_neighbors, head_timestamps, tail_neighbors, tail_timestamps = (
                    self.filter_second_order_neighbors_by_direction(
                        bridge_node, origin_node, head_neighbors, head_timestamps,
                        tail_neighbors, tail_timestamps, current_t
                    )
                )

            for idx, candidate in enumerate(head_neighbors):
                head_records.append({
                    'node': int(candidate),
                    'timestamp': head_timestamps[idx],
                    'bridge_index': bridge_index,
                })
            for idx, candidate in enumerate(tail_neighbors):
                tail_records.append({
                    'node': int(candidate),
                    'timestamp': tail_timestamps[idx],
                    'bridge_index': bridge_index,
                })
        return head_records, tail_records

    def compute_grouped_second_order_attention(self, records, bridge_nodes, bridge_reps, node2rep,
                                               temporal_weights, attention_type):
        if len(records) == 0:
            return torch.zeros(0, 1, device=self.device)

        if self.use_enhanced_gat and hasattr(self, 'head_attention'):
            if attention_type == 'head':
                attention_module = self.head_attention
            elif attention_type == 'tail':
                attention_module = self.tail_attention
            else:
                attention_module = self.attention
        else:
            attention_module = self.attention

        group_bridge_indices = []
        group_starts = []
        group_lengths = []
        start = 0
        while start < len(records):
            bridge_index = records[start]['bridge_index']
            end = start + 1
            while end < len(records) and records[end]['bridge_index'] == bridge_index:
                end += 1
            group_bridge_indices.append(bridge_index)
            group_starts.append(start)
            group_lengths.append(end - start)
            start = end

        max_len = max(group_lengths)
        if max_len == 1:
            self.add_profile_count('second_order_singleton_attention', len(records))
            return torch.ones(len(records), 1, device=self.device)

        num_groups = len(group_lengths)
        padded_reps = torch.zeros(num_groups, max_len, self.embedding_dims, device=self.device)
        mask = torch.zeros(num_groups, max_len, dtype=torch.bool, device=self.device)
        padded_temporal = None
        if temporal_weights is not None:
            padded_temporal = torch.zeros(num_groups, max_len, 1, device=self.device)

        flat_nodes = [record['node'] for record in records]
        flat_reps = self.get_rep(flat_nodes, 'node_rep', node2rep)
        for group_idx, (group_start, group_len) in enumerate(zip(group_starts, group_lengths)):
            padded_reps[group_idx, :group_len] = flat_reps[group_start:group_start + group_len]
            mask[group_idx, :group_len] = True
            if padded_temporal is not None:
                padded_temporal[group_idx, :group_len] = temporal_weights[group_start:group_start + group_len]

        center_reps = bridge_reps[
            torch.as_tensor(group_bridge_indices, dtype=torch.long, device=self.device)
        ]
        center_q = attention_module.W_q(center_reps).view(num_groups, attention_module.num_heads,
                                                          attention_module.head_dim)
        neighbor_k = attention_module.W_k(padded_reps).view(num_groups, max_len, attention_module.num_heads,
                                                            attention_module.head_dim)
        center_q = center_q.unsqueeze(1).expand(-1, max_len, -1, -1)
        attention_input = torch.cat([center_q, neighbor_k], dim=3)
        scores = (attention_input * attention_module.attention_weight.view(
            1, 1, attention_module.num_heads, -1
        )).sum(dim=3)
        scores = attention_module.leaky_relu(scores).mean(dim=2)
        if padded_temporal is not None:
            scores = scores * (1 + attention_module.temporal_weight * padded_temporal.squeeze(-1))
        scores = scores.masked_fill(~mask, -1e9)
        attention = torch.softmax(scores, dim=1).unsqueeze(-1)
        attention = attention_module.dropout_layer(attention)

        output = torch.empty(len(records), 1, device=self.device)
        for group_idx, (group_start, group_len) in enumerate(zip(group_starts, group_lengths)):
            output[group_start:group_start + group_len] = attention[group_idx, :group_len]
        return output

    def apply_batched_second_order_records(self, records, direction, bridge_nodes, current_t, bridge_edge_infos,
                                           bridge_reps, node2cell_head, node2hidden_head, node2cell_tail,
                                           node2hidden_tail, node2rep):
        if len(records) == 0:
            return
        update_start = time.time() if self.profile_runtime else None
        candidates = [record['node'] for record in records]
        bridge_indices = torch.as_tensor(
            [record['bridge_index'] for record in records], dtype=torch.long, device=self.device
        )
        timestamps = torch.as_tensor(
            [record['timestamp'] for record in records], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        selected_edge_infos = bridge_edge_infos.index_select(0, bridge_indices)
        delta_ts = current_t - timestamps
        transed_delta_ts = self.decayer(delta_ts)

        if not self.if_no_time:
            selected_edge_infos = selected_edge_infos * transed_delta_ts

        if self.is_att:
            att_start = time.time() if self.profile_runtime else None
            temporal_weights = None
            if self.use_enhanced_gat and not self.if_no_time:
                temporal_weights = transed_delta_ts * self.temporal_attention_scale
            att_score = self.compute_grouped_second_order_attention(
                records, bridge_nodes, bridge_reps, node2rep, temporal_weights, direction
            )
            selected_edge_infos = selected_edge_infos * att_score
            if att_start is not None:
                self.add_profile_time('second_order_attention', time.time() - att_start)

        unique_candidates, inverse_indices = np.unique(np.asarray(candidates, dtype=np.int64), return_inverse=True)
        inverse_tensor = torch.as_tensor(inverse_indices, dtype=torch.long, device=self.device)
        aggregated_edge_infos = torch.zeros(
            (len(unique_candidates), self.embedding_dims), dtype=selected_edge_infos.dtype, device=self.device
        )
        aggregated_edge_infos.index_add_(0, inverse_tensor, selected_edge_infos)
        self.add_profile_count('second_order_unique_update_nodes', len(unique_candidates))

        candidates = unique_candidates.tolist()
        if direction == 'head':
            nei_cell = self.get_rep(candidates, 'cell_head', node2cell_head)
        else:
            nei_cell = self.get_rep(candidates, 'cell_tail', node2cell_tail)

        nei_cell = nei_cell + aggregated_edge_infos
        nei_hidden = self.act(nei_cell)

        if direction == 'head':
            if self.is_undirected:
                nei_rep = self.combiner(nei_hidden, nei_hidden)
            else:
                other_hidden = self.get_rep(candidates, 'hidden_tail', node2hidden_tail)
                nei_rep = self.combiner(nei_hidden, other_hidden)
            for idx, nei in enumerate(candidates):
                node2cell_head[nei] = nei_cell[idx].view(-1, self.embedding_dims)
                node2hidden_head[nei] = nei_hidden[idx].view(-1, self.embedding_dims)
                if self.is_undirected:
                    node2cell_tail[nei] = nei_cell[idx].view(-1, self.embedding_dims)
                    node2hidden_tail[nei] = nei_hidden[idx].view(-1, self.embedding_dims)
                node2rep[nei] = nei_rep[idx].view(-1, self.embedding_dims)
        else:
            other_hidden = self.get_rep(candidates, 'hidden_head', node2hidden_head)
            nei_rep = self.combiner(other_hidden, nei_hidden)
            for idx, nei in enumerate(candidates):
                node2cell_tail[nei] = nei_cell[idx].view(-1, self.embedding_dims)
                node2hidden_tail[nei] = nei_hidden[idx].view(-1, self.embedding_dims)
                node2rep[nei] = nei_rep[idx].view(-1, self.embedding_dims)

        if update_start is not None:
            self.add_profile_time('second_order_update', time.time() - update_start)

    def second_propagation_batch(self, bridge_nodes, origin_node, current_t, head_edge_infos, tail_edge_infos,
                                 bridge_reps, node2cell_head, node2hidden_head, node2cell_tail,
                                 node2hidden_tail, node2rep):
        if len(bridge_nodes) == 0:
            return
        profile_start = time.time() if self.profile_runtime else None
        head_records, tail_records = self.collect_second_order_records_for_bridges(
            bridge_nodes, origin_node, current_t
        )
        self.apply_batched_second_order_records(
            head_records, 'head', bridge_nodes, current_t, head_edge_infos, bridge_reps,
            node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep
        )
        self.apply_batched_second_order_records(
            tail_records, 'tail', bridge_nodes, current_t, tail_edge_infos, bridge_reps,
            node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep
        )
        if profile_start is not None:
            self.add_profile_time('second_order_total', time.time() - profile_start)

    def apply_batched_second_order_records_undirected(
            self, records, bridge_nodes, current_t, head_edge_infos, tail_edge_infos,
            bridge_reps, node2cell_head, node2hidden_head, node2cell_tail,
            node2hidden_tail, node2rep):
        """Update both latent feature channels from one undirected candidate set."""
        if len(records) == 0:
            return
        update_start = time.time() if self.profile_runtime else None
        candidates = [record['node'] for record in records]
        bridge_indices = torch.as_tensor(
            [record['bridge_index'] for record in records], dtype=torch.long, device=self.device
        )
        timestamps = torch.as_tensor(
            [record['timestamp'] for record in records], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        selected_head_infos = head_edge_infos.index_select(0, bridge_indices)
        selected_tail_infos = tail_edge_infos.index_select(0, bridge_indices)
        transed_delta_ts = self.decayer(current_t - timestamps)
        if not self.if_no_time:
            selected_head_infos = selected_head_infos * transed_delta_ts
            selected_tail_infos = selected_tail_infos * transed_delta_ts

        if self.is_att:
            att_start = time.time() if self.profile_runtime else None
            temporal_weights = None
            if self.use_enhanced_gat and not self.if_no_time:
                temporal_weights = transed_delta_ts * self.temporal_attention_scale
            head_att = self.compute_grouped_second_order_attention(
                records, bridge_nodes, bridge_reps, node2rep, temporal_weights, 'head'
            )
            tail_att = self.compute_grouped_second_order_attention(
                records, bridge_nodes, bridge_reps, node2rep, temporal_weights, 'tail'
            )
            selected_head_infos = selected_head_infos * head_att
            selected_tail_infos = selected_tail_infos * tail_att
            if att_start is not None:
                self.add_profile_time('second_order_attention', time.time() - att_start)

        unique_candidates, inverse_indices = np.unique(
            np.asarray(candidates, dtype=np.int64), return_inverse=True
        )
        inverse_tensor = torch.as_tensor(inverse_indices, dtype=torch.long, device=self.device)
        aggregated_head = torch.zeros(
            (len(unique_candidates), self.embedding_dims),
            dtype=selected_head_infos.dtype, device=self.device
        )
        aggregated_tail = torch.zeros_like(aggregated_head)
        aggregated_head.index_add_(0, inverse_tensor, selected_head_infos)
        aggregated_tail.index_add_(0, inverse_tensor, selected_tail_infos)
        self.add_profile_count('second_order_unique_update_nodes', len(unique_candidates))

        candidates = unique_candidates.tolist()
        head_cell = self.get_rep(candidates, 'cell_head', node2cell_head) + aggregated_head
        tail_cell = self.get_rep(candidates, 'cell_tail', node2cell_tail) + aggregated_tail
        head_hidden = self.act(head_cell)
        tail_hidden = self.act(tail_cell)
        if self.is_att and self.use_enhanced_gat and hasattr(self, 'attention_fusion'):
            candidate_rep = self.attention_fusion(torch.cat([head_hidden, tail_hidden], dim=1))
        else:
            candidate_rep = self.combiner(head_hidden, tail_hidden)

        for idx, candidate in enumerate(candidates):
            node2cell_head[candidate] = head_cell[idx].view(-1, self.embedding_dims)
            node2hidden_head[candidate] = head_hidden[idx].view(-1, self.embedding_dims)
            node2cell_tail[candidate] = tail_cell[idx].view(-1, self.embedding_dims)
            node2hidden_tail[candidate] = tail_hidden[idx].view(-1, self.embedding_dims)
            node2rep[candidate] = candidate_rep[idx].view(-1, self.embedding_dims)

        if update_start is not None:
            self.add_profile_time('second_order_update', time.time() - update_start)

    def second_order_propagation_undirected(
            self, bridge_nodes, origin_node, current_t, head_edge_infos, tail_edge_infos,
            bridge_reps, node2cell_head, node2hidden_head, node2cell_tail,
            node2hidden_tail, node2rep):
        if len(bridge_nodes) == 0:
            return
        profile_start = time.time() if self.profile_runtime else None

        if self.second_order_mode == 'event_union':
            union_records = {}
            for bridge_index, bridge_node in enumerate(bridge_nodes):
                neighbors, _, timestamps, _ = self.get_neighbors(
                    bridge_node, current_t, self.threhold_2nd
                )
                self.collect_union_candidate_records(
                    union_records, bridge_node, neighbors, timestamps, bridge_index
                )
            union_records = self.score_union_candidate_records(
                union_records, origin_node, current_t
            )
            records = self.select_union_records(union_records, self.second_order_max_nodes)
            self.add_profile_count('second_order_union_head_candidates', len(union_records))
            self.add_profile_count('second_order_union_tail_candidates', 0)
            self.add_profile_count('second_order_union_used', len(records))
        else:
            records, _ = self.collect_second_order_records_for_bridges(
                bridge_nodes, origin_node, current_t
            )

        self.apply_batched_second_order_records_undirected(
            records, bridge_nodes, current_t, head_edge_infos, tail_edge_infos,
            bridge_reps, node2cell_head, node2hidden_head, node2cell_tail,
            node2hidden_tail, node2rep
        )
        if profile_start is not None:
            self.add_profile_time('second_order_total', time.time() - profile_start)

    def propagation_undirected(
            self, node, current_t, edge_infos, node2cell_head, node2hidden_head,
            node2cell_tail, node2hidden_tail, node2rep, threhold=None,
            second_order=False):
        """Propagate one unordered event through both latent feature channels."""
        neighbors, _, timestamps, _ = self.get_neighbors(node, current_t, threhold)
        neighbors = list(neighbors)
        timestamps = list(timestamps)
        if len(neighbors) == 0:
            return set(), set()

        edge_info_head, edge_info_tail = edge_infos
        head_message = 0.5 * (
            self.tran_head_edge_head(edge_info_head)
            + self.tran_tail_edge_head(edge_info_tail)
        )
        tail_message = 0.5 * (
            self.tran_head_edge_tail(edge_info_head)
            + self.tran_tail_edge_tail(edge_info_tail)
        )
        timestamp_tensor = torch.as_tensor(
            timestamps, dtype=torch.float32, device=self.device
        ).view(-1, 1)
        transed_delta_ts = self.decayer(current_t - timestamp_tensor)
        head_messages = head_message.expand(len(neighbors), -1)
        tail_messages = tail_message.expand(len(neighbors), -1)
        if not self.if_no_time:
            head_messages = head_messages * transed_delta_ts
            tail_messages = tail_messages * transed_delta_ts

        head_att = None
        tail_att = None
        if self.is_att:
            temporal_weights = None
            if self.use_enhanced_gat and not self.if_no_time:
                temporal_weights = transed_delta_ts * self.temporal_attention_scale
            head_att = self.get_enhanced_att_score(
                node, neighbors, node2rep, attention_type='head',
                temporal_weights=temporal_weights
            )
            tail_att = self.get_enhanced_att_score(
                node, neighbors, node2rep, attention_type='tail',
                temporal_weights=temporal_weights
            )
            head_messages = head_messages * head_att
            tail_messages = tail_messages * tail_att

        if self.is_att and head_att is not None:
            bridge_scores = 0.5 * (head_att.detach().view(-1) + tail_att.detach().view(-1))
        elif self.if_no_time:
            bridge_scores = torch.ones(len(neighbors), device=self.device)
        else:
            bridge_scores = transed_delta_ts.detach().view(-1)
        selected_indices = self.select_second_order_bridges(
            bridge_scores, self.second_order_bridge_max_nodes
        )
        if second_order:
            self.add_profile_count('second_order_bridge_candidates', len(neighbors))
            self.add_profile_count(
                'second_order_bridge_used',
                len(selected_indices) if selected_indices is not None else len(neighbors)
            )

        old_bridge_reps = self.get_rep(neighbors, 'node_rep', node2rep)
        head_cell = self.get_rep(neighbors, 'cell_head', node2cell_head) + head_messages
        tail_cell = self.get_rep(neighbors, 'cell_tail', node2cell_tail) + tail_messages
        head_hidden = self.act(head_cell)
        tail_hidden = self.act(tail_cell)
        if self.is_att and self.use_enhanced_gat and hasattr(self, 'attention_fusion'):
            neighbor_rep = self.attention_fusion(torch.cat([head_hidden, tail_hidden], dim=1))
        else:
            neighbor_rep = self.combiner(head_hidden, tail_hidden)

        for idx, neighbor in enumerate(neighbors):
            node2cell_head[neighbor] = head_cell[idx].view(-1, self.embedding_dims)
            node2hidden_head[neighbor] = head_hidden[idx].view(-1, self.embedding_dims)
            node2cell_tail[neighbor] = tail_cell[idx].view(-1, self.embedding_dims)
            node2hidden_tail[neighbor] = tail_hidden[idx].view(-1, self.embedding_dims)
            node2rep[neighbor] = neighbor_rep[idx].view(-1, self.embedding_dims)

        if second_order:
            bridge_indices = (
                list(range(len(neighbors))) if selected_indices is None
                else sorted(selected_indices)
            )
            if len(bridge_indices) > 0:
                bridge_tensor = torch.as_tensor(
                    bridge_indices, dtype=torch.long, device=self.device
                )
                bridge_nodes = [neighbors[idx] for idx in bridge_indices]
                second_head_infos = 0.5 * (
                    self.tran_head_edge_head(head_messages)
                    + self.tran_tail_edge_head(tail_messages)
                )
                second_tail_infos = 0.5 * (
                    self.tran_head_edge_tail(head_messages)
                    + self.tran_tail_edge_tail(tail_messages)
                )
                self.second_order_propagation_undirected(
                    bridge_nodes, node, current_t,
                    second_head_infos.index_select(0, bridge_tensor),
                    second_tail_infos.index_select(0, bridge_tensor),
                    old_bridge_reps.index_select(0, bridge_tensor),
                    node2cell_head, node2hidden_head, node2cell_tail,
                    node2hidden_tail, node2rep
                )

        return set(neighbors), set()

    def propagation(self, node, current_t, edge_info, node_type, node2cell_head, node2hidden_head, node2cell_tail,
                    node2hidden_tail, node2rep, threhold=None, second_order=False):
        if self.is_undirected:
            return self.propagation_undirected(
                node, current_t, edge_info, node2cell_head, node2hidden_head,
                node2cell_tail, node2hidden_tail, node2rep, threhold, second_order
            )
        head_neighbors, tail_neighbors, head_timestamps, tail_timestamps = self.get_neighbors(node, current_t, threhold)
        head_neighbors = list(head_neighbors)
        head_timestamps = list(head_timestamps)

        if len(head_neighbors) > 0:
            att_score_head = None
            if node_type == 'head':
                head_nei_edge_info = self.tran_head_edge_head(edge_info)
            else:
                head_nei_edge_info = self.tran_tail_edge_head(edge_info)

            head_timestamp_tensor = torch.as_tensor(head_timestamps, dtype=torch.float32,
                                                     device=self.device).view(-1, 1)
            head_delta_ts = current_t - head_timestamp_tensor
            transed_head_delta_ts = self.decayer(head_delta_ts)
            head_nei_cell = self.get_rep(head_neighbors, 'cell_head', node2cell_head)

            if self.if_no_time:
                tran_head_nei_edge_info = head_nei_edge_info.expand(len(head_neighbors), -1)
            else:
                tran_head_nei_edge_info = head_nei_edge_info.expand(len(head_neighbors), -1) * transed_head_delta_ts

            if self.is_att:
                temporal_weights = None
                if self.use_enhanced_gat and not self.if_no_time:
                    temporal_weights = transed_head_delta_ts * self.temporal_attention_scale
                att_score_head = self.get_enhanced_att_score(
                    node, head_neighbors, node2rep,
                    attention_type='head',
                    temporal_weights=temporal_weights
                )
                tran_head_nei_edge_info = tran_head_nei_edge_info * att_score_head

            if self.is_att and att_score_head is not None:
                head_bridge_scores = att_score_head.detach().view(-1)
            elif self.if_no_time:
                head_bridge_scores = torch.ones(len(head_neighbors), device=self.device)
            else:
                head_bridge_scores = transed_head_delta_ts.detach().view(-1)
            selected_head_bridge_indices = self.select_second_order_bridges(
                head_bridge_scores, self.second_order_bridge_max_nodes
            )
            if second_order:
                self.add_profile_count('second_order_bridge_candidates', len(head_neighbors))
                self.add_profile_count('second_order_bridge_used',
                                       len(selected_head_bridge_indices)
                                       if selected_head_bridge_indices is not None else len(head_neighbors))

            head_nei_cell = head_nei_cell + tran_head_nei_edge_info
            head_nei_hidden = self.act(head_nei_cell)
            if self.is_undirected:
                head_nei_tail_hidden = head_nei_hidden
            else:
                head_nei_tail_hidden = self.get_rep(head_neighbors, 'hidden_tail', node2hidden_tail)

            if self.is_att and self.use_enhanced_gat and hasattr(self, 'attention_fusion'):
                combined_features = torch.cat([head_nei_hidden, head_nei_tail_hidden], dim=1)
                head_nei_rep = self.attention_fusion(combined_features)
            else:
                head_nei_rep = self.combiner(head_nei_hidden, head_nei_tail_hidden)

            head_nei_reps_old = None
            second_head_edge_infos = None
            second_tail_edge_infos = None
            if second_order and self.second_order_mode == 'per_bridge':
                head_nei_reps_old = self.get_rep(head_neighbors, 'node_rep', node2rep)
                if node_type == 'head':
                    second_head_edge_infos = self.tran_head_edge_head(tran_head_nei_edge_info)
                    second_tail_edge_infos = self.tran_head_edge_tail(tran_head_nei_edge_info)
                else:
                    second_head_edge_infos = self.tran_tail_edge_head(tran_head_nei_edge_info)
                    second_tail_edge_infos = self.tran_tail_edge_tail(tran_head_nei_edge_info)

            for i, nei in enumerate(head_neighbors):
                node2cell_head[nei] = head_nei_cell[i].view(-1, self.embedding_dims)
                node2hidden_head[nei] = head_nei_hidden[i].view(-1, self.embedding_dims)
                if self.is_undirected:
                    node2cell_tail[nei] = head_nei_cell[i].view(-1, self.embedding_dims)
                    node2hidden_tail[nei] = head_nei_hidden[i].view(-1, self.embedding_dims)
                node2rep[nei] = head_nei_rep[i].view(-1, self.embedding_dims)
            if second_order and self.second_order_mode == 'per_bridge':
                if selected_head_bridge_indices is None:
                    bridge_indices = list(range(len(head_neighbors)))
                else:
                    bridge_indices = sorted(selected_head_bridge_indices)
                if len(bridge_indices) > 0:
                    bridge_nodes = [head_neighbors[i] for i in bridge_indices]
                    bridge_index_tensor = torch.as_tensor(bridge_indices, dtype=torch.long, device=self.device)
                    self.second_propagation_batch(
                        bridge_nodes, node, current_t,
                        second_head_edge_infos.index_select(0, bridge_index_tensor),
                        second_tail_edge_infos.index_select(0, bridge_index_tensor),
                        head_nei_reps_old.index_select(0, bridge_index_tensor),
                        node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep
                    )
            if second_order and self.second_order_mode == 'event_union':
                self.second_order_union_propagation(
                    head_neighbors, node, current_t, tran_head_nei_edge_info,
                    'head', node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep
                )

        tail_neighbors = list(tail_neighbors)
        tail_timestamps = list(tail_timestamps)

        if len(tail_neighbors) > 0:
            att_score_tail = None
            if node_type == 'head':
                tail_nei_edge_info = self.tran_head_edge_tail(edge_info)
            else:
                tail_nei_edge_info = self.tran_tail_edge_tail(edge_info)

            tail_timestamp_tensor = torch.as_tensor(tail_timestamps, dtype=torch.float32,
                                                     device=self.device).view(-1, 1)
            tail_delta_ts = current_t - tail_timestamp_tensor
            transed_tail_delta_ts = self.decayer(tail_delta_ts)
            tail_nei_cell = self.get_rep(tail_neighbors, 'cell_tail', node2cell_tail)

            if self.if_no_time:
                tran_tail_nei_edge_info = tail_nei_edge_info.expand(len(tail_neighbors), -1)
            else:
                tran_tail_nei_edge_info = tail_nei_edge_info.expand(len(tail_neighbors), -1) * transed_tail_delta_ts

            if self.is_att:
                temporal_weights = None
                if self.use_enhanced_gat and not self.if_no_time:
                    temporal_weights = transed_tail_delta_ts * self.temporal_attention_scale
                att_score_tail = self.get_enhanced_att_score(
                    node, tail_neighbors, node2rep,
                    attention_type='tail',
                    temporal_weights=temporal_weights
                )
                tran_tail_nei_edge_info = tran_tail_nei_edge_info * att_score_tail

            if self.is_att and att_score_tail is not None:
                tail_bridge_scores = att_score_tail.detach().view(-1)
            elif self.if_no_time:
                tail_bridge_scores = torch.ones(len(tail_neighbors), device=self.device)
            else:
                tail_bridge_scores = transed_tail_delta_ts.detach().view(-1)
            selected_tail_bridge_indices = self.select_second_order_bridges(
                tail_bridge_scores, self.second_order_bridge_max_nodes
            )
            if second_order:
                self.add_profile_count('second_order_bridge_candidates', len(tail_neighbors))
                self.add_profile_count('second_order_bridge_used',
                                       len(selected_tail_bridge_indices)
                                       if selected_tail_bridge_indices is not None else len(tail_neighbors))

            tail_nei_cell = tail_nei_cell + tran_tail_nei_edge_info
            tail_nei_hidden = self.act(tail_nei_cell)
            tail_nei_head_hidden = self.get_rep(tail_neighbors, 'hidden_head', node2hidden_head)

            if self.is_att and self.use_enhanced_gat and hasattr(self, 'attention_fusion'):
                combined_features = torch.cat([tail_nei_head_hidden, tail_nei_hidden], dim=1)
                tail_nei_rep = self.attention_fusion(combined_features)
            else:
                tail_nei_rep = self.combiner(tail_nei_head_hidden, tail_nei_hidden)

            tail_nei_reps_old = None
            second_head_edge_infos = None
            second_tail_edge_infos = None
            if second_order and self.second_order_mode == 'per_bridge':
                tail_nei_reps_old = self.get_rep(tail_neighbors, 'node_rep', node2rep)
                if node_type == 'head':
                    second_head_edge_infos = self.tran_head_edge_head(tran_tail_nei_edge_info)
                    second_tail_edge_infos = self.tran_head_edge_tail(tran_tail_nei_edge_info)
                else:
                    second_head_edge_infos = self.tran_tail_edge_head(tran_tail_nei_edge_info)
                    second_tail_edge_infos = self.tran_tail_edge_tail(tran_tail_nei_edge_info)

            for i, nei in enumerate(tail_neighbors):
                node2cell_tail[nei] = tail_nei_cell[i].view(-1, self.embedding_dims)
                node2hidden_tail[nei] = tail_nei_hidden[i].view(-1, self.embedding_dims)
                node2rep[nei] = tail_nei_rep[i].view(-1, self.embedding_dims)

            if second_order and self.second_order_mode == 'per_bridge':
                if selected_tail_bridge_indices is None:
                    bridge_indices = list(range(len(tail_neighbors)))
                else:
                    bridge_indices = sorted(selected_tail_bridge_indices)
                if len(bridge_indices) > 0:
                    bridge_nodes = [tail_neighbors[i] for i in bridge_indices]
                    bridge_index_tensor = torch.as_tensor(bridge_indices, dtype=torch.long, device=self.device)
                    self.second_propagation_batch(
                        bridge_nodes, node, current_t,
                        second_head_edge_infos.index_select(0, bridge_index_tensor),
                        second_tail_edge_infos.index_select(0, bridge_index_tensor),
                        tail_nei_reps_old.index_select(0, bridge_index_tensor),
                        node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep
                    )
            if second_order and self.second_order_mode == 'event_union':
                self.second_order_union_propagation(
                    tail_neighbors, node, current_t, tran_tail_nei_edge_info,
                    'tail', node2cell_head, node2hidden_head, node2cell_tail, node2hidden_tail, node2rep
                )

        return set(head_neighbors), set(tail_neighbors)

    def forward(self, interactions, sample_negatives=True):
        all_head_nodes = set()
        all_tail_nodes = set()
        steps = len(interactions[:, 0])
        node2cell_head = dict()
        node2cell_tail = dict()
        node2hidden_head = dict()
        node2hidden_tail = dict()
        node2rep = dict()
        output_rep_head = []
        output_rep_tail = []
        tail_neg_list = []
        head_neg_list = []
        pos_prior_list = []
        head_neg_prior_list = []
        tail_neg_prior_list = []

        for i in range(steps):
            self._temporal_dist_cache = {}
            self._degree_cache = {}
            self._wasserstein_score_cache = {}
            self._neighbors_cache = {}
            head_index = int(interactions[i, 0])
            tail_index = int(interactions[i, 1])
            # The scoring representation for a negative edge must come from the
            # same causal instant as the positive edge: immediately before the
            # current event is applied.  A shallow copy is sufficient because
            # event updates replace dictionary tensors instead of mutating them.
            # Validation/replay calls forward(..., sample_negatives=False), so a
            # pre-event negative cache would never be consumed there.  Avoiding
            # that per-event dictionary copy materially reduces validation time
            # without changing any score or temporal state.
            pre_event_node2rep = (
                dict(node2rep)
                if sample_negatives and self.negative_rep_timing == 'pre_event'
                else None
            )
            head_index, tail_index = self.canonical_endpoints(head_index, tail_index)
            if self.is_undirected:
                all_head_nodes.update((head_index, tail_index))
                all_tail_nodes.update((head_index, tail_index))
            else:
                all_head_nodes.add(head_index)
                all_tail_nodes.add(tail_index)

            head_inx_lt = torch.as_tensor([head_index], dtype=torch.long, device=self.device)
            tail_inx_lt = torch.as_tensor([tail_index], dtype=torch.long, device=self.device)
            timestamp = interactions[i, 2]
            self._current_t_value = float(timestamp)
            current_t = torch.as_tensor([[timestamp]], dtype=torch.float32, device=self.device)

            head_prev_t = self.recent_timestamp[head_index]
            tail_prev_t = self.recent_timestamp[tail_index]

            if head_index in node2rep:
                head_node_rep = node2rep[head_index]
            else:
                head_node_rep = self.node_representations(head_inx_lt)

            if tail_index in node2rep:
                tail_node_rep = node2rep[tail_index]
            else:
                tail_node_rep = self.node_representations(tail_inx_lt)

            if head_index in node2hidden_head:
                head_node_cell_head = node2cell_head[head_index]
                head_node_hidden_head = node2hidden_head[head_index]
            else:
                head_node_cell_head = self.cell_head(head_inx_lt)
                head_node_hidden_head = self.hidden_head(head_inx_lt)

            if head_index in node2hidden_tail:
                if self.is_undirected:
                    head_node_cell_tail = node2cell_tail[head_index]
                head_node_hidden_tail = node2hidden_tail[head_index]
            else:
                if self.is_undirected:
                    head_node_cell_tail = self.cell_tail(head_inx_lt)
                head_node_hidden_tail = self.hidden_tail(head_inx_lt)

            if tail_index in node2hidden_tail:
                tail_node_cell_tail = node2cell_tail[tail_index]
                tail_node_hidden_tail = node2hidden_tail[tail_index]
            else:
                tail_node_cell_tail = self.cell_tail(tail_inx_lt)
                tail_node_hidden_tail = self.hidden_tail(tail_inx_lt)

            if tail_index in node2hidden_head:
                tail_node_hidden_head = node2hidden_head[tail_index]
                if self.is_undirected:
                    tail_node_cell_head = node2cell_head[tail_index]
            else:
                tail_node_hidden_head = self.hidden_head(tail_inx_lt)
                if self.is_undirected:
                    tail_node_cell_head = self.cell_head(tail_inx_lt)

            head_delta_t = current_t - head_prev_t
            tail_delta_t = current_t - tail_prev_t

            with torch.no_grad():
                self.recent_timestamp[[head_index, tail_index]] = current_t

            transed_head_delta_t = self.decayer(head_delta_t)
            transed_tail_delta_t = self.decayer(tail_delta_t)

            if self.is_undirected:
                # Evaluate the two exchanged endpoint views and aggregate them
                # inside one temporal event.  Every endpoint is updated once in
                # each latent feature channel from the same pre-event snapshot.
                # This is endpoint-swap equivariant without tying away half of
                # the original model parameters.
                head_as_head = self.edge_updater_head(head_node_rep, tail_node_rep)
                head_as_tail = self.edge_updater_tail(tail_node_rep, head_node_rep)
                tail_as_head = self.edge_updater_head(tail_node_rep, head_node_rep)
                tail_as_tail = self.edge_updater_tail(head_node_rep, tail_node_rep)

                if self.if_no_time:
                    updated_head_node_hidden_head, updated_head_node_cell_head = self.node_updater_head(
                        head_as_head, (head_node_hidden_head, head_node_cell_head)
                    )
                    updated_head_node_hidden_tail, updated_head_node_cell_tail = self.node_updater_tail(
                        head_as_tail, (head_node_hidden_tail, head_node_cell_tail)
                    )
                    updated_tail_node_hidden_head, updated_tail_node_cell_head = self.node_updater_head(
                        tail_as_head, (tail_node_hidden_head, tail_node_cell_head)
                    )
                    updated_tail_node_hidden_tail, updated_tail_node_cell_tail = self.node_updater_tail(
                        tail_as_tail, (tail_node_hidden_tail, tail_node_cell_tail)
                    )
                else:
                    updated_head_node_cell_head, updated_head_node_hidden_head = self.node_updater_head(
                        head_as_head, head_node_cell_head, head_node_hidden_head, transed_head_delta_t
                    )
                    updated_head_node_cell_tail, updated_head_node_hidden_tail = self.node_updater_tail(
                        head_as_tail, head_node_cell_tail, head_node_hidden_tail, transed_head_delta_t
                    )
                    updated_tail_node_cell_head, updated_tail_node_hidden_head = self.node_updater_head(
                        tail_as_head, tail_node_cell_head, tail_node_hidden_head, transed_tail_delta_t
                    )
                    updated_tail_node_cell_tail, updated_tail_node_hidden_tail = self.node_updater_tail(
                        tail_as_tail, tail_node_cell_tail, tail_node_hidden_tail, transed_tail_delta_t
                    )

                updated_head_node_rep = self.combiner(
                    updated_head_node_hidden_head, updated_head_node_hidden_tail
                )
                updated_tail_node_rep = self.combiner(
                    updated_tail_node_hidden_head, updated_tail_node_hidden_tail
                )

                for node, cell_h, hidden_h, cell_t, hidden_t, rep in (
                    (head_index, updated_head_node_cell_head, updated_head_node_hidden_head,
                     updated_head_node_cell_tail, updated_head_node_hidden_tail, updated_head_node_rep),
                    (tail_index, updated_tail_node_cell_head, updated_tail_node_hidden_head,
                     updated_tail_node_cell_tail, updated_tail_node_hidden_tail, updated_tail_node_rep),
                ):
                    node2cell_head[node] = cell_h
                    node2cell_tail[node] = cell_t
                    node2hidden_head[node] = hidden_h
                    node2hidden_tail[node] = hidden_t
                    node2rep[node] = rep

                edge_info_head = (head_as_head, head_as_tail)
                edge_info_tail = (tail_as_head, tail_as_tail)

                output_rep_head.append(updated_head_node_rep if self.if_updated else head_node_rep)
                output_rep_tail.append(updated_tail_node_rep if self.if_updated else tail_node_rep)
            else:
                edge_info_head = self.edge_updater_head(head_node_rep, tail_node_rep)
                edge_info_tail = self.edge_updater_tail(head_node_rep, tail_node_rep)

                if self.if_no_time:
                    updated_head_node_hidden_head, updated_head_node_cell_head = self.node_updater_head(edge_info_head, (
                        head_node_hidden_head, head_node_cell_head))
                else:
                    updated_head_node_cell_head, updated_head_node_hidden_head = self.node_updater_head(edge_info_head,
                                                                                                        head_node_cell_head,
                                                                                                        head_node_hidden_head,
                                                                                                        transed_head_delta_t)
                updated_head_node_rep = self.combiner(updated_head_node_hidden_head, head_node_hidden_tail)

                node2cell_head[head_index] = updated_head_node_cell_head
                node2hidden_head[head_index] = updated_head_node_hidden_head
                node2rep[head_index] = updated_head_node_rep

                if self.if_updated:
                    output_rep_head.append(updated_head_node_rep)
                else:
                    output_rep_head.append(head_node_rep)

                if self.if_no_time:
                    updated_tail_node_hidden_tail, updated_tail_node_cell_tail, = self.node_updater_tail(edge_info_tail, (
                        tail_node_hidden_tail, tail_node_cell_tail))
                else:
                    updated_tail_node_cell_tail, updated_tail_node_hidden_tail = self.node_updater_tail(edge_info_tail,
                                                                                                        tail_node_cell_tail,
                                                                                                        tail_node_hidden_tail,
                                                                                                        transed_tail_delta_t)
                updated_tail_node_rep = self.combiner(tail_node_hidden_head, updated_tail_node_hidden_tail)

                node2cell_tail[tail_index] = updated_tail_node_cell_tail
                node2hidden_tail[tail_index] = updated_tail_node_hidden_tail
                node2rep[tail_index] = updated_tail_node_rep

                if self.if_updated:
                    output_rep_tail.append(updated_tail_node_rep)
                else:
                    output_rep_tail.append(tail_node_rep)

            if self.if_propagation:
                head_node_head_neighbors, head_node_tail_neighbors = self.propagation(head_index, current_t,
                                                                                      edge_info_head, 'head',
                                                                                      node2cell_head, node2hidden_head,
                                                                                      node2cell_tail, node2hidden_tail,
                                                                                      node2rep, self.threhold,
                                                                                      self.second_order)
                tail_node_head_neighbors, tail_node_tail_neighbors = self.propagation(tail_index, current_t,
                                                                                      edge_info_tail, 'tail',
                                                                                      node2cell_head, node2hidden_head,
                                                                                      node2cell_tail, node2hidden_tail,
                                                                                      node2rep, self.threhold,
                                                                                      self.second_order)
            else:
                head_node_head_neighbors, head_node_tail_neighbors, _, _ = self.get_neighbors(head_index, current_t,
                                                                                              self.threhold)
                tail_node_head_neighbors, tail_node_tail_neighbors, _, _ = self.get_neighbors(tail_index, current_t,
                                                                                              self.threhold)
                head_node_head_neighbors = set(head_node_head_neighbors)
                head_node_tail_neighbors = set(head_node_tail_neighbors)
                tail_node_head_neighbors = set(tail_node_head_neighbors)
                tail_node_tail_neighbors = set(tail_node_tail_neighbors)

            if sample_negatives:
                if self.is_undirected:
                    common_nodes = (
                        all_head_nodes | all_tail_nodes
                        | head_node_head_neighbors | head_node_tail_neighbors
                        | tail_node_head_neighbors | tail_node_tail_neighbors
                    )
                    all_head_nodes = set(common_nodes)
                    all_tail_nodes = set(common_nodes)
                else:
                    all_head_nodes = all_head_nodes | head_node_head_neighbors | tail_node_head_neighbors
                    all_tail_nodes = all_tail_nodes | head_node_tail_neighbors | tail_node_tail_neighbors

                tail_forbidden = set(self.adj_out[head_index].keys()) | {head_index, tail_index}
                tail_candidates = all_tail_nodes - tail_forbidden - head_node_tail_neighbors

                tail_neg_samples = self.sample_negative_nodes(
                    head_index, tail_candidates, tail_forbidden, self.node_representations
                )

                head_forbidden = set(self.adj_in[tail_index].keys()) | {tail_index, head_index}
                head_candidates = all_head_nodes - head_forbidden - tail_node_head_neighbors

                head_neg_samples = self.sample_negative_nodes(
                    tail_index, head_candidates, head_forbidden, self.node_representations
                )

                if len(tail_neg_samples) < self.num_negative:
                    remaining = self.num_negative - len(tail_neg_samples)
                    tail_neg_samples.extend(self.sample_random_nodes(remaining, tail_forbidden | set(tail_neg_samples)))
                elif len(tail_neg_samples) > self.num_negative:
                    tail_neg_samples = tail_neg_samples[:self.num_negative]

                if len(head_neg_samples) < self.num_negative:
                    remaining = self.num_negative - len(head_neg_samples)
                    head_neg_samples.extend(self.sample_random_nodes(remaining, head_forbidden | set(head_neg_samples)))
                elif len(head_neg_samples) > self.num_negative:
                    head_neg_samples = head_neg_samples[:self.num_negative]

                if len(tail_neg_samples) > 0:
                    negative_rep_cache = pre_event_node2rep if pre_event_node2rep is not None else node2rep
                    tail_neg_tensor = self.get_rep(tail_neg_samples, 'node_rep', negative_rep_cache)
                    tail_neg_list.append(tail_neg_tensor)
                    tail_neg_prior_list.extend([
                        self.get_persistence_prior(head_index, neg_tail, current_t)
                        for neg_tail in tail_neg_samples
                    ])

                if len(head_neg_samples) > 0:
                    negative_rep_cache = pre_event_node2rep if pre_event_node2rep is not None else node2rep
                    head_neg_tensor = self.get_rep(head_neg_samples, 'node_rep', negative_rep_cache)
                    head_neg_list.append(head_neg_tensor)
                    head_neg_prior_list.extend([
                        self.get_persistence_prior(neg_head, tail_index, current_t)
                        for neg_head in head_neg_samples
                    ])

                pos_prior_list.append(self.get_persistence_prior(head_index, tail_index, current_t))

            self.update_interaction_cache(head_index, tail_index, current_t)
            self._temporal_dist_cache = None
            self._degree_cache = None
            self._wasserstein_score_cache = None
            self._neighbors_cache = None
            self._current_t_value = None

        cell_head_inx = list(node2cell_head.keys())
        output_cell_head = list(node2cell_head.values())
        cell_tail_inx = list(node2cell_tail.keys())
        output_cell_tail = list(node2cell_tail.values())
        hidden_head_inx = list(node2hidden_head.keys())
        output_hidden_head = list(node2hidden_head.values())
        hidden_tail_inx = list(node2hidden_tail.keys())
        output_hidden_tail = list(node2hidden_tail.values())
        rep_inx = list(node2rep.keys())
        output_rep = list(node2rep.values())

        output_cell_head_tensor = torch.cat([*output_cell_head]).view(-1, self.embedding_dims)
        output_hidden_head_tensor = torch.cat([*output_hidden_head]).view(-1, self.embedding_dims)
        output_rep_head_tensor = torch.cat([*output_rep_head]).view(-1, self.embedding_dims)
        output_cell_tail_tensor = torch.cat([*output_cell_tail]).view(-1, self.embedding_dims)
        output_hidden_tail_tensor = torch.cat([*output_hidden_tail]).view(-1, self.embedding_dims)
        output_rep_tail_tensor = torch.cat([*output_rep_tail]).view(-1, self.embedding_dims)
        output_rep_tensor = torch.cat([*output_rep]).view(-1, self.embedding_dims)

        if not sample_negatives:
            with torch.no_grad():
                self.cell_head.weight[cell_head_inx, :] = output_cell_head_tensor
                self.hidden_head.weight[hidden_head_inx, :] = output_hidden_head_tensor
                self.cell_tail.weight[cell_tail_inx, :] = output_cell_tail_tensor
                self.hidden_tail.weight[hidden_tail_inx, :] = output_hidden_tail_tensor
                self.node_representations.weight[rep_inx, :] = output_rep_tensor
            return None

        self._last_score_priors = {
            'pos': torch.as_tensor(pos_prior_list, dtype=torch.float32, device=self.device).view(-1, 1),
            'head_neg': torch.as_tensor(head_neg_prior_list, dtype=torch.float32, device=self.device).view(
                -1, self.num_negative
            ),
            'tail_neg': torch.as_tensor(tail_neg_prior_list, dtype=torch.float32, device=self.device).view(
                -1, self.num_negative
            ),
        }

        tail_neg_tensors = torch.cat([*tail_neg_list]).view(-1, self.embedding_dims)
        head_neg_tensors = torch.cat([*head_neg_list]).view(-1, self.embedding_dims)

        # Directed training keeps the original role-specific projection path.
        # Undirected loss needs the raw representations to form a crossed,
        # symmetric bilinear score without collapsing the two projections.
        if self.transfer and not self.is_undirected:
            output_rep_head_tensor = self.dropout(
                self.project_for_scoring(output_rep_head_tensor, 'head')
            )
            output_rep_tail_tensor = self.dropout(
                self.project_for_scoring(output_rep_tail_tensor, 'tail')
            )
            head_neg_tensors = self.dropout(
                self.project_for_scoring(head_neg_tensors, 'head')
            )
            tail_neg_tensors = self.dropout(
                self.project_for_scoring(tail_neg_tensors, 'tail')
            )

        if self.nor and not self.is_undirected:
            output_rep_head_tensor = nn.functional.normalize(output_rep_head_tensor)
            output_rep_tail_tensor = nn.functional.normalize(output_rep_tail_tensor)
            head_neg_tensors = nn.functional.normalize(head_neg_tensors)
            tail_neg_tensors = nn.functional.normalize(tail_neg_tensors)

        with torch.no_grad():
            self.cell_head.weight[cell_head_inx, :] = output_cell_head_tensor
            self.hidden_head.weight[hidden_head_inx, :] = output_hidden_head_tensor
            self.cell_tail.weight[cell_tail_inx, :] = output_cell_tail_tensor
            self.hidden_tail.weight[hidden_tail_inx, :] = output_hidden_tail_tensor
            self.node_representations.weight[rep_inx, :] = output_rep_tensor

        return output_rep_head_tensor, output_rep_tail_tensor, head_neg_tensors, tail_neg_tensors

    def get_rep(self, nodes, rep_type, rep_dict):
        if not isinstance(nodes, list):
            nodes = list(nodes)
        if self.state_lookup_mode == 'exact':
            # A temporal batch contains a mixture of nodes already updated in
            # the current batch and nodes still stored in the global table.  The
            # legacy all-or-nothing fallback discarded every cached update when
            # only one requested node was absent.  Resolve each node separately
            # so its representation matches the actual causal state.
            embedding_table = {
                'cell_head': self.cell_head,
                'cell_tail': self.cell_tail,
                'hidden_head': self.hidden_head,
                'hidden_tail': self.hidden_tail,
                'node_rep': self.node_representations,
            }.get(rep_type)
            if embedding_table is None:
                raise ValueError('Unknown representation type: %s' % rep_type)
            node_ids = [int(node) for node in nodes]
            cached_values = [rep_dict.get(node) for node in node_ids]
            # Reuse within-batch states without a redundant embedding lookup.
            if all(value is not None for value in cached_values):
                return torch.cat(
                    [value.view(1, -1) for value in cached_values], dim=0
                ).view(-1, self.embedding_dims)

            # Fetch all fallback values in one GPU operation and overwrite only
            # the cached positions.  This preserves exact per-node semantics.
            node_tensor = torch.as_tensor(node_ids, dtype=torch.long, device=self.device)
            values = embedding_table(node_tensor)
            cached_positions = []
            present_values = []
            for position, cached_value in enumerate(cached_values):
                if cached_value is not None:
                    cached_positions.append(position)
                    present_values.append(cached_value.view(1, -1))
            if cached_positions:
                position_tensor = torch.as_tensor(cached_positions, dtype=torch.long, device=self.device)
                values = values.clone().index_copy(
                    0, position_tensor, torch.cat(present_values, dim=0)
                )
            return values.view(-1, self.embedding_dims)

        cached_values = []
        missing = False
        for nei in nodes:
            cached_value = rep_dict.get(nei)
            if cached_value is None:
                missing = True
                break
            cached_values.append(cached_value)
        if not missing:
            return torch.cat(cached_values, dim=0).view(-1, self.embedding_dims)

        node_tensor = torch.as_tensor(nodes, dtype=torch.long, device=self.device)
        if rep_type == 'node_rep':
            rep = self.node_representations(node_tensor)
        elif rep_type == 'cell_head':
            rep = self.cell_head(node_tensor)
        elif rep_type == 'cell_tail':
            rep = self.cell_tail(node_tensor)
        elif rep_type == 'hidden_head':
            rep = self.hidden_head(node_tensor)
        else:
            rep = self.hidden_tail(node_tensor)
        cached_indices = []
        cached_values = []
        for idx, nei in enumerate(nodes):
            cached_value = rep_dict.get(nei)
            if cached_value is not None:
                cached_indices.append(idx)
                cached_values.append(cached_value)
        if cached_indices:
            index_tensor = torch.as_tensor(cached_indices, dtype=torch.long, device=self.device)
            rep.index_copy_(0, index_tensor, torch.cat(cached_values, dim=0).view(-1, self.embedding_dims))
        return rep

    def get_adj_arrays(self, node, direction):
        if direction == 'in':
            cached = self._adj_in_arrays[node]
            version = self._adj_in_version[node]
            adj_dict = self.adj_in[node]
            cache_store = self._adj_in_arrays
        else:
            cached = self._adj_out_arrays[node]
            version = self._adj_out_version[node]
            adj_dict = self.adj_out[node]
            cache_store = self._adj_out_arrays

        if cached is not None and cached[0] == version:
            return cached[1], cached[2]

        if not adj_dict:
            neighbors = np.empty(0, dtype=np.int64)
            timestamps = np.empty(0, dtype=np.float32)
        else:
            neighbors = np.fromiter(adj_dict.keys(), dtype=np.int64, count=len(adj_dict))
            timestamps = np.fromiter(adj_dict.values(), dtype=np.float32, count=len(adj_dict))

        cache_store[node] = (version, neighbors, timestamps)
        return neighbors, timestamps

    def get_neighbors(self, node, current_t, threhold=None):
        node = int(node)
        t_value = self._time_value(current_t)
        cache = getattr(self, '_neighbors_cache', None)
        cache_key = None
        if cache is not None:
            cache_key = (node, t_value, threhold)
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        if self.is_undirected:
            # adj_out already contains the mirrored incident neighborhood.
            # Return it in one channel only so propagation does not count every
            # physical neighbor twice through the legacy in/out branches.
            head_neighbors, head_timestamps = self.get_adj_arrays(node, 'out')
            tail_neighbors = np.empty(0, dtype=np.int64)
            tail_timestamps = np.empty(0, dtype=np.float32)
        else:
            head_neighbors, head_timestamps = self.get_adj_arrays(node, 'in')
            tail_neighbors, tail_timestamps = self.get_adj_arrays(node, 'out')

        if threhold is not None:
            if len(head_neighbors) > 0:
                mask_head = (t_value - head_timestamps) <= threhold
                head_neighbors = head_neighbors[mask_head]
                head_timestamps = head_timestamps[mask_head]

            if len(tail_neighbors) > 0:
                mask_tail = (t_value - tail_timestamps) <= threhold
                tail_neighbors = tail_neighbors[mask_tail]
                tail_timestamps = tail_timestamps[mask_tail]

        result = (head_neighbors, tail_neighbors, head_timestamps, tail_timestamps)
        if cache is not None:
            cache[cache_key] = result
        return result

    def get_att_score(self, node, neighbors, node2rep):
        return self.get_enhanced_att_score(node, neighbors, node2rep)

    def second_propagation(self, node, origin_node, current_t, edge_info, node_type, node2cell_head, node2hidden_head,
                           node2cell_tail, node2hidden_tail, node2rep, threhold=None, old_bridge_rep=None,
                           precomputed_head_edge_info=None, precomputed_tail_edge_info=None):
        profile_start = time.time() if self.profile_runtime else None
        head_neighbors, tail_neighbors, head_timestamps, tail_timestamps = self.get_neighbors(node, current_t, threhold)
        head_neighbors = list(head_neighbors)
        head_timestamps = list(head_timestamps)
        tail_neighbors = list(tail_neighbors)
        tail_timestamps = list(tail_timestamps)

        if len(head_neighbors) > 0 or len(tail_neighbors) > 0:
            head_neighbors, head_timestamps, tail_neighbors, tail_timestamps = (
                self.filter_second_order_neighbors_by_direction(
                    node, origin_node, head_neighbors, head_timestamps, tail_neighbors, tail_timestamps, current_t
                )
            )

        if len(head_neighbors) > 0:
            update_start = time.time() if self.profile_runtime else None
            if precomputed_head_edge_info is not None:
                head_nei_edge_info = precomputed_head_edge_info
            elif node_type == 'head':
                head_nei_edge_info = self.tran_head_edge_head(edge_info)
            else:
                head_nei_edge_info = self.tran_tail_edge_head(edge_info)

            head_timestamp_tensor = torch.as_tensor(head_timestamps, dtype=torch.float32,
                                                     device=self.device).view(-1, 1)
            head_delta_ts = current_t - head_timestamp_tensor
            transed_head_delta_ts = self.decayer(head_delta_ts)
            head_nei_cell = self.get_rep(head_neighbors, 'cell_head', node2cell_head)
            if self.if_no_time:
                tran_head_nei_edge_info = head_nei_edge_info.view(1, -1).expand(len(head_neighbors), -1)
            else:
                tran_head_nei_edge_info = head_nei_edge_info.view(1, -1).expand(len(head_neighbors),
                                                                                -1) * transed_head_delta_ts

            if self.is_att:
                att_start = time.time() if self.profile_runtime else None
                temporal_weights = None
                if self.use_enhanced_gat and not self.if_no_time:
                    temporal_weights = transed_head_delta_ts * self.temporal_attention_scale
                att_score_head = self.get_enhanced_att_score(
                    node, head_neighbors, node2rep,
                    attention_type='head', temporal_weights=temporal_weights,
                    center_node_rep=old_bridge_rep
                )
                tran_head_nei_edge_info = tran_head_nei_edge_info * att_score_head
                if att_start is not None:
                    self.add_profile_time('second_order_attention', time.time() - att_start)

            head_nei_cell = head_nei_cell + tran_head_nei_edge_info
            head_nei_hidden = self.act(head_nei_cell)
            head_nei_tail_hidden = self.get_rep(head_neighbors, 'hidden_tail', node2hidden_tail)
            head_nei_rep = self.combiner(head_nei_hidden, head_nei_tail_hidden)

            for i, nei in enumerate(head_neighbors):
                node2cell_head[nei] = head_nei_cell[i].view(-1, self.embedding_dims)
                node2hidden_head[nei] = head_nei_hidden[i].view(-1, self.embedding_dims)
                node2rep[nei] = head_nei_rep[i].view(-1, self.embedding_dims)
            if update_start is not None:
                self.add_profile_time('second_order_update', time.time() - update_start)

        if len(tail_neighbors) > 0:
            update_start = time.time() if self.profile_runtime else None
            if precomputed_tail_edge_info is not None:
                tail_nei_edge_info = precomputed_tail_edge_info
            elif node_type == 'head':
                tail_nei_edge_info = self.tran_head_edge_tail(edge_info)
            else:
                tail_nei_edge_info = self.tran_tail_edge_tail(edge_info)

            tail_timestamp_tensor = torch.as_tensor(tail_timestamps, dtype=torch.float32,
                                                     device=self.device).view(-1, 1)
            tail_delta_ts = current_t - tail_timestamp_tensor
            transed_tail_delta_ts = self.decayer(tail_delta_ts)
            tail_nei_cell = self.get_rep(tail_neighbors, 'cell_tail', node2cell_tail)
            if self.if_no_time:
                tran_tail_nei_edge_info = tail_nei_edge_info.view(1, -1).expand(len(tail_neighbors), -1)
            else:
                tran_tail_nei_edge_info = tail_nei_edge_info.view(1, -1).expand(len(tail_neighbors),
                                                                                -1) * transed_tail_delta_ts

            if self.is_att:
                att_start = time.time() if self.profile_runtime else None
                temporal_weights = None
                if self.use_enhanced_gat and not self.if_no_time:
                    temporal_weights = transed_tail_delta_ts * self.temporal_attention_scale
                att_score_tail = self.get_enhanced_att_score(
                    node, tail_neighbors, node2rep,
                    attention_type='tail', temporal_weights=temporal_weights,
                    center_node_rep=old_bridge_rep
                )
                tran_tail_nei_edge_info = tran_tail_nei_edge_info * att_score_tail
                if att_start is not None:
                    self.add_profile_time('second_order_attention', time.time() - att_start)

            tail_nei_cell = tail_nei_cell + tran_tail_nei_edge_info
            tail_nei_hidden = self.act(tail_nei_cell)
            tail_nei_head_hidden = self.get_rep(tail_neighbors, 'hidden_head', node2hidden_head)
            tail_nei_rep = self.combiner(tail_nei_head_hidden, tail_nei_hidden)

            for i, nei in enumerate(tail_neighbors):
                node2cell_tail[nei] = tail_nei_cell[i].view(-1, self.embedding_dims)
                node2hidden_tail[nei] = tail_nei_hidden[i].view(-1, self.embedding_dims)
                node2rep[nei] = tail_nei_rep[i].view(-1, self.embedding_dims)
            if update_start is not None:
                self.add_profile_time('second_order_update', time.time() - update_start)

        if profile_start is not None:
            self.add_profile_time('second_order_total', time.time() - profile_start)
        return head_neighbors, tail_neighbors

    def loss(self, interactions, pos_weight=3.0, ranking_loss_weight=0.0, ranking_margin=0.0,
             focal_gamma=0.0, loss_balance='legacy'):
        output_rep_head_tensor, output_rep_tail_tensor, head_neg_tensors, tail_neg_tensors = self.forward(interactions)

        head_pos_tensors = output_rep_head_tensor.unsqueeze(1).expand(-1, self.num_negative, -1).reshape(
            -1, self.embedding_dims)
        tail_pos_tensors = output_rep_tail_tensor.unsqueeze(1).expand(-1, self.num_negative, -1).reshape(
            -1, self.embedding_dims)

        num_pp = output_rep_head_tensor.size()[0]
        labels_p = torch.ones(num_pp, dtype=torch.float32, device=self.device)
        labels_n = torch.zeros(num_pp * 2 * self.num_negative, dtype=torch.float32, device=self.device)

        labels = torch.cat((labels_p, labels_n))

        if self.is_undirected:
            # Project each representation group once and reuse its views.
            pos_head_h, pos_head_t = self.project_undirected_views(
                output_rep_head_tensor, apply_dropout=True
            )
            pos_tail_h, pos_tail_t = self.project_undirected_views(
                output_rep_tail_tensor, apply_dropout=True
            )
            neg_head_h, neg_head_t = self.project_undirected_views(
                head_neg_tensors, apply_dropout=True
            )
            neg_tail_h, neg_tail_t = self.project_undirected_views(
                tail_neg_tensors, apply_dropout=True
            )

            pos_head_h_exp = pos_head_h.unsqueeze(1).expand(-1, self.num_negative, -1).reshape(
                -1, self.embedding_dims
            )
            pos_head_t_exp = pos_head_t.unsqueeze(1).expand(-1, self.num_negative, -1).reshape(
                -1, self.embedding_dims
            )
            pos_tail_h_exp = pos_tail_h.unsqueeze(1).expand(-1, self.num_negative, -1).reshape(
                -1, self.embedding_dims
            )
            pos_tail_t_exp = pos_tail_t.unsqueeze(1).expand(-1, self.num_negative, -1).reshape(
                -1, self.embedding_dims
            )

            scores_p = self.score_undirected_views(
                pos_head_h, pos_head_t, pos_tail_h, pos_tail_t
            ).view(num_pp, 1)
            scores_n_1 = self.score_undirected_views(
                neg_head_h, neg_head_t, pos_tail_h_exp, pos_tail_t_exp
            ).view(num_pp, self.num_negative)
            scores_n_2 = self.score_undirected_views(
                pos_head_h_exp, pos_head_t_exp, neg_tail_h, neg_tail_t
            ).view(num_pp, self.num_negative)
        else:
            scores_p = torch.bmm(output_rep_head_tensor.view(num_pp, 1, self.embedding_dims),
                                 output_rep_tail_tensor.view(num_pp, self.embedding_dims, 1))

            scores_n_1 = torch.bmm(head_neg_tensors.view(num_pp * self.num_negative, 1, self.embedding_dims),
                                   tail_pos_tensors.view(num_pp * self.num_negative, self.embedding_dims, 1))

            scores_n_2 = torch.bmm(head_pos_tensors.view(num_pp * self.num_negative, 1, self.embedding_dims),
                                   tail_neg_tensors.view(num_pp * self.num_negative, self.embedding_dims, 1))

            scores_p = scores_p.view(num_pp, 1)
            scores_n_1 = scores_n_1.view(num_pp, self.num_negative)
            scores_n_2 = scores_n_2.view(num_pp, self.num_negative)

            scale = np.sqrt(float(self.embedding_dims))
            scores_p = scores_p / scale
            scores_n_1 = scores_n_1 / scale
            scores_n_2 = scores_n_2 / scale

        if self.persistence_weight > 0 and self._last_score_priors is not None:
            scores_p = scores_p + self.persistence_weight * self._last_score_priors['pos']
            scores_n_1 = scores_n_1 + self.persistence_weight * self._last_score_priors['head_neg']
            scores_n_2 = scores_n_2 + self.persistence_weight * self._last_score_priors['tail_neg']

        scores = torch.cat((scores_p.view(-1), scores_n_1.view(-1), scores_n_2.view(-1)))

        if loss_balance == 'directional_softmax':
            # Parameter-free sampled ranking objective.  Each positive competes
            # separately with corrupted-head and corrupted-tail alternatives,
            # matching the ranking nature of AP/AUC without a manually tuned
            # class weight or auxiliary-loss coefficient.
            head_logits = torch.cat((scores_p, scores_n_1), dim=1)
            tail_logits = torch.cat((scores_p, scores_n_2), dim=1)
            head_loss = torch.logsumexp(head_logits, dim=1) - scores_p.view(-1)
            tail_loss = torch.logsumexp(tail_logits, dim=1) - scores_p.view(-1)
            loss = 0.5 * (head_loss.mean() + tail_loss.mean())
        elif loss_balance == 'group_balanced':
            # Give the positive, corrupted-head, and corrupted-tail groups fixed
            # mass, independent of the number of sampled negatives.  This removes
            # the accidental objective change caused by varying num_negative and
            # needs no dataset-specific class-weight tuning.
            def group_bce(group_scores, target):
                group_labels = torch.full_like(group_scores, float(target))
                values = nn.functional.binary_cross_entropy_with_logits(
                    group_scores, group_labels, reduction='none'
                )
                if focal_gamma > 0:
                    probabilities = torch.sigmoid(group_scores)
                    pt = probabilities if target > 0 else 1.0 - probabilities
                    values = values * torch.pow(
                        1.0 - pt.clamp(min=1e-6, max=1.0 - 1e-6), float(focal_gamma)
                    )
                return values.mean()

            loss = (
                0.5 * group_bce(scores_p, 1.0)
                + 0.25 * group_bce(scores_n_1, 0.0)
                + 0.25 * group_bce(scores_n_2, 0.0)
            )
        elif loss_balance == 'legacy':
            pos_weight_tensor = torch.tensor([float(pos_weight)], dtype=torch.float32, device=self.device)
            bce_loss = nn.functional.binary_cross_entropy_with_logits(
                scores, labels, pos_weight=pos_weight_tensor, reduction='none'
            )
            if focal_gamma > 0:
                probabilities = torch.sigmoid(scores)
                pt = torch.where(labels > 0, probabilities, 1.0 - probabilities)
                bce_loss = bce_loss * torch.pow(
                    1.0 - pt.clamp(min=1e-6, max=1.0 - 1e-6), float(focal_gamma)
                )
            loss = bce_loss.mean()
        else:
            raise ValueError("Unknown loss_balance: %s" % loss_balance)

        if ranking_loss_weight > 0:
            pos_scores = scores_p
            neg_scores = torch.cat((scores_n_1, scores_n_2), dim=1)
            ranking_loss = nn.functional.softplus(float(ranking_margin) - (pos_scores - neg_scores)).mean()
            loss = loss + float(ranking_loss_weight) * ranking_loss

        return loss
