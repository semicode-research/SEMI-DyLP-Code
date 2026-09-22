import argparse
import csv
import json
import os
import random
import time

# Required by CUDA >= 10.2 for deterministic CuBLAS matrix multiplication.
# Set it before importing torch so --deterministic 1 is genuinely effective.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from datasets import Temporal_Dataset
from model import DyGNN


def get_args():
    parser = argparse.ArgumentParser(description="Train SEMI-DyLP")
    parser.add_argument("-data", "--dataset", type=str, default="uci")
    parser.add_argument("-df", "--data_file", type=str, default="data/example.csv")
    parser.add_argument("--graph_mode", choices=["auto", "directed", "undirected"], default="auto",
                        help="Graph semantics. auto treats only Hypertext and IA-Contact as undirected; "
                             "all other datasets remain directed.")
    parser.add_argument("--delimiter", type=str, default="auto",
                        help="Data delimiter: auto/space/whitespace/comma/tab or a literal delimiter.")
    parser.add_argument("--source_col", type=int, default=0)
    parser.add_argument("--target_col", type=int, default=1)
    parser.add_argument("--time_col", type=int, default=-1,
                        help="Timestamp column. Default -1 uses the last column, matching 3-column CSV and 4-column weighted data.")
    parser.add_argument("--time_div", type=float, default=3600.0,
                        help="Timestamp divisor used to convert raw times into model time units.")
    parser.add_argument("-out", "--model_save_dir", type=str, default="checkpoints")
    parser.add_argument("-sr", "--skip_rows", type=int, default=0)
    parser.add_argument("-st", "--starting", type=int, default=0)
    parser.add_argument("-b", "--batch_size", type=int, default=200)
    parser.add_argument("-l", "--learning_rate", type=float, default=0.001)
    # Number of negative samples per endpoint-corruption group.
    parser.add_argument("-nn", "--num_negative", type=int, default=5)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.8)
    parser.add_argument("-vr", "--valid_ratio", type=float, default=0.1)
    parser.add_argument("-tb", "--test_batch_size", type=int, default=200)
    parser.add_argument("-act", "--act", type=str, default="tanh")
    parser.add_argument("-trans", "--transfer", type=int, default=1)
    parser.add_argument("-dp", "--drop_p", type=float, default=0.3)
    parser.add_argument("-w", "--w", type=float, default=2)
    parser.add_argument("-s", "--seed", type=int, default=2024)
    parser.add_argument("-wd", "--weight_decay", type=float, default=1e-5)
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-ee", "--eval_every", type=int, default=1)
    parser.add_argument("-se", "--save_every", type=int, default=1)
    parser.add_argument("-le", "--log_every", type=int, default=20)
    parser.add_argument("-ia", "--is_att", type=int, default=1)
    parser.add_argument("-nt", "--if_no_time", type=int, default=0)
    parser.add_argument("-th", "--threhold", type=float, default=1)
    parser.add_argument("-th2", "--threhold_2nd", type=float, default=1)
    parser.add_argument("-gh", "--gat_heads", type=int, default=4)
    parser.add_argument("-n2", "--num_second_order", type=int, default=5)
    parser.add_argument("-n2b", "--num_second_order_bridges", type=int, default=0)
    parser.add_argument("-2mode", "--second_order_mode", type=str, default="per_bridge",
                        choices=["per_bridge", "event_union"])
    parser.add_argument("-2hop", "--second_order", type=int, default=1)
    parser.add_argument("-rp", "--reset_rep", type=int, default=1)
    parser.add_argument("-dc", "--decay_method", type=str, default="log")
    parser.add_argument("-nor", "--nor", type=int, default=0)
    parser.add_argument("-iu", "--if_updated", type=int, default=0)
    parser.add_argument("-ip", "--if_propagation", type=int, default=1)
    parser.add_argument("-dim", "--embedding_dim", type=int, default=128)
    parser.add_argument("-pw", "--pos_weight", type=str, default="5.0",
                        help="Positive class weight for BCE. Use 'auto' to match the sampled negative/positive ratio.")
    parser.add_argument("-rlw", "--ranking_loss_weight", type=float, default=0.0,
                        help="Weight for the auxiliary pairwise ranking loss used to better align training with AUC/AP.")
    parser.add_argument("-rm", "--ranking_margin", type=float, default=0.0)
    parser.add_argument("-fg", "--focal_gamma", type=float, default=0.0,
                        help="Focal BCE gamma. Values around 0.5-2.0 can help when easy negatives dominate.")
    parser.add_argument("--loss_balance", choices=["legacy", "group_balanced", "directional_softmax"],
                        default="legacy",
                        help="legacy reproduces the original weighted BCE; group_balanced assigns "
                             "fixed mass to positive/corrupted-head/corrupted-tail groups; "
                             "directional_softmax directly ranks the positive against each corruption group.")
    parser.add_argument("--state_lookup_mode", choices=["legacy", "exact"], default="legacy",
                        help="legacy reproduces the optimized 0706 cache fast path; exact keeps the "
                             "same per-node semantics while using the stricter functional update path.")
    parser.add_argument("--negative_rep_timing", choices=["legacy", "pre_event"], default="legacy",
                        help="pre_event scores negatives from the same causal instant as the positive edge.")
    parser.add_argument("-hnr", "--hard_negative_ratio", type=float, default=0.5,
                        help="Fraction of sampled negatives selected as representation-based hard negatives.")
    parser.add_argument("-nps", "--negative_pool_size", type=int, default=0,
                        help="Optional candidate pool size for hard negative sampling. 0 uses history-only candidates.")
    parser.add_argument("-ns", "--negative_sampling_strategy", type=str, default="legacy",
                        choices=["auto", "random", "history", "hard", "mixed", "legacy"],
                        help="Negative sampler used during training.")
    parser.add_argument("-hnw", "--hard_negative_warmup_epochs", type=int, default=0,
                        help="Epochs to use pure random negatives before mixed hard-negative sampling.")
    parser.add_argument("-hnramp", "--hard_negative_ramp_epochs", type=int, default=0,
                        help="Epochs used to linearly ramp to --hard_negative_ratio after warmup.")
    parser.add_argument("-psw", "--persistence_weight", type=float, default=0.0,
                        help="Causal historical-edge persistence score added by the decoder.")
    parser.add_argument("-pst", "--persistence_tau", type=float, default=0.0,
                        help="Time-decay scale for persistence score in model time units. 0 disables decay.")
    parser.add_argument("-pr", "--profile_runtime", type=int, default=0)
    parser.add_argument("--run_test", type=int, default=1, choices=[0, 1],
                        help="Set to 0 during validation-only hyperparameter sweeps so the test split is never evaluated.")
    parser.add_argument("-bm", "--best_metric", type=str, default="auc",
                        choices=["ap", "auc", "mean", "harmonic"],
                        help="Validation criterion. mean/harmonic select checkpoints jointly by AP and AUC.")
    parser.add_argument("-es", "--eval_seed", type=int, default=12345)
    parser.add_argument("--state_seed", type=int, default=None,
                        help="Seed for stochastic state transitions during validation/test. "
                             "Defaults to eval_seed for exact backward compatibility.")
    parser.add_argument("--deterministic", type=int, default=0, choices=[0, 1],
                        help="Use deterministic CUDA settings where PyTorch supports them. The default 0 "
                             "matches the faster 0706 execution; use 1 for final reproducibility checks.")
    parser.add_argument("-ckpt", "--checkpoint_name", type=str, default="best_model.pt")
    parser.add_argument("-blog", "--best_log_name", type=str, default="best_epoch_log.json")
    parser.add_argument("-ps", "--pretest_state_name", type=str, default="pretest_state.pt")
    parser.add_argument("-sps", "--save_pretest_state", type=int, default=1)
    parser.add_argument("-im", "--init_model", type=str, default="")
    parser.add_argument("-pat", "--patience", type=int, default=10)
    parser.add_argument("-md", "--min_delta", type=float, default=0.0)
    parser.add_argument("-slr", "--scheduler_step", type=int, default=10)
    parser.add_argument("-sg", "--scheduler_gamma", type=float, default=0.5)
    return parser.parse_args()


def resolve_pos_weight(pos_weight, num_negative):
    if isinstance(pos_weight, str) and pos_weight.lower() == "auto":
        return float(2 * num_negative)
    return float(pos_weight)


def validation_selection_score(val_ap, val_auc, metric):
    if metric == "ap":
        return val_ap
    if metric == "auc":
        return val_auc
    if metric == "mean":
        return 0.5 * (val_ap + val_auc)
    if metric == "harmonic":
        denominator = val_ap + val_auc
        return 0.0 if denominator <= 0 else 2.0 * val_ap * val_auc / denominator
    raise ValueError(f"Unknown best_metric: {metric}")


def sample_negative_tail(model, head, tail, num_nodes, rng=None):
    forbidden = set(model.adj_out[head].keys())
    forbidden.add(head)
    forbidden.add(tail)
    if len(forbidden) >= num_nodes:
        neg_tail = int(rng.integers(0, num_nodes)) if rng is not None else np.random.randint(0, num_nodes)
        while neg_tail == tail:
            neg_tail = int(rng.integers(0, num_nodes)) if rng is not None else np.random.randint(0, num_nodes)
        return neg_tail

    neg_tail = int(rng.integers(0, num_nodes)) if rng is not None else np.random.randint(0, num_nodes)
    while neg_tail in forbidden:
        neg_tail = int(rng.integers(0, num_nodes)) if rng is not None else np.random.randint(0, num_nodes)
    return neg_tail


def clone_state_dict(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def clone_runtime_state(model):
    return {
        "recent_timestamp": model.recent_timestamp.detach().clone(),
        "adj_out": [dict(x) for x in model.adj_out],
        "adj_in": [dict(x) for x in model.adj_in],
        "degree_array": model._degree_array.copy(),
    }


def restore_runtime_state(model, state):
    model.recent_timestamp = state["recent_timestamp"].detach().clone().to(model.device)
    model.adj_out = [dict(x) for x in state["adj_out"]]
    model.adj_in = [dict(x) for x in state["adj_in"]]
    model._adj_out_arrays = [None for _ in range(model.num_embeddings)]
    model._adj_in_arrays = [None for _ in range(model.num_embeddings)]
    model._adj_out_version = [0 for _ in range(model.num_embeddings)]
    model._adj_in_version = [0 for _ in range(model.num_embeddings)]
    model._degree_array = state["degree_array"].copy()
    model._temporal_dist_cache = None
    model._degree_cache = None
    model._wasserstein_score_cache = None
    model._neighbors_cache = None
    model._current_t_value = None


EXACT_PRETEST_EVENTS = {
    "best_validation_state_after_online_validation",
    "reconstructed_best_validation_state",
}


def save_pretest_state(path, model, args, best_model_path, best_log_path, split_sizes,
                       event="best_validation_state_after_online_validation",
                       best_epoch=None, val_ap=None, val_auc=None):
    payload = {
        "schema_version": 2,
        "event": event,
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "runtime_state": {
            "recent_timestamp": model.recent_timestamp.detach().cpu(),
            "adj_out": [dict(x) for x in model.adj_out],
            "adj_in": [dict(x) for x in model.adj_in],
            "degree_array": model._degree_array.copy(),
        },
        "model_file": os.path.abspath(best_model_path),
        "best_log_file": os.path.abspath(best_log_path),
        "data_file": os.path.abspath(args.data_file),
        "split": split_sizes,
        "config": vars(args),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "val_ap": None if val_ap is None else float(val_ap),
        "val_auc": None if val_auc is None else float(val_auc),
    }
    torch.save(payload, path)


def load_exact_pretest_state(path, model, device, expected_best_epoch=None, expected_split=None,
                             expected_val_ap=None, expected_val_auc=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    event = checkpoint.get("event")
    if event not in EXACT_PRETEST_EVENTS:
        print(f"Ignoring non-exact pre-test state ({event}): {path}", flush=True)
        return False
    if expected_best_epoch is not None and checkpoint.get("best_epoch") != int(expected_best_epoch):
        print(f"Ignoring pre-test state from epoch {checkpoint.get('best_epoch')}; "
              f"expected epoch {expected_best_epoch}.", flush=True)
        return False
    if expected_split is not None and checkpoint.get("split") != expected_split:
        print("Ignoring pre-test state because its data split does not match the current run.", flush=True)
        return False
    for name, expected in (("val_ap", expected_val_ap), ("val_auc", expected_val_auc)):
        actual = checkpoint.get(name)
        if expected is not None and (actual is None or abs(float(actual) - float(expected)) > 1e-8):
            print(f"Ignoring pre-test state because {name}={actual} does not match "
                  f"the best log value {expected}.", flush=True)
            return False
    model.load_state_dict(checkpoint["model_state_dict"])
    restore_runtime_state(model, checkpoint["runtime_state"])
    model.to(device)
    model.eval()
    print(f"Loaded exact pre-test state: {path} | event={event}", flush=True)
    return True


def load_state_dict_safely(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def replay_history(model, interactions, batch_size, reset_reps=True):
    model.reset_time()
    if reset_reps:
        model.reset_reps()
    model.eval()
    with torch.no_grad():
        for start in range(0, len(interactions), batch_size):
            batch = interactions[start:start + batch_size]
            if len(batch) > 0:
                model.forward(batch, sample_negatives=False)


def rebuild_runtime_history_only(model, interactions):
    """Rebuild causal adjacency/timestamps without changing checkpointed node states."""
    model.reset_time()
    with torch.no_grad():
        for interaction in interactions:
            head = int(interaction[0])
            tail = int(interaction[1])
            timestamp = float(interaction[2])
            current_t = torch.as_tensor([[timestamp]], dtype=torch.float32, device=model.device)
            model.recent_timestamp[[head, tail]] = current_t
            model.update_interaction_cache(head, tail, current_t)
    model._temporal_dist_cache = None
    model._degree_cache = None
    model._wasserstein_score_cache = None
    model._neighbors_cache = None
    model._current_t_value = None


def compute_online_repeat_rate(initial_edges, interactions):
    history = set((int(edge[0]), int(edge[1])) for edge in initial_edges)
    if len(interactions) == 0:
        return 0.0
    repeated = 0
    for edge in interactions:
        pair = (int(edge[0]), int(edge[1]))
        if pair in history:
            repeated += 1
        history.add(pair)
    return repeated / len(interactions)


def scheduled_hard_negative_ratio(args, epoch):
    target_ratio = float(args.hard_negative_ratio)
    if args.negative_sampling_strategy != "mixed":
        return target_ratio
    if args.hard_negative_warmup_epochs <= 0 and args.hard_negative_ramp_epochs <= 0:
        return target_ratio
    if epoch < args.hard_negative_warmup_epochs:
        return 0.0
    if args.hard_negative_ramp_epochs <= 0:
        return target_ratio
    progress = (epoch - args.hard_negative_warmup_epochs + 1) / args.hard_negative_ramp_epochs
    return target_ratio * min(max(progress, 0.0), 1.0)


def resolve_negative_sampling_strategy(args, train_repeat_rate, num_nodes):
    if args.negative_sampling_strategy != "auto":
        return args.negative_sampling_strategy
    if train_repeat_rate >= 0.90 and num_nodes <= 1000:
        return "legacy"
    return "random"


def score_pair(model, head, tail, transfer, nor, current_t=None):
    head, tail = model.canonical_endpoints(head, tail)
    head_tensor = model.node_representations.weight[head].view(1, -1)
    tail_tensor = model.node_representations.weight[tail].view(1, -1)
    if model.is_undirected:
        score = model.score_undirected_pairs(
            head_tensor, tail_tensor, apply_dropout=False
        ).item()
    else:
        if transfer:
            head_tensor = model.project_for_scoring(head_tensor, "head")
            tail_tensor = model.project_for_scoring(tail_tensor, "tail")
        # Match DyGNN.forward(): normalize after the transfer in directed mode.
        if nor:
            head_tensor = nn.functional.normalize(head_tensor)
            tail_tensor = nn.functional.normalize(tail_tensor)
        score = (torch.mm(head_tensor, tail_tensor.t()) / np.sqrt(float(model.embedding_dims))).item()
    if getattr(model, "persistence_weight", 0.0) > 0:
        score += model.persistence_weight * model.get_persistence_prior(head, tail, current_t)
    return score


def append_training_log(log_file, row):
    exists = os.path.exists(log_file)
    with open(log_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_best_metrics(path, args, epoch, avg_loss, val_ap, val_auc, train_time, val_time, elapsed, model_file=None):
    payload = {
        "best_epoch": epoch,
        "best_metric": args.best_metric,
        "val_ap": val_ap,
        "val_auc": val_auc,
        "avg_loss": avg_loss,
        "train_time_sec": train_time,
        "val_time_sec": val_time,
        "total_time_sec": elapsed,
        "model_file": os.path.abspath(model_file or os.path.join(args.model_save_dir, args.checkpoint_name)),
        "config": vars(args),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_best_epoch_log(path, args, epoch, avg_loss, val_ap, val_auc, train_time, val_time, elapsed,
                        best_score, model_file, split_sizes):
    payload = {
        "schema_version": 1,
        "event": "best_validation_epoch",
        "best_epoch": int(epoch),
        "best_metric": args.best_metric,
        "best_score": float(best_score),
        "val_ap": float(val_ap),
        "val_auc": float(val_auc),
        "avg_loss": float(avg_loss),
        "train_time_sec": float(train_time),
        "val_time_sec": float(val_time),
        "total_time_sec": float(elapsed),
        "model_file": os.path.abspath(model_file),
        "pretest_state_file": os.path.abspath(os.path.join(args.model_save_dir, args.pretest_state_name)),
        "data_file": os.path.abspath(args.data_file),
        "split": split_sizes,
        "config": vars(args),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def evaluate_metrics(model, data_loader, device, transfer, nor, eval_seed, state_seed=None):
    model.eval()
    all_scores = []
    all_labels = []
    rng = np.random.default_rng(eval_seed)
    python_rng_state = random.getstate()
    # Negative sampling uses the NumPy generator.  Keep the model's stochastic
    # propagation stream separate so changing evaluation negatives cannot also
    # change the temporal node states.
    random.seed(eval_seed if state_seed is None else state_seed)

    try:
        with torch.no_grad():
            num_nodes = model.node_representations.weight.shape[0]

            for interactions in data_loader:
                if isinstance(interactions, torch.Tensor):
                    interactions = interactions.cpu().numpy()

                for i in range(len(interactions)):
                    interaction = interactions[i]
                    head = int(interaction[0])
                    tail = int(interaction[1])
                    timestamp = float(interaction[2])

                    pos_score = score_pair(model, head, tail, transfer, nor, timestamp)
                    all_scores.append(pos_score)
                    all_labels.append(1)

                    neg_tail = sample_negative_tail(model, head, tail, num_nodes, rng)

                    neg_score = score_pair(model, head, neg_tail, transfer, nor, timestamp)
                    all_scores.append(neg_score)
                    all_labels.append(0)

                    model.forward(np.reshape(interaction, (1, 3)), sample_negatives=False)
    finally:
        random.setstate(python_rng_state)

    if not all_scores:
        return 0, 0

    ap = average_precision_score(all_labels, all_scores)
    auc = roc_auc_score(all_labels, all_scores)
    return ap, auc


def train(args, data, num_nodes):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = not bool(args.deterministic)
        torch.backends.cudnn.deterministic = bool(args.deterministic)
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_array = data.data
    if hasattr(data, "time_span"):
        print(f"Dataset model time span: {data.time_span:.6g}", flush=True)
    train_end = int(len(data_array) * args.train_ratio)
    val_end = int(len(data_array) * (args.train_ratio + args.valid_ratio))
    train_data = data_array[:train_end]
    if train_end <= 0 or val_end <= train_end or val_end >= len(data_array):
        raise ValueError("Require train_ratio > 0, valid_ratio > 0, and train_ratio + valid_ratio < 1.0")
    validation_data = data_array[train_end:val_end]
    test_data = data_array[val_end:]

    print(f"Train length: {len(train_data)}, Valid length: {len(validation_data)}, Test length: {len(test_data)}")
    train_repeat = compute_online_repeat_rate([], train_data)
    valid_repeat = compute_online_repeat_rate(train_data, validation_data)
    test_repeat = compute_online_repeat_rate(np.concatenate([train_data, validation_data], axis=0), test_data)
    print(f"Online repeat rate | Train: {train_repeat:.2%}, Valid: {valid_repeat:.2%}, "
          f"Test: {test_repeat:.2%}", flush=True)
    resolved_negative_sampling_strategy = resolve_negative_sampling_strategy(args, train_repeat, num_nodes)
    if args.negative_sampling_strategy == "auto":
        print(f"Auto negative sampling selected: {resolved_negative_sampling_strategy} "
              f"(train_repeat={train_repeat:.2%}, num_nodes={num_nodes})", flush=True)
        args.negative_sampling_strategy = resolved_negative_sampling_strategy
        if args.negative_sampling_strategy == "legacy":
            args.hard_negative_ratio = 0.5
            print("Auto negative sampling set hard_negative_ratio=0.5 for legacy sampling.", flush=True)
        elif args.negative_sampling_strategy == "random":
            args.hard_negative_ratio = 0.0
            print("Auto negative sampling set hard_negative_ratio=0.0 for random sampling.", flush=True)
    if len(train_data) > 0:
        train_time_span = float(train_data[-1, 2] - train_data[0, 2])
        for name, value in [("threhold", args.threhold), ("threhold_2nd", args.threhold_2nd)]:
            if value is not None and train_time_span > 0 and float(value) > 0.5 * train_time_span:
                print(f"Warning: --{name}={value} is larger than half of the training time span "
                      f"({train_time_span:.6g}). This may include very old neighbors and hurt AUC/AP.",
                      flush=True)

    sampler = SequentialSampler(train_data)
    train_loader = DataLoader(train_data, args.batch_size, sampler=sampler)
    val_loader = DataLoader(validation_data, batch_size=args.test_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False)

    os.makedirs(args.model_save_dir, exist_ok=True)

    model = DyGNN(num_nodes, args.embedding_dim, args.embedding_dim, device, args.w, args.is_att, args.transfer,
                  args.nor, args.if_no_time, threhold=args.threhold, threhold_2nd=args.threhold_2nd,
                  second_order=args.second_order, if_updated=args.if_updated, drop_p=args.drop_p,
                  num_negative=args.num_negative, act=args.act, if_propagation=args.if_propagation,
                  decay_method=args.decay_method, gat_heads=args.gat_heads,
                  second_order_max_nodes=args.num_second_order,
                  second_order_bridge_max_nodes=args.num_second_order_bridges,
                  second_order_mode=args.second_order_mode,
                  hard_negative_ratio=args.hard_negative_ratio,
                  negative_pool_size=args.negative_pool_size,
                  negative_sampling_strategy=args.negative_sampling_strategy,
                  persistence_weight=args.persistence_weight,
                  persistence_tau=args.persistence_tau,
                  profile_runtime=bool(args.profile_runtime),
                  state_lookup_mode=args.state_lookup_mode,
                  negative_rep_timing=args.negative_rep_timing,
                  graph_mode=args.graph_mode)

    model.to(device)
    if args.init_model:
        model.load_state_dict(load_state_dict_safely(args.init_model, device))
        print(f"Loaded initial model: {args.init_model}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler_step > 0:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step,
                                                    gamma=args.scheduler_gamma)

    pos_weight = resolve_pos_weight(args.pos_weight, args.num_negative)
    args.resolved_pos_weight = pos_weight
    print(f"Training objective: pos_weight={pos_weight:.4g}, "
          f"loss_balance={args.loss_balance}, "
          f"state_lookup={args.state_lookup_mode}, negative_rep_timing={args.negative_rep_timing}, "
          f"ranking_loss_weight={args.ranking_loss_weight}, ranking_margin={args.ranking_margin}, "
          f"focal_gamma={args.focal_gamma}")
    print(f"Negative sampling: strategy={args.negative_sampling_strategy}, "
          f"target_hard_ratio={args.hard_negative_ratio}, "
          f"warmup={args.hard_negative_warmup_epochs}, ramp={args.hard_negative_ramp_epochs}",
          flush=True)

    best_score = -float("inf")
    best_epoch = None
    best_val_ap = None
    best_val_auc = None
    epochs_without_improvement = 0
    log_file = os.path.join(args.model_save_dir, "training_log.csv")
    best_metrics_path = os.path.join(args.model_save_dir, "best_metrics.json")
    best_log_path = os.path.join(args.model_save_dir, args.best_log_name)
    final_metrics_path = os.path.join(args.model_save_dir, "final_metrics.json")
    best_model_path = os.path.join(args.model_save_dir, args.checkpoint_name)
    pretest_state_path = os.path.join(args.model_save_dir, args.pretest_state_name)
    split_sizes = {"train": len(train_data), "valid": len(validation_data), "test": len(test_data)}
    print(f"Pre-test state saving: {'ON' if args.save_pretest_state else 'OFF'} | Path: {pretest_state_path}", flush=True)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        model.hard_negative_ratio = scheduled_hard_negative_ratio(args, epoch)
        model.reset_profile_stats()
        model.reset_time()
        if args.reset_rep:
            model.reset_reps()

        total_loss = 0
        batch_count = 0
        train_start = time.time()

        last_log_time = time.time()
        for i, interactions in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(interactions, pos_weight=pos_weight,
                              ranking_loss_weight=args.ranking_loss_weight,
                              ranking_margin=args.ranking_margin,
                              focal_gamma=args.focal_gamma,
                              loss_balance=args.loss_balance)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch}, batch={i}; aborting instead of saving a corrupt model."
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

            if args.log_every > 0 and i % args.log_every == 0:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                now = time.time()
                batch_time = (now - last_log_time) / max(args.log_every, 1) if i > 0 else now - train_start
                last_log_time = time.time()
                print(f"Epoch {epoch} Batch {i} | Loss: {loss.item():.4f} | BatchTime: {batch_time:.2f}s")

        train_time = time.time() - train_start
        if scheduler is not None:
            scheduler.step()
        val_ap = float("nan")
        val_auc = float("nan")
        val_time = 0.0

        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            val_start = time.time()
            train_state = clone_state_dict(model)
            runtime_state = clone_runtime_state(model)
            val_ap, val_auc = evaluate_metrics(
                model, val_loader, device, args.transfer, args.nor, args.eval_seed,
                args.state_seed,
            )
            val_time = time.time() - val_start
            current_score = validation_selection_score(val_ap, val_auc, args.best_metric)
            improved = not np.isnan(current_score) and current_score > best_score + args.min_delta
            if improved and args.save_pretest_state:
                pretest_state_start = time.time()
                save_pretest_state(
                    pretest_state_path, model, args, best_model_path, best_log_path, split_sizes,
                    event="best_validation_state_after_online_validation",
                    best_epoch=epoch, val_ap=val_ap, val_auc=val_auc,
                )
                pretest_state_time = time.time() - pretest_state_start
                print(f"Saved improved pre-test state: {pretest_state_path} | Time: {pretest_state_time:.2f}s", flush=True)

            model.load_state_dict(train_state)
            restore_runtime_state(model, runtime_state)

        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            current_epoch_path = os.path.join(args.model_save_dir, f"model_epoch_{epoch}.pt")
            torch.save(model.state_dict(), current_epoch_path)

        avg_loss = total_loss / max(batch_count, 1)
        elapsed = time.time() - epoch_start

        current_score = validation_selection_score(val_ap, val_auc, args.best_metric)
        improved = args.eval_every > 0 and not np.isnan(current_score) and current_score > best_score + args.min_delta
        if improved:
            best_score = current_score
            best_epoch = epoch
            best_val_ap = val_ap
            best_val_auc = val_auc
            epochs_without_improvement = 0
            torch.save(train_state, best_model_path)
            save_best_metrics(best_metrics_path, args, epoch, avg_loss, val_ap, val_auc,
                              train_time, val_time, elapsed, best_model_path)
            save_best_epoch_log(best_log_path, args, epoch, avg_loss, val_ap, val_auc,
                                train_time, val_time, elapsed, best_score, best_model_path, split_sizes)
            print(f"Saved best epoch log: {best_log_path}")
        elif args.eval_every > 0 and not np.isnan(current_score):
            epochs_without_improvement += 1

        append_training_log(log_file, {
            "epoch": epoch,
            "avg_loss": avg_loss,
            "val_ap": val_ap,
            "val_auc": val_auc,
            "train_time_sec": train_time,
            "val_time_sec": val_time,
            "total_time_sec": elapsed,
            "best_score": best_score,
            "best_metric": args.best_metric,
            "second_order": args.second_order,
            "first_order_threshold": args.threhold,
            "second_order_threshold": args.threhold_2nd,
            "second_order_k": args.num_second_order,
            "second_order_bridge_k": args.num_second_order_bridges,
            "second_order_mode": args.second_order_mode,
            "ranking_loss_weight": args.ranking_loss_weight,
            "ranking_margin": args.ranking_margin,
            "focal_gamma": args.focal_gamma,
            "loss_balance": args.loss_balance,
            "state_lookup_mode": args.state_lookup_mode,
            "negative_rep_timing": args.negative_rep_timing,
            "hard_negative_ratio": args.hard_negative_ratio,
            "effective_hard_negative_ratio": model.hard_negative_ratio,
            "negative_pool_size": args.negative_pool_size,
            "negative_sampling_strategy": args.negative_sampling_strategy,
            "persistence_weight": args.persistence_weight,
            "persistence_tau": args.persistence_tau,
            "seed": args.seed,
            "improved": int(improved),
            "epochs_without_improvement": epochs_without_improvement,
        })

        print(f"Epoch {epoch} done | Total: {elapsed:.2f}s | Train: {train_time:.2f}s | "
              f"Val: {val_time:.2f}s | Avg Loss: {avg_loss:.4f} | "
              f"HardRatio: {model.hard_negative_ratio:.3f} | "
              f"VAL AP: {val_ap:.4f} | VAL AUC: {val_auc:.4f}")
        if args.profile_runtime:
            profile_parts = []
            for name, value in sorted(model.profile_stats.items()):
                if name in model.profile_count_keys:
                    profile_parts.append(f"{name}: {int(value)}")
                else:
                    profile_parts.append(f"{name}: {value:.2f}s")
            profile_text = ", ".join(profile_parts)
            print(f"Runtime profile | {profile_text}")
        if args.patience > 0 and epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch} | Best {args.best_metric}: {best_score:.4f}")
            break

    if not os.path.exists(best_model_path):
        torch.save(clone_state_dict(model), best_model_path)

    if not args.run_test:
        sweep_metrics = {
            "validation_only": True,
            "model_file": os.path.abspath(best_model_path),
            "best_log_file": os.path.abspath(best_log_path),
            "best_metric": args.best_metric,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_val_ap": best_val_ap,
            "best_val_auc": best_val_auc,
            "test_ap": None,
            "test_auc": None,
            "train_edges": len(train_data),
            "valid_edges": len(validation_data),
            "test_edges": len(test_data),
            "config": vars(args),
        }
        with open(final_metrics_path, "w", encoding="utf-8") as f:
            json.dump(sweep_metrics, f, indent=2)
        print(
            f"Validation-only sweep complete | Best epoch: {best_epoch} | "
            f"VAL AP: {best_val_ap:.4f} | VAL AUC: {best_val_auc:.4f} | "
            "test split not evaluated.",
            flush=True,
        )
        return

    pretest_state_time = 0.0
    exact_pretest_loaded = False
    if args.save_pretest_state and os.path.exists(pretest_state_path):
        exact_pretest_loaded = load_exact_pretest_state(
            pretest_state_path, model, device,
            expected_best_epoch=best_epoch, expected_split=split_sizes,
            expected_val_ap=best_val_ap, expected_val_auc=best_val_auc,
        )

    if not exact_pretest_loaded:
        model.load_state_dict(load_state_dict_safely(best_model_path, device))
        model.to(device)
        model.eval()
        rebuild_runtime_history_only(model, train_data)
        replay_val_ap, replay_val_auc = evaluate_metrics(
            model, val_loader, device, args.transfer, args.nor, args.eval_seed, args.state_seed
        )
        print(f"Reconstructed best validation state | VAL AP: {replay_val_ap:.4f} | "
              f"VAL AUC: {replay_val_auc:.4f}", flush=True)
        if best_val_ap is not None and best_val_auc is not None:
            print(f"Expected best validation metrics | VAL AP: {best_val_ap:.4f} | "
                  f"VAL AUC: {best_val_auc:.4f}", flush=True)
            if abs(replay_val_ap - best_val_ap) > 1e-6 or abs(replay_val_auc - best_val_auc) > 1e-6:
                raise RuntimeError(
                    "Reconstructed validation metrics do not match the selected best epoch; "
                    "test evaluation is aborted to avoid reporting an inconsistent result."
                )

    if args.save_pretest_state and not exact_pretest_loaded:
        pretest_state_start = time.time()
        save_pretest_state(
            pretest_state_path, model, args, best_model_path, best_log_path, split_sizes,
            event="reconstructed_best_validation_state",
            best_epoch=best_epoch, val_ap=best_val_ap, val_auc=best_val_auc,
        )
        pretest_state_time = time.time() - pretest_state_start
        print(f"Saved reconstructed pre-test state: {pretest_state_path} | "
              f"Time: {pretest_state_time:.2f}s", flush=True)
    test_start = time.time()
    test_ap, test_auc = evaluate_metrics(
        model, test_loader, device, args.transfer, args.nor, args.eval_seed + 1,
        None if args.state_seed is None else args.state_seed + 1,
    )
    test_time = time.time() - test_start
    final_metrics = {
        "model_file": os.path.abspath(best_model_path),
        "best_log_file": os.path.abspath(best_log_path),
        "pretest_state_file": os.path.abspath(pretest_state_path) if args.save_pretest_state else "",
        "best_metric": args.best_metric,
        "best_score": best_score,
        "test_ap": test_ap,
        "test_auc": test_auc,
        "test_time_sec": test_time,
        "pretest_state_save_time_sec": pretest_state_time,
        "train_edges": len(train_data),
        "valid_edges": len(validation_data),
        "test_edges": len(test_data),
        "config": vars(args),
    }
    with open(final_metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Best checkpoint: {best_model_path}")
    print(f"TEST AP: {test_ap:.4f} | TEST AUC: {test_auc:.4f} | Test time: {test_time:.2f}s")


if __name__ == "__main__":
    args = get_args()
    dataset = Temporal_Dataset(args.data_file, args.starting, args.skip_rows, args.time_div,
                               delimiter=args.delimiter, source_col=args.source_col,
                               target_col=args.target_col, time_col=args.time_col,
                               dataset_name=args.dataset, graph_mode=args.graph_mode)
    # Store the resolved mode (rather than "auto") in logs and checkpoints so
    # every published run is fully reproducible and auditable.
    args.graph_mode = dataset.graph_mode
    train(args, dataset, dataset.max_node + 1)
