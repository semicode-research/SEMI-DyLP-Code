import argparse
import json
import os
import random
import sys
import time

# Keep standalone evaluation consistent with deterministic training.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from datasets import Temporal_Dataset
from model import DyGNN


EXACT_PRETEST_EVENTS = {
    "best_validation_state_after_online_validation",
    "reconstructed_best_validation_state",
}


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate SEMI-DyLP")
    parser.add_argument("-m", "--model_file", type=str, default="")
    parser.add_argument("-log", "--best_log_file", type=str, default="")
    parser.add_argument("-ps", "--pretest_state_file", type=str, default="")
    parser.add_argument("-df", "--data_file", type=str, default="data/example.csv")
    parser.add_argument("--dataset", type=str, default="")
    parser.add_argument("--graph_mode", choices=["auto", "directed", "undirected"], default="auto")
    parser.add_argument("--delimiter", type=str, default="auto")
    parser.add_argument("--source_col", type=int, default=0)
    parser.add_argument("--target_col", type=int, default=1)
    parser.add_argument("--time_col", type=int, default=-1)
    parser.add_argument("--time_div", type=float, default=3600.0)
    parser.add_argument("-sr", "--skip_rows", type=int, default=0)
    parser.add_argument("-st", "--starting", type=int, default=0)
    parser.add_argument("-tr", "--train_ratio", type=float, default=0.8)
    parser.add_argument("-vr", "--valid_ratio", type=float, default=0.1)
    parser.add_argument("-tb", "--test_batch_size", type=int, default=200)
    parser.add_argument("-dim", "--embedding_dim", type=int, default=128)
    parser.add_argument("-nn", "--num_negative", type=int, default=5)
    parser.add_argument("-w", "--w", type=float, default=2)
    parser.add_argument("-ia", "--is_att", type=int, default=1)
    parser.add_argument("-nor", "--nor", type=int, default=0)
    parser.add_argument("-trans", "--transfer", type=int, default=1)
    parser.add_argument("-nt", "--if_no_time", type=int, default=0)
    parser.add_argument("-seed", "--seed", type=int, default=2024)
    parser.add_argument("-es", "--eval_seed", type=int, default=None)
    parser.add_argument("--state_seed", type=int, default=None,
                        help="Independent seed for stochastic temporal-state transitions.")
    parser.add_argument("--state_lookup_mode", choices=["legacy", "exact"], default="legacy")
    parser.add_argument("--negative_rep_timing", choices=["legacy", "pre_event"], default="legacy")
    parser.add_argument("-th", "--threhold", type=float, default=24)
    parser.add_argument("-th2", "--threhold_2nd", type=float, default=6)
    parser.add_argument("-gh", "--gat_heads", type=int, default=4)
    parser.add_argument("-n2", "--num_second_order", type=int, default=5)
    parser.add_argument("-n2b", "--num_second_order_bridges", type=int, default=0)
    parser.add_argument("-2mode", "--second_order_mode", type=str, default="per_bridge",
                        choices=["per_bridge", "event_union"])
    parser.add_argument("-2hop", "--second_order", type=int, default=1)
    parser.add_argument("-hnr", "--hard_negative_ratio", type=float, default=0.0)
    parser.add_argument("-nps", "--negative_pool_size", type=int, default=0)
    parser.add_argument("-ns", "--negative_sampling_strategy", type=str, default="random",
                        choices=["auto", "random", "history", "hard", "mixed", "legacy"])
    parser.add_argument("-psw", "--persistence_weight", type=float, default=0.0)
    parser.add_argument("-pst", "--persistence_tau", type=float, default=0.0)
    parser.add_argument("--allow_approximate_reconstruction", type=int, default=0,
                        help="Continue testing when an old run's reconstructed validation state "
                             "does not exactly match its best log. The resulting test metrics are approximate.")
    return parser.parse_args()



def apply_best_log(args):
    if not args.best_log_file:
        if not args.model_file:
            raise ValueError("Either --model_file or --best_log_file must be provided.")
        return args

    with open(args.best_log_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    option_to_key = {
        "-m": "model_file", "--model_file": "model_file",
        "-log": "best_log_file", "--best_log_file": "best_log_file",
        "-ps": "pretest_state_file", "--pretest_state_file": "pretest_state_file",
        "-df": "data_file", "--data_file": "data_file",
        "--dataset": "dataset", "--graph_mode": "graph_mode",
        "--delimiter": "delimiter",
        "--source_col": "source_col",
        "--target_col": "target_col",
        "--time_col": "time_col",
        "--time_div": "time_div",
        "-sr": "skip_rows", "--skip_rows": "skip_rows",
        "-st": "starting", "--starting": "starting",
        "-tr": "train_ratio", "--train_ratio": "train_ratio",
        "-vr": "valid_ratio", "--valid_ratio": "valid_ratio",
        "-tb": "test_batch_size", "--test_batch_size": "test_batch_size",
        "-dim": "embedding_dim", "--embedding_dim": "embedding_dim",
        "-nn": "num_negative", "--num_negative": "num_negative",
        "-w": "w", "--w": "w",
        "-ia": "is_att", "--is_att": "is_att",
        "-nor": "nor", "--nor": "nor",
        "-trans": "transfer", "--transfer": "transfer",
        "-nt": "if_no_time", "--if_no_time": "if_no_time",
        "-seed": "seed", "--seed": "seed",
        "-es": "eval_seed", "--eval_seed": "eval_seed",
        "--state_seed": "state_seed",
        "--state_lookup_mode": "state_lookup_mode",
        "--negative_rep_timing": "negative_rep_timing",
        "-th": "threhold", "--threhold": "threhold",
        "-th2": "threhold_2nd", "--threhold_2nd": "threhold_2nd",
        "-gh": "gat_heads", "--gat_heads": "gat_heads",
        "-n2": "num_second_order", "--num_second_order": "num_second_order",
        "-n2b": "num_second_order_bridges", "--num_second_order_bridges": "num_second_order_bridges",
        "-2mode": "second_order_mode", "--second_order_mode": "second_order_mode",
        "-2hop": "second_order", "--second_order": "second_order",
        "-hnr": "hard_negative_ratio", "--hard_negative_ratio": "hard_negative_ratio",
        "-nps": "negative_pool_size", "--negative_pool_size": "negative_pool_size",
        "-ns": "negative_sampling_strategy", "--negative_sampling_strategy": "negative_sampling_strategy",
        "-psw": "persistence_weight", "--persistence_weight": "persistence_weight",
        "-pst": "persistence_tau", "--persistence_tau": "persistence_tau",
    }
    cli_keys = {option_to_key[arg] for arg in sys.argv[1:] if arg in option_to_key}

    config = payload.get("config", {})
    for key, value in config.items():
        if hasattr(args, key) and key not in {"model_file", "best_log_file"} and key not in cli_keys:
            setattr(args, key, value)

    log_dir = os.path.dirname(os.path.abspath(args.best_log_file))
    model_file = payload.get("model_file", args.model_file)
    if not model_file:
        raise ValueError(f"No model_file found in {args.best_log_file}")
    if not os.path.isabs(model_file):
        model_file = os.path.abspath(os.path.join(log_dir, model_file))
    args.model_file = model_file

    data_file = payload.get("data_file") or config.get("data_file")
    if data_file:
        if not os.path.isabs(data_file):
            data_file = os.path.abspath(os.path.join(log_dir, data_file))
        args.data_file = data_file

    pretest_state_file = payload.get("pretest_state_file")
    if not pretest_state_file and config.get("pretest_state_name"):
        pretest_state_file = os.path.join(log_dir, config["pretest_state_name"])
    if pretest_state_file and "pretest_state_file" not in cli_keys:
        if not os.path.isabs(pretest_state_file):
            pretest_state_file = os.path.abspath(os.path.join(log_dir, pretest_state_file))
        args.pretest_state_file = pretest_state_file

    print(f"Loaded best epoch log: {args.best_log_file}")
    print(f"Best epoch: {payload.get('best_epoch')} | {payload.get('best_metric')}: {payload.get('best_score')}")
    args.expected_best_epoch = payload.get("best_epoch")
    args.expected_val_ap = payload.get("val_ap")
    args.expected_val_auc = payload.get("val_auc")
    args.expected_split = payload.get("split")
    return args

def load_state_dict_safely(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint


def restore_runtime_state(model, state):
    model.recent_timestamp = state["recent_timestamp"].to(model.device)
    model.adj_out = [dict(x) for x in state["adj_out"]]
    model.adj_in = [dict(x) for x in state["adj_in"]]
    model._degree_array = state["degree_array"].copy()
    model._adj_out_arrays = [None for _ in range(model.num_embeddings)]
    model._adj_in_arrays = [None for _ in range(model.num_embeddings)]
    model._adj_out_version = [len(item) for item in model.adj_out]
    model._adj_in_version = [len(item) for item in model.adj_in]
    model._temporal_dist_cache = None
    model._degree_cache = None
    model._wasserstein_score_cache = None
    model._neighbors_cache = None
    model._current_t_value = None


def load_pretest_state(model, path, device, expected_best_epoch=None, expected_split=None,
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
        print("Ignoring pre-test state because its data split does not match the best log.", flush=True)
        return False
    for metric_name, expected_value in (("val_ap", expected_val_ap),
                                        ("val_auc", expected_val_auc)):
        actual_value = checkpoint.get(metric_name)
        if expected_value is not None and (
                actual_value is None or abs(float(actual_value) - float(expected_value)) > 1e-12):
            print(f"Ignoring pre-test state because {metric_name}={actual_value} does not match "
                  f"the best log ({expected_value}).", flush=True)
            return False
    model.load_state_dict(checkpoint["model_state_dict"])
    restore_runtime_state(model, checkpoint["runtime_state"])
    model.to(device)
    model.eval()
    print(f"Loaded exact pre-test state: {path} | event={event}", flush=True)
    return True


def replay_history(model, interactions, batch_size):
    model.reset_time()
    model.reset_reps()
    model.eval()
    print(f"Replaying history: {len(interactions)} interactions | batch_size={batch_size}", flush=True)
    replay_start = time.time()
    with torch.no_grad():
        for start in range(0, len(interactions), batch_size):
            batch = interactions[start:start + batch_size]
            if len(batch) > 0:
                model.forward(batch, sample_negatives=False)
            if (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(interactions):
                done = min(start + batch_size, len(interactions))
                print(f"Replay progress: {done}/{len(interactions)}", flush=True)
    print(f"Replay done: {time.time() - replay_start:.2f}s", flush=True)


def rebuild_runtime_history_only(model, interactions):
    """Rebuild causal adjacency/timestamps while preserving checkpointed node states."""
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


def score_pair(model, head, tail, transfer, nor, current_t=None):
    head, tail = model.canonical_endpoints(head, tail)
    head_emb = model.node_representations.weight[head].view(1, -1)
    tail_emb = model.node_representations.weight[tail].view(1, -1)
    if transfer:
        head_emb = model.project_for_scoring(head_emb, "head")
        tail_emb = model.project_for_scoring(tail_emb, "tail")
    if nor:
        head_emb = nn.functional.normalize(head_emb)
        tail_emb = nn.functional.normalize(tail_emb)
    score = (torch.mm(head_emb, tail_emb.t()) / np.sqrt(float(model.embedding_dims))).item()
    if getattr(model, "persistence_weight", 0.0) > 0:
        score += model.persistence_weight * model.get_persistence_prior(head, tail, current_t)
    return score


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


def evaluate_online(model, interactions, num_nodes, transfer, nor, eval_seed, state_seed=None,
                    progress_label=None):
    model.eval()
    all_scores = []
    all_labels = []
    rng = np.random.default_rng(eval_seed)
    python_rng_state = random.getstate()
    random.seed(eval_seed if state_seed is None else state_seed)

    try:
        with torch.no_grad():
            for i, interaction in enumerate(interactions):
                head = int(interaction[0])
                tail = int(interaction[1])
                timestamp = float(interaction[2])
                all_scores.append(score_pair(model, head, tail, transfer, nor, timestamp))
                all_labels.append(1)

                neg_tail = sample_negative_tail(model, head, tail, num_nodes, rng)
                all_scores.append(score_pair(model, head, neg_tail, transfer, nor, timestamp))
                all_labels.append(0)
                model.forward(np.reshape(interaction, (1, 3)), sample_negatives=False)

                if progress_label and (i + 1) % 500 == 0:
                    print(f"{progress_label} progress: {i + 1}/{len(interactions)}", flush=True)
    finally:
        random.setstate(python_rng_state)

    scores = np.asarray(all_scores)
    labels = np.asarray(all_labels)
    ap = average_precision_score(labels, scores)
    auc = roc_auc_score(labels, scores)
    f1 = f1_score(labels, (scores > 0).astype(int))
    return ap, auc, f1


def run_test():
    total_start = time.time()
    args = apply_best_log(get_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = Temporal_Dataset(args.data_file, args.starting, args.skip_rows, args.time_div,
                            delimiter=args.delimiter, source_col=args.source_col,
                            target_col=args.target_col, time_col=args.time_col,
                            dataset_name=args.dataset, graph_mode=args.graph_mode)
    args.graph_mode = data.graph_mode
    num_nodes = data.max_node + 1
    data_array = data.data

    train_idx = int(len(data_array) * args.train_ratio)
    val_idx = int(len(data_array) * (args.train_ratio + args.valid_ratio))
    train_data = data_array[:train_idx]
    validation_data = data_array[train_idx:val_idx]
    test_data = data_array[val_idx:]

    model = DyGNN(num_nodes, args.embedding_dim, args.embedding_dim, device,
                  w=args.w, is_att=args.is_att, transfer=args.transfer,
                  nor=args.nor, if_no_time=args.if_no_time,
                  threhold=args.threhold, threhold_2nd=args.threhold_2nd,
                  second_order=args.second_order, num_negative=args.num_negative, gat_heads=args.gat_heads,
                  second_order_max_nodes=args.num_second_order,
                  second_order_bridge_max_nodes=args.num_second_order_bridges,
                  second_order_mode=args.second_order_mode,
                  hard_negative_ratio=args.hard_negative_ratio,
                  negative_pool_size=args.negative_pool_size,
                  negative_sampling_strategy=args.negative_sampling_strategy,
                  persistence_weight=args.persistence_weight,
                   persistence_tau=args.persistence_tau,
                   state_lookup_mode=args.state_lookup_mode,
                   negative_rep_timing=args.negative_rep_timing,
                   graph_mode=args.graph_mode)

    model.load_state_dict(load_state_dict_safely(args.model_file, device))
    model.to(device)
    model.eval()
    exact_pretest_loaded = False
    approximate_pretest_state = False
    if args.pretest_state_file and os.path.exists(args.pretest_state_file):
        exact_pretest_loaded = load_pretest_state(
            model, args.pretest_state_file, device,
            expected_best_epoch=getattr(args, "expected_best_epoch", None),
            expected_split=getattr(args, "expected_split", None),
            expected_val_ap=getattr(args, "expected_val_ap", None),
            expected_val_auc=getattr(args, "expected_val_auc", None),
        )

    if not exact_pretest_loaded:
        if args.pretest_state_file:
            print(f"Exact pre-test state unavailable; reconstructing from best checkpoint: "
                  f"{args.pretest_state_file}", flush=True)
        model.load_state_dict(load_state_dict_safely(args.model_file, device))
        model.to(device)
        model.eval()
        rebuild_runtime_history_only(model, train_data)
        val_seed = args.eval_seed if args.eval_seed is not None else args.seed
        reconstructed_val_ap, reconstructed_val_auc, _ = evaluate_online(
            model, validation_data, num_nodes, args.transfer, args.nor, val_seed,
            state_seed=args.state_seed,
            progress_label="Validation reconstruction",
        )
        print(f"Reconstructed validation | AP: {reconstructed_val_ap:.4f} | "
              f"AUC: {reconstructed_val_auc:.4f}", flush=True)
        expected_val_ap = getattr(args, "expected_val_ap", None)
        expected_val_auc = getattr(args, "expected_val_auc", None)
        if expected_val_ap is not None and expected_val_auc is not None:
            print(f"Best-log validation      | AP: {expected_val_ap:.4f} | "
                  f"AUC: {expected_val_auc:.4f}", flush=True)
            if (abs(reconstructed_val_ap - float(expected_val_ap)) > 1e-6 or
                    abs(reconstructed_val_auc - float(expected_val_auc)) > 1e-6):
                if not args.allow_approximate_reconstruction:
                    raise RuntimeError(
                        "Reconstructed validation metrics do not match the selected best epoch. "
                        "Refusing to report a test score from an inconsistent temporal state. "
                        "Pass --allow_approximate_reconstruction 1 to continue explicitly."
                    )
                approximate_pretest_state = True
                print("WARNING: continuing from an approximate reconstructed validation state. "
                      "The test metrics are diagnostic and are not an exact reproduction of the old run.",
                      flush=True)

    test_seed = args.eval_seed + 1 if args.eval_seed is not None else args.seed
    print(f"Test negative sampling seed: {test_seed}", flush=True)
    start_t = time.time()
    ap, auc, f1 = evaluate_online(
        model, test_data, num_nodes, args.transfer, args.nor, test_seed,
        state_seed=(None if args.state_seed is None else args.state_seed + 1),
        progress_label="Test",
    )

    print(f"MODEL: {args.model_file}")
    result_label = "APPROXIMATE TEST" if approximate_pretest_state else "TEST"
    print(f"{result_label} AP:  {ap:.4f}")
    print(f"{result_label} AUC: {auc:.4f}")
    print(f"TEST F1:  {f1:.4f}")
    print(f"Time: {time.time() - start_t:.2f}s")
    print(f"Total time including replay: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    run_test()
