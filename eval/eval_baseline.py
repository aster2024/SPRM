#!/usr/bin/env python
import argparse
import json
import os
import sys
import warnings
from collections import defaultdict
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


def aggregate_step_scores(step_scores, agg_method):
    """
    Given a list of step scores (external rewards), aggregate them using the specified aggregation method.
    """
    if not step_scores:
        return 0.0
    if agg_method == "min":
        return min(step_scores)
    elif agg_method == "max":
        return max(step_scores)
    elif agg_method == "mean":
        return sum(step_scores) / len(step_scores)
    else:
        raise ValueError(f"Invalid aggregation method: {agg_method}")


def evaluate_baseline(args):
    """
    Evaluate candidate groups using only external scores.

    In 'orm' mode, each sample is expected to have a 'score' field, which is used directly as the candidate reward.
    In other modes ('min', 'mean', 'max'), each candidate aggregates its 'step_scores' using the chosen method.
    Then, pass@k accuracy is computed over prompt groups.
    """
    print("\n========== Starting Baseline Evaluation ==========")
    set_seed(args.seed)
    ds = load_data(args.dataset_file)
    print(f"Loaded {len(ds)} candidate outputs from {args.dataset_file}.")

    groups_dict = defaultdict(list)
    for sample in ds:
        groups_dict[sample["idx"]].append(sample)
    if args.max_samples:
        groups_dict = dict(list(groups_dict.items())[:args.max_samples])

    results = []
    eval_mode = args.eval_mode.lower()
    for idx, candidate_list in tqdm(groups_dict.items(), desc="Processing candidate groups"):
        group_candidates = []
        for candidate in candidate_list:
            if eval_mode == "orm":
                if "score" not in candidate:
                    warnings.warn("Candidate missing 'score' field in orm mode; skipping candidate.")
                    continue
                agg_score = float(candidate["score"])
            else:
                step_scores = candidate.get("step_scores", [])
                if not step_scores:
                    warnings.warn("Candidate missing step_scores; skipping candidate.")
                    continue
                agg_score = aggregate_step_scores(step_scores, eval_mode)
            result = {
                "prompt": candidate["prompt"],
                "reference": candidate.get("reference", ""),
                "extracted_output": candidate.get("extracted_output", candidate.get("response", "")),
                "correctness": int(candidate["correctness"]),
                "reward": agg_score
            }
            group_candidates.append(result)
        if group_candidates:
            results.append(group_candidates)

    metrics = compute_metrics(results, args.k_vals)
    print("\n========== Baseline Pass@k Evaluation Results ==========")
    for k, val in metrics.items():
        print(f"Pass@{k}: {val}%")

    if args.metrics_file:
        with open(args.metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Baseline evaluation metrics saved to {args.metrics_file}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation using external scores aggregation without a reward model."
    )
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the dataset JSON file containing candidate outputs.")
    parser.add_argument("--k_vals", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                        help="List of candidate numbers for pass@k evaluation (default: 1 2 4 8 16 32 64).")
    parser.add_argument("--eval_mode", type=str, required=True, choices=["min", "mean", "max", "orm"],
                        help="Evaluation mode: 'orm' uses candidate['score'] while 'min', 'mean', or 'max' aggregate step_scores.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="If set, save evaluation metrics to this file (JSON format).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of prompt groups to evaluate (default: None).")
    args = parser.parse_args()

    evaluate_baseline(args)


if __name__ == "__main__":
    main()
