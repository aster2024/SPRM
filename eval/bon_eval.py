#!/usr/bin/env python
import argparse
import json
import warnings
import os
import sys
from collections import defaultdict
from tqdm import tqdm
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


def evaluate_all_models(args, reward_model_files):
    """
    Evaluate candidate groups using *multiple* reward models.
    For each candidate, immediately after extracting detailed info (and token features),
    the reward is computed with all reward models.
    """
    print("\n========== Starting Evaluation ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    ds = load_data(args.dataset_file)
    print(f"Loaded {len(ds)} candidate outputs from {args.dataset_file}.")

    # Load tokenizer and LM (model_lm) once for extraction.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model_lm.eval()

    # Use the first sample to determine the feature dimension.
    sample0 = ds[0]
    if len(sample0.get("steps", [])) == 0:
        print("No reasoning steps in the first sample, exiting evaluation.")
        return None

    detailed_info0 = extract_detailed_info_for_reasoning_path(
        sample0["prompt"],
        sample0["steps"],
        args.separator,
        args.layers,
        tokenizer,
        model_lm,
        apply_norm=args.apply_norm,
        to_cpu=False
    )
    hidden_states = detailed_info0["hidden_states"]
    sorted_layers = sorted(hidden_states.keys(), key=lambda x: int(x))
    feature_dim = 0
    for layer in sorted_layers:
        if hidden_states[layer] is not None:
            feature_dim += hidden_states[layer].shape[-1]
    print(f"Reward model feature dimension: {feature_dim}")

    # For each reward method (key in reward_model_files), build and load the reward model.
    reward_models = {}
    for method, ckpt_file in reward_model_files.items():
        base_model = LinearRewardModel(feature_dim, disable_gate=args.disable_gate).to(device)
        if args.use_dim_reduction:
            dim_reduction = DimReduction(feature_dim, args.dim_reduction_dim).to(device)
            reward_model = RewardModelWithDimReduction(base_model, dim_reduction).to(device)
        else:
            reward_model = base_model
        checkpoint = torch.load(ckpt_file, map_location=device)
        reward_model.load_state_dict(checkpoint)
        reward_model.eval()
        reward_models[method] = reward_model
        print(f"Loaded reward model [{method}] from checkpoint: {ckpt_file}")

    # Group candidates by prompt index.
    groups_dict = defaultdict(list)
    for sample in ds:
        groups_dict[sample["idx"]].append(sample)
    if args.max_samples:
        groups_dict = dict(list(groups_dict.items())[:args.max_samples])

    # Initialize results dictionary per method.
    results = { method: [] for method in reward_models.keys() }
    total_time = 0.0

    for idx, candidate_list in tqdm(groups_dict.items(), desc="Processing candidate groups"):
        # For each group (same prompt), maintain a separate list per reward model.
        group_results = { method: [] for method in reward_models.keys() }
        for candidate in candidate_list:
            prompt = candidate["prompt"]
            reference = candidate.get("reference", "")
            detailed_info = extract_detailed_info_for_reasoning_path(
                prompt,
                candidate.get("steps", []),
                args.separator,
                args.layers,
                tokenizer,
                model_lm,
                apply_norm=args.apply_norm,
                to_cpu=False
            )
            token_features = get_token_features(detailed_info)
            if token_features is None:
                continue
            seq_len = token_features.size(0)
            # Move token features to device and add a batch dimension.
            token_features = token_features.unsqueeze(0).to(device)
            boundaries = detailed_info.get("boundaries", None)
            if boundaries is not None and len(boundaries) >= 2:
                step_boundaries = boundaries[1:]
            else:
                warnings.warn("No valid boundaries found, using the whole sequence.")
                step_boundaries = [(0, seq_len)]
            candidate_text = args.separator.join(candidate.get("steps", []))
            correct = bool(candidate["correctness"])

            # For each reward model, immediately compute reward.
            for method, reward_model in reward_models.items():
                start_time = time.time()
                if args.eval_mode in ["min", "mean", "max"]:
                    reward_score = reward_model(
                        token_features, [seq_len],
                        is_eval=True,
                        boundaries=[step_boundaries],
                        reward_mode=args.eval_mode
                    )
                elif args.eval_mode == "orm":
                    reward_score = reward_model(
                        token_features, [seq_len],
                        is_eval=False  # In orm mode, the reward model would act like in training mode
                    )
                else:
                    raise ValueError(f"Invalid evaluation mode: {args.eval_mode}")
                end_time = time.time()
                total_time += end_time - start_time
                reward_score = reward_score.item()
                candidate_result = {
                    "prompt": prompt,
                    "reference": reference,
                    "extracted_output": candidate.get("extracted_output", candidate_text),
                    "correctness": int(correct),
                    "reward": reward_score
                }
                group_results[method].append(candidate_result)

        for method in reward_models.keys():
            if group_results[method]:
                results[method].append(group_results[method])
    # End for each group.
    print(f"Total time taken for reward computation: {total_time:.4f} seconds")

    # Compute and print pass@k metrics for each reward model.
    all_metrics = {}
    for method, groups in results.items():
        if len(groups) == 0:
            print(f"No valid candidate results for method {method}")
            continue
        metrics = compute_metrics(groups, args.k_vals)
        print(f"\n======== Pass@k Evaluation Results for model [{method}] ========")
        for k, val in metrics.items():
            print(f"Pass@{k}: {val}%")
        print("===========================================")
        all_metrics[method] = metrics
        # Save metrics per model.
        if args.metrics_file is None:
            norm_part = "norm_" if args.apply_norm else ""
            metrics_file = f"res/eval_metrics_{method}_{args.eval_mode}_{norm_part}{args.model_name.replace('/', '_')}_2.json"
        else:
            base, ext = os.path.splitext(args.metrics_file)
            metrics_file = f"{base}_{method}{ext}"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation metrics saved to {metrics_file}")

    if len(all_metrics) > 1:
        print("\n===== Overall Evaluation Metrics =====")
        for method, m in all_metrics.items():
            print(f"Method [{method}]:")
            for k, v in m.items():
                print(f"   Pass@{k}: {v}%")
        print("=======================================")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate candidate reasoning paths using multiple trained PRM reward models"
    )
    # Model and dataset related arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained LM model name or path (e.g., 'gpt2').")
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the dataset JSON file containing candidate outputs.")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator used to join reasoning steps (default: two newlines).")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Hidden layer indices to extract; if empty, extract all layers.")
    # Reward model arguments
    parser.add_argument("--reward_model_load", type=str, default=None,
                        help="Path to a saved reward model checkpoint. Optional if --methods is provided.")
    parser.add_argument("--methods", type=str, nargs="+", choices=["ce", "hinge", "dpo", "infonca", "nca"],
                        default=None,
                        help="List of reward methods to evaluate. If set, reward model checkpoints are automatically loaded from the default directory.")
    parser.add_argument("--disable_gate", action="store_true",
                        help="Disable gating mechanism in the reward model.")
    parser.add_argument("--apply_norm", action="store_true",
                        help="Apply normalization to hidden states before reward computation.")  # This does not lead to better performance by my observation
    parser.add_argument("--use_dim_reduction", action="store_true",
                        help="Add a dimension reduction layer before the reward model if set.")
    parser.add_argument("--dim_reduction_dim", type=int, default=128,
                        help="Target dimension for dimension reduction (default: 128).")
    # Pass@k evaluation arguments
    parser.add_argument("--k_vals", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                        help="List of candidate numbers for pass@k evaluation (default: 1 2 4 8 16 32 64).")
    parser.add_argument("--eval_mode", type=str, default="min", choices=["min", "mean", "max", "orm"],
                        help="Aggregation mode for step rewards in evaluation ('min', 'mean', 'max' or 'orm', default: min).")
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="If set, save evaluation metrics to this JSON file. For multiple evaluations, the method name will be appended.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (default: None).")
    args = parser.parse_args()

    # Determine evaluation mode: either use --methods or a single reward_model_load.
    reward_model_files = {}
    if args.methods is not None and len(args.methods) > 0:
        if args.reward_model_load:
            raise ValueError("Either --methods or --reward_model_load must be provided, not both.")

        norm_part = "norm_" if args.apply_norm else ""
        for method in args.methods:
            # Expect reward checkpoints to be stored with this naming convention:
            file_name = f"model/reward_model_{method}_{norm_part}{args.model_name.replace('/', '_')}.pt"
            reward_model_files[method] = file_name
    else:
        if args.reward_model_load is None:
            raise ValueError("Either --methods or --reward_model_load must be provided.")
        reward_model_files["single"] = args.reward_model_load

    evaluate_all_models(args, reward_model_files)


if __name__ == "__main__":
    main()
