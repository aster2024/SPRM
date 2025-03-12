#!/usr/bin/env python
import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


def evaluate_all_models(args, reward_model_files):
    """
    Evaluate candidate groups using CombinedRewardModel.
    For each candidate, extract detailed info & token features, then for each reasoning step,
    compute a reward from the model (by passing the step’s token features and external step reward).
    Finally, aggregate per-step rewards (using min, mean, or max) to obtain the candidate reward.
    """
    print("\n========== Starting CombinedRewardModel Evaluation ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    ds = load_data(args.dataset_file)
    print(f"Loaded {len(ds)} candidate outputs from {args.dataset_file}.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model_lm.eval()

    sample0 = ds[0]
    if len(sample0.get("steps", [])) == 0:
        print("No reasoning steps in the first sample; exiting evaluation.")
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
    feature_dim = sum(hidden_states[layer].shape[-1] for layer in sorted_layers if hidden_states[layer] is not None)
    print(f"Reward model feature dimension: {feature_dim}")

    reward_models = {}
    for method, ckpt_file in reward_model_files.items():
        reward_model = CombinedRewardModel(feature_dim).to(device)
        checkpoint = torch.load(ckpt_file, map_location=device)
        reward_model.load_state_dict(checkpoint)
        reward_model.eval()
        reward_models[method] = reward_model
        print(f"Loaded CombinedRewardModel [{method}] from checkpoint: {ckpt_file}")
        print(f"Value of sigmoid(alpha) for [{method}]: {torch.sigmoid(reward_model.alpha).item()}")

    groups_dict = defaultdict(list)
    for sample in ds:
        groups_dict[sample["idx"]].append(sample)
    if args.max_samples:
        groups_dict = dict(list(groups_dict.items())[:args.max_samples])

    results = {method: [] for method in reward_models.keys()}
    total_time = 0.0

    for idx, candidate_list in tqdm(groups_dict.items(), desc="Processing candidate groups"):
        group_results = {method: [] for method in reward_models.keys()}
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
            if detailed_info is None:
                continue
            full_features = get_token_features(detailed_info)
            if full_features is None:
                continue
            seq_len = full_features.size(0)
            boundaries = detailed_info.get("boundaries", None)
            if boundaries is None or len(boundaries) < 2:
                warnings.warn("No valid boundaries found, using whole candidate as single step.")
                step_boundaries = [(0, seq_len)]
            else:
                step_boundaries = boundaries[1:]
            candidate_text = candidate.get("extracted_output", candidate.get("response", ""))
            correct = bool(candidate["correctness"])

            for method, reward_model in reward_models.items():
                start_time = time.time()
                if args.eval_mode == "orm":
                    if "score" not in candidate:
                        warnings.warn("Missing overall score in candidate; skipping candidate.")
                        continue
                    candidate_reward = candidate["score"]
                    overall_ext_reward_tensor = torch.tensor([float(candidate_reward)], dtype=torch.float32, device=device)
                    reward_score = reward_model(full_features.unsqueeze(0).to(device), [seq_len], overall_ext_reward_tensor)
                    candidate_reward = reward_score.item()
                else:
                    step_rewards = []
                    if "step_scores" not in candidate or len(candidate["step_scores"]) < len(step_boundaries):
                        warnings.warn("Missing step_scores in candidate; skipping candidate.")
                        continue
                    # In training, candidate features were extracted from tokens starting at boundaries[1][0].
                    # Use that as the base offset.
                    base = step_boundaries[0][0]
                    for i, sb in enumerate(step_boundaries):
                        rel_start = sb[0] - base
                        rel_end = sb[1] - base
                        if rel_end <= rel_start:
                            warnings.warn(f"Invalid step boundaries: {sb}; skipping step.")
                            continue
                        step_feature = full_features[rel_start:rel_end]
                        step_len = step_feature.size(0)
                        ext_score = candidate["step_scores"][i]
                        ext_reward_tensor = torch.tensor([float(ext_score)], dtype=torch.float32, device=device)
                        step_r = reward_model(step_feature.unsqueeze(0).to(device), [step_len], ext_reward_tensor)
                        step_rewards.append(step_r.item())
                    if len(step_rewards) == 0:
                        continue
                    if args.eval_mode == "min":
                        candidate_reward = min(step_rewards)
                    elif args.eval_mode == "max":
                        candidate_reward = max(step_rewards)
                    elif args.eval_mode == "mean":
                        candidate_reward = sum(step_rewards) / len(step_rewards)
                    else:
                        raise ValueError(f"Invalid eval_mode: {args.eval_mode}")
                end_time = time.time()
                total_time += end_time - start_time
                candidate_result = {
                    "prompt": prompt,
                    "reference": reference,
                    "extracted_output": candidate.get("extracted_output", candidate.get("response", "")),
                    "correctness": int(correct),
                    "reward": candidate_reward
                }
                group_results[method].append(candidate_result)
        for method in reward_models.keys():
            if group_results[method]:
                results[method].append(group_results[method])
    print(f"Total time for reward computation: {total_time:.4f} seconds")

    all_metrics = {}
    for method, groups in results.items():
        if len(groups) == 0:
            print(f"No valid candidate results for method {method}")
            continue
        metrics = compute_metrics(groups, args.k_vals)
        print(f"\n======== Pass@k Evaluation Results for CombinedRewardModel [{method}] ========")
        for k, val in metrics.items():
            print(f"Pass@{k}: {val}%")
        all_metrics[method] = metrics
        if args.metrics_file is None:
            norm_part = "norm_" if args.apply_norm else ""
            metrics_file = f"res/eval_metrics_combined_{method}_{args.eval_mode}_{norm_part}{args.model_name.replace('/', '_')}.json"
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
        description="Evaluate candidate reasoning paths using the CombinedRewardModel."
    )
    # Model and dataset arguments.
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained LM model name or path (e.g., 'gpt2').")
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the dataset JSON file containing candidate outputs.")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator used to join reasoning steps (default: two newlines).")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Hidden layer indices to extract (if empty, extract all layers).")
    # Reward model arguments.
    parser.add_argument("--reward_model_load", type=str, default=None,
                        help="Path to a saved CombinedRewardModel checkpoint. Optional if --methods is provided.")
    parser.add_argument("--methods", type=str, nargs="+", choices=["ce", "hinge"],
                        default=None,
                        help="List of reward methods (for naming) to evaluate. If set, checkpoints are loaded from the default directory.")
    parser.add_argument("--apply_norm", action="store_true",
                        help="Apply normalization to hidden states before reward computation.")
    parser.add_argument("--use_dim_reduction", action="store_true",
                        help="(Not used in this evaluation.)")
    # Pass@k evaluation arguments.
    parser.add_argument("--k_vals", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                        help="List of candidate numbers for pass@k evaluation (default: 1 2 4 8 16 32 64).")
    parser.add_argument("--eval_mode", type=str, required=True, choices=["min", "mean", "max", "orm"],
                        help="Evaluation mode: aggregate step rewards with 'min', 'mean', 'max', or use overall candidate ('orm').")
    # Other parameters.
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="If set, save evaluation metrics to this JSON file (method name will be appended).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of prompt groups to evaluate (default: None).")
    args = parser.parse_args()

    reward_model_files = {}
    if args.methods is not None and len(args.methods) > 0:
        if args.reward_model_load:
            raise ValueError("Provide either --methods or --reward_model_load, not both.")
        norm_part = "norm_" if args.apply_norm else ""
        eval_mode_part = "orm" if args.eval_mode == "orm" else "prm"
        for method in args.methods:
            file_name = f"model/combined_reward_model_{eval_mode_part}_{method}_{norm_part}{args.model_name.replace('/', '_')}.pt"
            reward_model_files[method] = file_name
    else:
        if args.reward_model_load is None:
            raise ValueError("Either --methods or --reward_model_load must be provided.")
        reward_model_files["single"] = args.reward_model_load

    evaluate_all_models(args, reward_model_files)

if __name__ == "__main__":
    main()
