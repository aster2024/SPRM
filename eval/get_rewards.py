#!/usr/bin/env python
import argparse
import os
import sys
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


def save_rewards(args):
    """
    Compute reward for each candidate sample using the trained reward model,
    then save the computed rewards to an output file.

    For PRM mode: Save per-sample step rewards (raw list from reward model).
    For ORM mode: Save a single aggregated reward value per sample.
    """
    print("\n========== Starting Reward Computation ==========")
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

    sample0 = None
    for sample in ds:
        if len(sample.get("steps", [])) > 0:
            sample0 = sample
            break
    if sample0 is None:
        print("No samples with reasoning steps found; exiting.")
        return

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

    base_model = LinearRewardModel(feature_dim, disable_gate=args.disable_gate).to(device)
    if args.use_dim_reduction:
        dim_reduction = DimReduction(feature_dim, args.dim_reduction_dim).to(device)
        reward_model = RewardModelWithDimReduction(base_model, dim_reduction).to(device)
    else:
        reward_model = base_model

    checkpoint = torch.load(args.reward_model_load, map_location=device)
    reward_model.load_state_dict(checkpoint)
    reward_model.eval()
    print(f"Loaded reward model from {args.reward_model_load}")

    # Print parameter count for the reward model.
    num_params = sum(p.numel() for p in reward_model.parameters())
    print(f"Reward model parameter count: {num_params}")

    results = []
    total_time = 0.0
    for idx, sample in tqdm(enumerate(ds), total=len(ds), desc="Processing samples"):
        prompt = sample["prompt"]
        detailed_info = extract_detailed_info_for_reasoning_path(
            prompt,
            sample.get("steps", []),
            args.separator,
            args.layers,
            tokenizer,
            model_lm,
            apply_norm=args.apply_norm,
            to_cpu=False
        )
        token_features = get_token_features(detailed_info)
        if token_features is None:
            warnings.warn(f"Sample at index {idx} has no valid token features, skipping.")
            continue
        seq_len = token_features.size(0)
        token_features = token_features.unsqueeze(0).to(device)
        boundaries = detailed_info.get("boundaries", None)
        if boundaries is not None and len(boundaries) >= 2:
            step_boundaries = boundaries[1:]
        else:
            warnings.warn("No valid boundaries found, using the whole sequence as one step.")
            step_boundaries = [(0, seq_len)]

        result_entry = {
            "idx": sample.get("idx", idx),
            "prompt": prompt,
            "reference": sample.get("reference", ""),
            "correctness": int(sample.get("correctness", False))
        }
        start_time = time.time()
        if args.calc_mode == "prm":
            # PRM mode: Get raw reward for each step.
            step_rewards = reward_model(
                token_features, [seq_len],
                is_eval=True,
                boundaries=[step_boundaries],
                reward_mode="all"
            )
            step_rewards = step_rewards.tolist()
            result_entry["step_rewards"] = step_rewards
        elif args.calc_mode == "orm":
            # ORM mode: Get aggregated reward (a single number).
            reward_score = reward_model(
                token_features, [seq_len],
                is_eval=False
            )
            result_entry["reward"] = reward_score.item()
        else:
            raise ValueError(f"Invalid calc_mode: {args.calc_mode}")
        end_time = time.time()
        total_time += (end_time - start_time)
        results.append(result_entry)
    print(f"Total reward computation time: {total_time:.4f} seconds")

    # Save computed reward results into the output file.
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Reward results saved to {args.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Load trained reward model and compute rewards for candidate samples, then save rewards to a file."
    )
    # Model and dataset arguments.
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained LM model name or path (e.g., 'gpt2').")
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the dataset JSON file containing candidate outputs.")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator used to join reasoning steps (default: two newlines).")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Hidden layer indices to extract; if not provided, extract all layers.")
    # Reward model arguments.
    parser.add_argument("--method", type=str, choices=["ce", "hinge", "dpo", "infonca", "nca"],
                        default=None,
                        help="Reward method to evaluate. If set, reward model checkpoints are automatically loaded from the default directory.")
    parser.add_argument("--reward_model_load", type=str, default=None,
                        help="Path to a saved reward model checkpoint.")
    parser.add_argument("--calc_mode", type=str, choices=["prm", "orm"], required=True,
                        help="Calculation mode: 'prm' to save per-step rewards; 'orm' to save a single aggregated reward.")
    parser.add_argument("--disable_gate", action="store_true",
                        help="Disable gating mechanism in the reward model.")
    parser.add_argument("--apply_norm", action="store_true",
                        help="Apply normalization to hidden states before reward computation.")
    parser.add_argument("--use_dim_reduction", action="store_true",
                        help="Add a dimension reduction layer before the reward model if set.")
    parser.add_argument("--dim_reduction_dim", type=int, default=128,
                        help="Target dimension for dimension reduction (default: 128).")
    # Other parameters.
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to the output JSON file to save computed rewards.")
    args = parser.parse_args()

    if args.method is not None:
        if args.reward_model_load:
            raise ValueError("Either --method or --reward_model_load must be provided, not both.")

        norm_part = "norm_" if args.apply_norm else ""
        file_name = f"model/reward_model_{args.method}_{norm_part}{args.model_name.replace('/', '_')}.pt"
        args.reward_model_load = file_name
    else:
        if args.reward_model_load is None:
            raise ValueError("Either --method or --reward_model_load must be provided.")

    if args.output_file is None:
        if args.method:
            args.output_file = os.path.splitext(args.dataset_file)[0] + f"_extracted_rewards_{args.method}.json"
        else:
            args.output_file = os.path.splitext(args.dataset_file)[0] + "_extracted_rewards.json"

    save_rewards(args)


if __name__ == "__main__":
    main()
