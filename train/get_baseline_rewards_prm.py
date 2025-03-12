#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Append parent directory to import set_seed from utils
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import set_seed


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Load a JSON file containing samples, compute step rewards "
                    "for each rollout using a reward model, and output a new JSONL file with added step_scores."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--reward_name_or_path",
        type=str,
        required=True,
        help="Path or name of the reward model.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input JSON file containing the samples.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to the output JSONL file with updated samples.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["Mistral", "Deepseek"],
        required=True,
        help="Model type (for splitting answer if needed).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per rollout (optional).",
    )
    return parser.parse_args()


def compute_step_rewards(args, sample, model, tokenizer, candidate_tokens, device):
    """
    Compute reward scores for each step in all rollouts of the sample.

    For each rollout, for the first step the prompt is concatenated with the step text,
    while for subsequent steps only the step text is used. The reward score is derived from
    the model logits of the '+' token (whose id is candidate_tokens[0]).

    The sample now contains a "steps" field that is a list of lists. Each inner list
    corresponds to the steps for a rollout.
    """
    prompt = sample["prompt"]

    rollout_scores_all = []

    for rollout_steps in sample["steps"]:
        rollout_scores = []
        for i, step in enumerate(rollout_steps):
            if i == 0:
                text = prompt + " " + step
            else:
                text = step

            conversation = [
                {"content": text, "role": "user"},
                {"content": "+", "role": "assistant"},
            ]

            input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)

            with torch.no_grad():
                logits = model(input_ids).logits[:, -3, candidate_tokens]
                scores = torch.softmax(logits, dim=-1)[:, 0]
                score_value = scores[0].detach().to("cpu", dtype=torch.float32).item()

            rollout_scores.append(score_value)

            if args.max_samples is not None and i + 1 >= args.max_samples:
                break

        rollout_scores_all.append(rollout_scores)

    # Add a new field "step_scores" to the sample. Each element in the list corresponds to the scores for one rollout.
    sample["step_scores"] = rollout_scores_all
    if "lens" in sample:
        del sample["lens"]
    return sample


def process_samples(args, model, tokenizer, samples, device):
    """
    Process each sample in the list and compute the step rewards.
    """
    # Encode '+' and '-' tokens to get candidate token IDs.
    plus_token_id = tokenizer.encode("+")[-1]
    minus_token_id = tokenizer.encode("-")[-1]
    candidate_tokens = [plus_token_id, minus_token_id]

    updated_samples = []
    for sample in tqdm(samples, desc="Processing samples"):
        updated_sample = compute_step_rewards(args, sample, model, tokenizer, candidate_tokens, device)
        updated_samples.append(updated_sample)
    return updated_samples


def main():
    args = parse_args()

    set_seed(args.seed)

    if args.output_file is None:
        args.output_file = os.path.splitext(args.input_file)[0] + f"_with_rewards_type{args.model_type}.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.input_file, "r", encoding="utf-8") as infile:
        samples = json.load(infile)

    print("Loading reward model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.reward_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    updated_samples = process_samples(args, model, tokenizer, samples, device)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Processing complete. Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
