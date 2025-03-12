#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import json
import time
import os
import argparse
from tqdm import tqdm
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import set_seed


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Load a JSON file containing samples, compute step rewards using a reward model, and output a new JSON file with added step_scores."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--reward_name_or_path", type=str,
                        default="PRIME-RL/EurusPRM-Stage2",
                        help="Path or name of the reward model.")
    parser.add_argument("--ref_name_or_path", type=str,
                        default="Qwen/Qwen2.5-Math-7B-Instruct",
                        help="Path or name of the reference model.")
    parser.add_argument("--input_file", type=str,
                        required=True,
                        help="Path to the input JSON file containing the samples.")
    parser.add_argument("--output_file", type=str,
                        default=None,
                        help="Path to the output JSON file with updated samples.")
    parser.add_argument("--max_samples", type=int,
                        default=None,
                        help="Maximum number of samples to process.")
    parser.add_argument("--coef", type=float, default=0.001,
                        help="Coefficient for the reward calculation.")

    return parser.parse_args()


def compute_step_rewards(args, sample, model, tokenizer, ref_model, device):
    """
    Compute reward scores for each step in the sample.
    This function now follows the official demo's logic for reward calculation.
    """
    prompt = sample["prompt"]
    steps = sample["steps"]
    step_scores = []
    total_time = 0.0

    input_ids = tokenizer.apply_chat_template([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "\n\n".join(steps)},
    ], tokenize=True, add_generation_prompt=False, return_tensors='pt').to(device)

    attention_mask = input_ids != tokenizer.pad_token_id

    step_last_tokens = []
    for step_num in range(0, len(steps) + 1):
        conv = tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "\n\n".join(steps[:step_num])},
        ], tokenize=False, add_generation_prompt=False)
        conv = conv.strip()
        if step_num != 0 and step_num != len(steps):
            conv += '\n\n'
        currect_ids = tokenizer.encode(conv, add_special_tokens=False)
        step_last_tokens.append(len(currect_ids) - 2)

    inputs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': input_ids}
    label_mask = torch.tensor([[0] * step_last_tokens[0] + [1] * (input_ids.shape[-1] - step_last_tokens[0])]).to(device)
    step_last_tokens = torch.tensor([step_last_tokens]).to(device)

    def get_logps(model, inputs):
        logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
        labels = inputs['labels'][:, 1:].clone().long()
        logits = logits[:, :-1, :]
        labels[labels == -100] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return per_token_logps

    start_time = time.time()
    with torch.no_grad():
        per_token_logps = get_logps(model, inputs)
        ref_per_token_logps = get_logps(ref_model, inputs)

    raw_reward = per_token_logps - ref_per_token_logps
    beta_reward = args.coef * raw_reward * label_mask[:, 1:]
    beta_reward = beta_reward.cumsum(-1)
    beta_reward = beta_reward.gather(dim=-1, index=step_last_tokens[:, 1:])
    total_time += time.time() - start_time

    step_scores = beta_reward.squeeze().tolist()
    if isinstance(step_scores, float):
        step_scores = [step_scores]

    sample["step_scores"] = step_scores
    return sample, total_time


def process_samples(args, model, tokenizer, ref_model, samples, device):
    """
    Process each sample in the list and compute the step rewards.
    """
    total_time = 0.0
    updated_samples = []
    for sample in tqdm(samples, desc="Processing samples"):
        updated_sample, compute_time = compute_step_rewards(args, sample, model, tokenizer, ref_model, device)
        updated_samples.append(updated_sample)
        total_time += compute_time

    print(f"Total time taken: {total_time:.4f} seconds")
    return updated_samples


def main():
    args = parse_args()

    set_seed(args.seed)

    if args.output_file is None:
        args.output_file = os.path.splitext(args.input_file)[0] + f"_with_rewards_implicit_prm.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.input_file, "r", encoding="utf-8") as infile:
        samples = json.load(infile)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print("Loading reward model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.reward_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_name_or_path, torch_dtype=torch.bfloat16, device_map="auto"
    )

    updated_samples = process_samples(args, model, tokenizer, ref_model, samples, device)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Processing complete.  Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
