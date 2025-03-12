#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import torch
import json
import time
import os
import argparse
from tqdm import tqdm
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *

def parse_args():
    """
    Parses command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Loads a JSON file containing samples, computes rewards using a reward model, and outputs a new JSON file with a 'score' field."
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--reward_name_or_path", type=str,
                        required=True,
                        help="Path or name of the reward model.")
    parser.add_argument("--input_file", type=str,
                        required=True,
                        help="Path to the input JSON file containing the samples.")
    parser.add_argument("--output_file", type=str,
                        default=None,
                        help="Path to the output JSON file with updated samples.")
    parser.add_argument("--max_samples", type=int,
                        default=None,
                        help="Maximum number of samples to process.")
    parser.add_argument("--model_type", type=str,
                        choices=["Eurus", "Armo", "Skywork"], required=True,
                        help="Model type (for formatting answer).")

    return parser.parse_args()


def compute_reward(sample, model, tokenizer, device, model_type):
    """
    Computes the reward score for a given sample.
    Constructs the prompt with [INST] tokens using apply_chat_template, and gets the scalar reward output from the model.
    """
    prompt = sample["prompt"]
    response = sample["response"]

    if model_type in ["Armo", "Skywork"]:
        conversation = [{"content": prompt, "role": "user"}, {"content": response, "role": "assistant"}]
        input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(device)
    elif model_type == "Eurus":
        input_text = f"[INST] {prompt} [/INST] {response}"
        inputs = tokenizer(input_text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        raise ValueError(f"Unknown model type: {model_type}")


    with torch.no_grad():
        if model_type == "Eurus":
            score_value = model(**inputs).item()
        elif model_type == "Armo":
            score_value = model(input_ids).score.float().item()
        elif model_type == "Skywork":
            score_value = model(input_ids).logits[0][0].item()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return score_value



def process_samples(args, model, tokenizer, samples, device):
    """
    Processes each sample in the list and computes the reward.
    """
    total_time = 0.0
    updated_samples = []

    for sample in tqdm(samples, desc="Processing samples"):
        start_time = time.time()
        score = compute_reward(sample, model, tokenizer, device, args.model_type)
        total_time += time.time() - start_time

        sample["score"] = score
        updated_samples.append(sample)

        if args.max_samples is not None and len(updated_samples) >= args.max_samples:
            break

    print(f"Total time taken: {total_time:.4f} seconds")
    return updated_samples


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.output_file is None:
        args.output_file = os.path.splitext(args.input_file)[0] + f"_with_rewards_orm{args.reward_name_or_path.replace('/', '_')}.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_data(args.input_file)

    print(f"Loaded {len(samples)} samples")

    print("Loading reward model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.reward_name_or_path)
    if args.model_type == "Eurus":
        model = AutoModel.from_pretrained(
            args.reward_name_or_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    elif args.model_type == "Armo":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_name_or_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    elif args.model_type == "Skywork":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.reward_name_or_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation="flash_attention_2", num_labels=1
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    model.to(device)
    model.eval()

    updated_samples = process_samples(args, model, tokenizer, samples, device)

    with open(args.output_file, "w", encoding="utf-8") as outfile:
        for sample in updated_samples:
            outfile.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Processing complete. Output saved to {args.output_file}")


if __name__ == "__main__":
    main()
