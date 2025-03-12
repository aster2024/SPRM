#!/usr/bin/env python
import argparse
import json
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import pickle

def segment_token_ids(full_token_id_list, boundaries):
    """
    Segment token id sequence based on boundaries
    """
    segments = []
    for start, end in boundaries:
        segments.append(full_token_id_list[start:end])
    return segments

def extract_detailed_info_for_reasoning_path(prompt, reasoning_steps, args, tokenizer, model):
    """
    Process a reasoning path (one or multiple steps),
    construct dialogue and input, extract hidden states, boundaries, and segmented token information.
    Raise an exception if the corresponding prompt/step cannot be matched to skip this instance.
    """
    # Construct dialogue (prompt and assistant's response, response content is each step concatenated with separator)
    conversation = [
       {"role": "user", "content": prompt},
       {"role": "assistant", "content": args.separator.join(reasoning_steps)}
    ]
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    step_ids_list = [tokenizer.encode(step, add_special_tokens=False) for step in reasoning_steps]

    def find_subsequence(seq, subseq, trim_first_and_last=True, start=0):
        """
        Find the matching position of subseq in seq. If trim_first_and_last is True, remove the first and last token of subseq first.
        Return None if not found, otherwise return (start, end) index
        """
        if trim_first_and_last:
            if len(subseq) < 3:
                raise ValueError("Subsequence too short to remove first and last, skipping this instance.")
            trimmed = subseq[1:-2]
            n = len(seq)
            m = len(trimmed)
            for i in range(start+1, n - m - 1):
                if seq[i:i + m] == trimmed:
                    return i - 1, i + m + 2
            return None
        else:
            n = len(seq)
            m = len(subseq)
            for i in range(start, n - m + 1):
                if seq[i:i + m] == subseq:
                    return i, i + m
            return None

    last_end = 0
    boundaries = []
    token_ids_list = inputs["input_ids"][0].tolist()

    prompt_bounds = find_subsequence(token_ids_list, prompt_ids, trim_first_and_last=True, start=last_end)
    if prompt_bounds is None:
        raise ValueError(f"Could not match token sequence for prompt: {prompt}")
    boundaries.append(prompt_bounds)
    last_end = prompt_bounds[1]

    for idx, step_ids in enumerate(step_ids_list):
        step_bounds = find_subsequence(token_ids_list, step_ids, trim_first_and_last=True, start=last_end)
        if step_bounds is None:
            raise ValueError(f"Could not match token sequence for step {idx+1}: {reasoning_steps[idx]}")
        boundaries.append(step_bounds)
        last_end = step_bounds[1]

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # tuple, each element shape is (batch, seq_length, hidden_dim)

    if args.layers is None:
        layers_to_save = list(range(len(hidden_states)))
    else:
        layers_to_save = args.layers

    selected_hidden_states = {}
    with torch.no_grad():
        for layer in layers_to_save:
            if layer < len(hidden_states):
                selected_hidden_states[layer] = hidden_states[layer].detach().cpu()
            else:
                selected_hidden_states[layer] = None

    # Segment the entire input token id sequence based on boundaries
    segments = segment_token_ids(token_ids_list, boundaries)

    detailed_info = {
        "conversations": conversation,
        "inputs": {k: v.detach().cpu().tolist() for k, v in inputs.items()},
        "hidden_states": selected_hidden_states,
        "boundaries": boundaries,
        "segments": segments
    }
    return detailed_info

def process_data(args):
    ds = load_dataset("Windy0822/ultrainteract_math_rollout", split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    summary_samples = []
    detailed_samples = []

    for i, sample in enumerate(tqdm(ds, desc="Processing samples")):
        prompt = sample.get("prompt", "")
        raw_paths = sample.get("steps", [])
        if not isinstance(raw_paths, list):
            print(f"Skipping sample {i}: steps field format error.")
            continue

        try:
            detailed_info_paths = []
            for reasoning_steps in raw_paths:
                if not isinstance(reasoning_steps, list):
                    raise ValueError("reasoning steps format is not list.")
                info = extract_detailed_info_for_reasoning_path(prompt, reasoning_steps, args, tokenizer, model)
                detailed_info_paths.append(info)
        except Exception as e:
            print(f"Skipping sample {i}: {str(e)}")
            continue

        output_sample = {
            "prompt": prompt,
            "reference": sample.get("reference", ""),
            "dataset": sample.get("dataset", ""),
            "completions": sample.get("completions", []),
            "correctness": sample.get("correctness", []),
            "steps": raw_paths
        }
        summary_samples.append(output_sample)

        detailed_sample = {
            "prompt": prompt,
            "steps": raw_paths,
            "detailed_paths": detailed_info_paths
        }
        detailed_samples.append(detailed_sample)

        # Write to file every args.write_interval samples or at the end
        if (i + 1) % args.write_interval == 0 or i == len(ds) - 1 or (args.max_samples and i >= args.max_samples - 1):
            # Write summary data (JSON format)
            if os.path.exists(args.output_file):
                with open(args.output_file, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        data = []
            else:
                data = []
            data.extend(summary_samples)
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            summary_samples = []

            # Write detailed data (pickle format)
            with open(args.detailed_output_file, "wb") as f:
                pickle.dump(detailed_samples, f)

        if args.max_samples and i >= args.max_samples - 1:
            break

def main():
    parser = argparse.ArgumentParser(
        description="Process Windy0822/ultrainteract_math_rollout dataset: concatenate prompt with each reasoning path, extract and save detailed information (dialogue, inputs, hidden states, boundaries, etc.)."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained model name or path, e.g., 'gpt2'")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save summary data JSON file")
    parser.add_argument("--detailed_output_file", type=str, required=True,
                        help="Path to save detailed data (dialogue, inputs, hidden states, boundaries, etc.) file (pickle format)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to save hidden states, if not provided, save all layers")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator used to connect prompt and each step (default two newlines)")
    parser.add_argument("--write_interval", type=int, default=100,
                        help="Write to file after processing how many samples (default 100)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (default process all samples)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.detailed_output_file) or ".", exist_ok=True)

    process_data(args)
    print(f"Summary data saved to {args.output_file}")
    print(f"Detailed data saved to {args.detailed_output_file}")

if __name__ == "__main__":
    main()
