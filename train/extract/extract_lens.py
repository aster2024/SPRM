#!/usr/bin/env python
import argparse
import json
import os
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings


class LogitLensWrapper:
    """
    This wrapper extracts logit lens outputs from specified layers. It uses the
    model's normalization function and lm_head to convert hidden states into logits,
    and then extracts the predicted tokens (via argmax) at every token position.
    """

    def __init__(self, model, tokenizer, layers):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.norm_fn, self.head_fn = self.get_norm_and_head(model)

    @staticmethod
    def get_norm_and_head(model):
        """
        Retrieve an appropriate LayerNorm function and lm_head for logit lens extraction.
        """
        if model.config.is_encoder_decoder:
            pointer = model.decoder
        else:
            pointer = model

        if hasattr(pointer, "final_layer_norm"):
            norm_fn = pointer.final_layer_norm
        elif hasattr(pointer, "gpt_neox"):
            norm_fn = pointer.gpt_neox.final_layer_norm
        elif hasattr(pointer.model, "norm"):
            norm_fn = pointer.model.norm
        elif hasattr(pointer.model, "final_layernorm"):
            norm_fn = pointer.model.final_layernorm
        else:
            raise NotImplementedError("Could not find a suitable LayerNorm function.")

        if hasattr(model, "lm_head"):
            head_fn = model.lm_head
        elif hasattr(model, "embed_out"):
            head_fn = model.embed_out
        else:
            raise NotImplementedError("Could not find a suitable lm_head function.")

        return norm_fn, head_fn

    def extract_logit_lens_sequence(self, inputs):
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states

        results = {}
        for layer in self.layers:
            if layer < len(hidden_states):
                hidden = hidden_states[layer]
                hidden_norm = self.norm_fn(hidden)
                logits = self.head_fn(hidden_norm)
                max_token_ids = torch.argmax(logits, dim=-1)[0]
                results[layer] = max_token_ids.tolist()
            else:
                results[layer] = None
        return results


class TunedLensWrapper:
    """
    This wrapper extracts outputs using the tuned lens method.
    It utilizes the tuned_lens.nn.unembed.Unembed module to map hidden states to logits.
    """

    def __init__(self, model, tokenizer, layers):
        try:
            from tuned_lens.nn.unembed import Unembed
        except ImportError:
            raise ImportError("Please install tuned-lens (pip install tuned-lens) to use this option.")
        self.unembed = Unembed(model)
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers

    def extract_tuned_lens_sequence(self, inputs):
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states

        results = {}
        for layer in self.layers:
            if layer < len(hidden_states):
                hidden = hidden_states[layer]
                logits = self.unembed.forward(hidden)
                max_token_ids = torch.argmax(logits, dim=-1)[0]
                results[layer] = max_token_ids.tolist()
            else:
                results[layer] = None
        return results


def segment_token_ids(full_token_id_list, boundaries):
    """
    Segment the full token ID list according to specified boundaries.

    Args:
        full_token_id_list (list): List of token IDs for the full text.
        boundaries (list of tuple): Each tuple is (start, end) indices indicating a segment.

    Returns:
        list: A list of token ID lists, each corresponding to one segment.
    """
    segments = []
    for start, end in boundaries:
        segments.append(full_token_id_list[start:end])
    return segments


def extract_lens_for_reasoning_path(prompt, reasoning_steps, args, tokenizer, lens_wrapper):
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

    inputs = {k: v.to(lens_wrapper.model.device) for k, v in inputs.items()}

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    step_ids_list = [tokenizer.encode(step, add_special_tokens=False) for step in reasoning_steps]

    def find_subsequence(seq, subseq, trim_first_and_last=True, start=0):
        # Trim the first one token and last two tokens in the subsequence for better matching
        if trim_first_and_last:
            if len(subseq) < 3:
                warnings.warn("Subsequence too short to trim.")
                return None

            subseq = subseq[1:-2]
            n, m = len(seq), len(subseq)
            for i in range(start+1, n-m-1):
                if seq[i:i+m] == subseq:
                    return i-1, i+m+2
            return None
        else:
            n, m = len(seq), len(subseq)
            for i in range(start, n-m+1):
                if seq[i:i+m] == subseq:
                    return i, i+m
            return None

    last_end = 0
    boundaries = []
    prompt_bounds = find_subsequence(inputs['input_ids'][0].tolist(), prompt_ids, start=last_end)
    if prompt_bounds:
        boundaries.append(prompt_bounds)
        last_end = prompt_bounds[1]
    else:
        warnings.warn(f"Prompt subsequence not found: {prompt}")

    for i, step_ids in enumerate(step_ids_list):
        step_bounds = find_subsequence(inputs['input_ids'][0].tolist(), step_ids, start=last_end)
        if step_bounds:
            boundaries.append(step_bounds)
            last_end = step_bounds[1]
        else:
            warnings.warn(f"Step {i+1} subsequence not found: {reasoning_steps[i]}")

    if args.use_tuned_lens:
        full_seq_result = lens_wrapper.extract_tuned_lens_sequence(inputs)
    else:
        full_seq_result = lens_wrapper.extract_logit_lens_sequence(inputs)

    segmented_results = {}
    for layer, token_ids in full_seq_result.items():
        if token_ids is None:
            segmented_results[layer] = {"prompt": None, "steps": []}
            warnings.warn(f"Layer {layer} output is None.")
        else:
            segments = segment_token_ids(token_ids, boundaries)

            segmented_results[layer] = {
                "prompt": segments[0] if segments else None,
                "steps": segments[1:] if len(segments) > 1 else []
            }
    return segmented_results


def process_data(args):
    ds = load_dataset("Windy0822/ultrainteract_math_rollout", split="train")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    if args.use_tuned_lens:
        lens_wrapper = TunedLensWrapper(model, tokenizer, args.layers)
    else:
        lens_wrapper = LogitLensWrapper(model, tokenizer, args.layers)

    with open(args.output_file, "w") as f:
        json.dump([], f)

    processed_samples = []
    for i, sample in enumerate(tqdm(ds, desc="Processing samples")):
        prompt = sample.get("prompt", "")
        raw_paths = sample.get("steps", [])
        lens_paths = []

        for reasoning_steps in raw_paths:
            if isinstance(reasoning_steps, list):
                lens_result = extract_lens_for_reasoning_path(prompt, reasoning_steps, args, tokenizer, lens_wrapper)
                lens_paths.append(lens_result)
            else:
                lens_paths.append(None)

        output_sample = {
            "prompt": prompt,
            "reference": sample.get("reference", ""),
            "dataset": sample.get("dataset", ""),
            "completions": sample.get("completions", []),
            "correctness": sample.get("correctness", []),
            "steps": raw_paths,
            "lens": {"paths": lens_paths}
        }

        processed_samples.append(output_sample)

        # Write to file every args.write_interval samples or at the end
        if (i + 1) % args.write_interval == 0 or i == len(ds) - 1 or (args.max_samples and i >= args.max_samples - 1):
            with open(args.output_file, "r+") as f:
                data = json.load(f)
                data.extend(processed_samples)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.truncate()
            processed_samples = []  # Clear the list after writing

            if args.max_samples and i >= args.max_samples - 1:
                break


def main():
    parser = argparse.ArgumentParser(
        description="Process Windy0822/ultrainteract_math_rollout by concatenating the prompt and each reasoning path (steps), then extracting logit/tuned lens outputs."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pretrained model name or path, e.g., 'gpt2'.")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output JSON file to save processed data with lens results.")
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Target layer indices for lens extraction, e.g., --layers 3 5 7")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator to insert between the prompt and each step (default: two newlines).")
    parser.add_argument("--use_tuned_lens", action="store_true",
                        help="If set, use tuned lens instead of the original logit lens.")
    parser.add_argument("--write_interval", type=int, default=100,
                        help="Number of samples to process before writing to file (default: 100)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (default: process all samples)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    process_data(args)
    print(f"Processed data saved to {args.output_file}")


if __name__ == "__main__":
    main()

