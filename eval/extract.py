#!/usr/bin/env python
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import warnings

class LogitLensWrapper:
    def __init__(self, model, tokenizer, layers):
        """
        Initialize the wrapper with the model, tokenizer, and target layers.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.norm_fn, self.head_fn = self.get_norm_and_head(model)

    @staticmethod
    def get_norm_and_head(model):
        """
        Retrieve the appropriate normalization function and lm_head for logit lens extraction.
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

    def extract_logit_lens_sequence(self, inputs, device="cuda"):
        """
        Forward the input text through the model and extract the logit lens outputs
        for every token position in the sequence for each target layer.
        """
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states

        results = {}
        for layer in self.layers:
            if layer < len(hidden_states):
                hidden = hidden_states[layer]  # [batch, seq_len, hidden_size]
                hidden_norm = self.norm_fn(hidden)
                logits = self.head_fn(hidden_norm)  # [batch, seq_len, vocab_size]
                max_token_ids = torch.argmax(logits, dim=-1)[0]  # [seq_len]
                results[layer] = max_token_ids.tolist()
            else:
                results[layer] = None
        return results


# Tuned Lens Wrapper that uses tuned_lens.nn.unembed.Unembed for operations
class TunedLensWrapper:
    def __init__(self, model, tokenizer, layers):
        """
        Initialize the tuned lens wrapper with the model, tokenizer, and target layers.
        This implementation utilizes the tuned_lens.nn.unembed.Unembed module.
        """
        try:
            from tuned_lens.nn.unembed import Unembed
        except ImportError:
            raise ImportError("Please install tuned-lens (pip install tuned-lens) to use this option.")

        # Note: Unembed internally registers the model
        self.unembed = Unembed(model)
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers

    def extract_tuned_lens_sequence(self, inputs, device="cuda"):
        """
        Forward the input text through the model and extract tuned lens outputs
        by applying unembed.forward on the hidden states from the target layers.
        """
        # Get all hidden states from the model
        outputs = self.model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states  # tuple: (num_layers+1, batch, seq_len, hidden_size)

        results = {}
        for layer in self.layers:
            if layer < len(hidden_states):
                hidden = hidden_states[layer]  # [batch, seq_len, hidden_size]
                # Use tuned lens unembedding module to map hidden states to logits
                logits = self.unembed.forward(hidden)  # [batch, seq_len, vocab_size]
                max_token_ids = torch.argmax(logits, dim=-1)[0]  # [seq_len]
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


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate prompt and steps, then extract logit/tuned lens outputs for segments."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name or path, e.g., 'gpt2'.")
    parser.add_argument("--json_file", type=str, required=True,
                        help="Path to the JSON file containing task data with 'prompt' and 'steps'.")
    parser.add_argument("--layers", type=int, nargs="+", required=True,
                        help="Target layer indices for lens extraction, e.g., --layers 3 5 7.")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output JSON file to save extraction results.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run the model on ('cuda' or 'cpu').")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator inserted between prompt and steps.")
    parser.add_argument("--use_tuned_lens", action="store_true",
                        help="If set, use tuned lens (with tuned_lens.nn.unembed.Unembed) instead of the original logit lens.")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.json_file.replace(".json", "_lens.json")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, output_hidden_states=True)
    model.to(args.device)
    model.eval()

    if args.use_tuned_lens:
        lens_wrapper = TunedLensWrapper(model, tokenizer, args.layers)
    else:
        lens_wrapper = LogitLensWrapper(model, tokenizer, args.layers)

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_data = []
    for item in tqdm(data, desc="Processing items"):
        prompt = item.get("prompt", "")
        reasoning_steps = item.get("steps", [])  # list of step strings

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
                for i in range(start + 1, n - m - 1):
                    if seq[i:i + m] == subseq:
                        return i - 1, i + m + 2
                return None
            else:
                n, m = len(seq), len(subseq)
                for i in range(start, n - m + 1):
                    if seq[i:i + m] == subseq:
                        return i, i + m
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
                warnings.warn(f"Step {i + 1} subsequence not found: {reasoning_steps[i]}")

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

        new_item = item.copy()
        key_name = "lens"
        new_item[key_name] = segmented_results
        output_data.append(new_item)

    with open(args.output_file, "w", encoding="utf-8") as f_out:
        json.dump(output_data, f_out, ensure_ascii=False, indent=2)
        

if __name__ == "__main__":
    main()
