#!/usr/bin/env python
import argparse
import json
import warnings
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

from utils import *

def get_token_features(detailed_info):
    """
    Extract token features from the detailed_info.  Returns all token features.
    """
    hidden_states = detailed_info["hidden_states"]
    if not hidden_states:
        warnings.warn("No hidden states found.")
        return None, None

    boundaries = detailed_info.get("boundaries", None)
    if boundaries is not None and len(boundaries) >= 2:
        reward_start = boundaries[1][0]
        reward_end = boundaries[-1][1]
    else:
        warnings.warn("No boundaries found, using all tokens.")
        first_layer_key = sorted(hidden_states.keys(), key=lambda x: int(x))[0]
        token_seq_length = hidden_states[first_layer_key][0].shape[0]
        reward_start = 0
        reward_end = token_seq_length

    sorted_layers = sorted(hidden_states.keys(), key=lambda x: int(x))
    layer_tensors = []
    for layer in sorted_layers:
        tensor = hidden_states[layer][0].clone().detach().to(torch.float32)
        tensor = tensor[reward_start:reward_end]
        layer_tensors.append(tensor)

    if not layer_tensors:
        warnings.warn("No valid token features extracted.")
        return None, None

    token_features = torch.cat(layer_tensors, dim=-1)
    token_indices = torch.arange(reward_start, reward_end)

    return token_features, token_indices

def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden state representations, perform LDA, and plot distributions."
    )
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the dataset JSON file.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained LM model name or path.")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator for reasoning steps.")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Indices of hidden layers to extract.")
    parser.add_argument("--apply_norm", action="store_true",
                        help="Apply normalization to hidden states.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of candidate samples to process.")
    parser.add_argument("--output_plot", type=str, default="figs/hidden_states_lda.png",
                        help="Filename to save the output plot.")
    args = parser.parse_args()

    dataset = load_data(args.dataset_file)
    print(f"Loaded {len(dataset)} candidate outputs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    lm_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    lm_model.eval()

    all_candidate_features = []
    all_labels = []
    num_processed = 0

    for candidate in tqdm(dataset, desc="Processing candidates"):
        if not candidate.get("steps"):
            continue

        prompt = candidate["prompt"]
        detailed_info = extract_detailed_info_for_reasoning_path(
            prompt=prompt,
            reasoning_steps=candidate.get("steps", []),
            separator=args.separator,
            layers=args.layers,
            tokenizer=tokenizer,
            model=lm_model,
            apply_norm=args.apply_norm,
            to_cpu=True
        )

        token_features, _ = get_token_features(detailed_info)
        if token_features is None:
            continue

        candidate_features = token_features.mean(dim=0).cpu().numpy()
        all_candidate_features.append(candidate_features)

        correctness = int(candidate.get("correctness", 0))
        all_labels.append(correctness)

        num_processed += 1
        if args.max_samples is not None and num_processed >= args.max_samples:
            break

    if num_processed == 0:
        print("No valid samples processed. Exiting.")
        return

    features_np = np.stack(all_candidate_features)
    labels_np = np.array(all_labels)

    lda = LinearDiscriminantAnalysis(n_components=1)
    features_1d = lda.fit_transform(features_np, labels_np).flatten()

    correct_idx = labels_np == 1
    incorrect_idx = labels_np == 0

    if not np.any(correct_idx):
        print("Warning: No correct samples found.")
        correct_points = np.array([])
    else:
        correct_points = features_1d[correct_idx]

    if not np.any(incorrect_idx):
        print("Warning: No incorrect samples found.")
        incorrect_points = np.array([])
    else:
        incorrect_points = features_1d[incorrect_idx]


    plt.figure(figsize=(10, 6))

    if correct_points.size > 0:
        sns.histplot(x=correct_points, color='darkblue', label="Correct", kde=True, element="step")
    if incorrect_points.size > 0:
        sns.histplot(x=incorrect_points, color='orange', label="Incorrect", kde=True, element="step")

    plt.xlabel("LDA Component 1")
    plt.ylabel("Count")
    plt.title("LDA Projection and Distribution of Averaged Reasoning Step Hidden States")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Plot saved to {args.output_plot}")
    plt.show()

if __name__ == "__main__":
    main()
