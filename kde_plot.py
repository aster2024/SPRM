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
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
import seaborn as sns

from utils import extract_detailed_info_for_reasoning_path


def load_data(file_name):
    """
    Load the dataset from a JSON file.
    """
    if file_name.endswith('json'):
        with open(file_name, encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(file_name, encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    return data


def get_token_features(detailed_info):
    """
    Extract token features from the detailed_info.  Now returns all token features.
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

    if len(layer_tensors) == 0:
        warnings.warn("No valid token features extracted.")
        return None, None

    token_features = torch.cat(layer_tensors, dim=-1)
    token_indices = torch.arange(reward_start, reward_end)

    return token_features, token_indices


def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden state representations, perform PCA, and plot KDE contours."
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
    parser.add_argument("--output_plot", type=str, default="figs/hidden_states_kde.png",
                        help="Filename to save the output plot.")
    parser.add_argument("--pca_batch_size", type=int, default=10000,
                        help="Batch size for incremental PCA.")
    args = parser.parse_args()

    dataset = load_data(args.dataset_file)
    print(f"Loaded {len(dataset)} candidate outputs.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    lm_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    lm_model.eval()

    pca = IncrementalPCA(n_components=2, batch_size=args.pca_batch_size)
    all_labels = []
    num_processed = 0

    for candidate in tqdm(dataset, desc="Processing candidates"):
        if len(candidate.get("steps", [])) == 0:
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

        token_features, token_indices = get_token_features(detailed_info)
        if token_features is None:
            continue

        token_features = token_features.cpu().numpy()

        pca.partial_fit(token_features)

        correctness = int(candidate.get("correctness", 0))
        labels = np.full(token_features.shape[0], correctness)
        all_labels.append(labels)

        num_processed += 1
        if args.max_samples is not None and num_processed >= args.max_samples:
            break

    if num_processed == 0:
        print("No valid samples processed. Exiting.")
        return

    all_features_2d = []
    for candidate in tqdm(dataset, desc="Transforming with PCA"):
        if len(candidate.get("steps", [])) == 0:
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
        token_features = token_features.cpu().numpy()

        batch_2d = pca.transform(token_features)
        all_features_2d.append(batch_2d)


    all_features_2d = np.concatenate(all_features_2d, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("PCA transformation completed. Explained variance ratios:", pca.explained_variance_ratio_)

    correct_idx = all_labels == 1
    incorrect_idx = all_labels == 0
    correct_points = all_features_2d[correct_idx]
    incorrect_points = all_features_2d[incorrect_idx]

    plt.figure(figsize=(10, 8))

    if len(correct_points) > 1:
        sns.kdeplot(x=correct_points[:, 0], y=correct_points[:, 1],
                    levels=10, color='blue', linewidths=1, label="Correct")
    if len(incorrect_points) > 1:
        sns.kdeplot(x=incorrect_points[:, 0], y=incorrect_points[:, 1],
                    levels=10, color='red', linewidths=1, label="Incorrect")

    plt.title("PCA Projection and KDE of Reasoning Step Hidden States (Per Token)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.output_plot)
    print(f"Plot saved to {args.output_plot}")
    plt.show()


if __name__ == "__main__":
    main()
