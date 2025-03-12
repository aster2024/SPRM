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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import *

def get_token_features_per_layer(detailed_info):
    """
    Extract token features from the detailed_info, separately for each layer.
    Returns a dictionary where keys are layer indices and values are token features.
    """
    hidden_states = detailed_info["hidden_states"]
    if not hidden_states:
        warnings.warn("No hidden states found.")
        return None

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

    layer_features = {}
    sorted_layers = sorted(hidden_states.keys(), key=lambda x: int(x))
    for layer in sorted_layers:
        tensor = hidden_states[layer][0].clone().detach().to(torch.float32)
        tensor = tensor[reward_start:reward_end]
        layer_features[int(layer)] = tensor.mean(dim=0).cpu().numpy()

    return layer_features

def main():
    parser = argparse.ArgumentParser(
        description="Extract hidden state representations, perform PCA+LDA, and cross-validate."
    )
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the dataset JSON file.")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained LM model name or path.")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator for reasoning steps.")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Indices of hidden layers to extract. If None, use all available layers.")
    parser.add_argument("--apply_norm", action="store_true",
                        help="Apply normalization to hidden states.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of candidate samples to process.")
    parser.add_argument("--pca_components", type=int, default=50,
                        help="Number of PCA components to keep.")
    parser.add_argument("--output_file", type=str, default="res/lda_pca_results.txt",
                        help="Filename to save the cross-validation results.")
    parser.add_argument("--output_plot", type=str, default="figs/lda_pca_mean_cv_accuracy.png",
                        help="Filename to save the output plot.")

    print("Running PCA+LDA pipeline for hidden states.")

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

    all_candidate_features = defaultdict(list)
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

        layer_features = get_token_features_per_layer(detailed_info)
        if layer_features is None:
            continue

        for layer, features in layer_features.items():
            all_candidate_features[layer].append(features)

        correctness = int(candidate.get("correctness", 0))
        all_labels.append(correctness)

        num_processed += 1
        if args.max_samples is not None and num_processed >= args.max_samples:
            break

    if num_processed == 0:
        print("No valid samples processed. Exiting.")
        return

    all_labels = np.array(all_labels)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    with open(args.output_file, "w") as f:
        for layer in sorted(all_candidate_features.keys()):
            features_np = np.stack(all_candidate_features[layer])

            pipeline = Pipeline([
                ('pca', PCA(n_components=args.pca_components)),
                ('lda', LinearDiscriminantAnalysis())
            ])

            scores = cross_val_score(pipeline, features_np, all_labels, cv=cv, scoring='accuracy')

            mean_cv_accuracy = scores.mean()

            print(f"Layer {layer}:")
            print(f"  Cross-validation scores: {scores}")
            print(f"  Mean cross-validation accuracy: {mean_cv_accuracy:.4f}")
            results[layer] = {
                'cv_scores': scores.tolist(),
                'mean_cv_accuracy': mean_cv_accuracy
            }
            f.write(f"Layer {layer}:\n")
            f.write(f"  Cross-validation scores: {scores}\n")
            f.write(f"  Mean cross-validation accuracy: {mean_cv_accuracy:.4f}\n")

    print(f"Results saved to {args.output_file}")

    results_df = pd.DataFrame.from_dict(results, orient='index')

    plt.figure(figsize=(10, 6))
    plt.bar(results_df.index.astype(str), results_df['mean_cv_accuracy'], color='skyblue')
    plt.xlabel("Layer")
    plt.ylabel("Mean Cross-Validation Accuracy")
    plt.title("Mean Cross-Validation Accuracy per Layer (PCA + LDA)")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(args.output_plot)
    plt.show()

    print(f"Plot saved to {args.output_plot}")

if __name__ == "__main__":
    main()

