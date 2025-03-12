#!/usr/bin/env python
import argparse
import os
import warnings
import gc
import sys

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


class GPUPCAWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper for cuML-based PCA implementation to accelerate PCA computation on GPU,
    and convert the result to a numpy array for subsequent concatenation with sklearn's LDA.
    """
    def __init__(self, n_components=100):
        self.n_components = n_components
        try:
            from cuml.decomposition import PCA as cumlPCA
        except ImportError as e:
            raise ImportError("cuML is not installed. Please install RAPIDS cuML to use GPU acceleration for PCA.") from e
        self._pca = cumlPCA(n_components=self.n_components)

    def fit(self, X, y=None):
        try:
            import cupy as cp
        except ImportError as e:
            raise ImportError("cupy is not installed. Please install cupy to use GPU acceleration for PCA.") from e
        X_cp = cp.asarray(X)
        self._pca.fit(X_cp)
        return self

    def transform(self, X):
        try:
            import cupy as cp
        except ImportError as e:
            raise ImportError("cupy is not installed. Please install cupy to use GPU acceleration for PCA.") from e
        X_cp = cp.asarray(X)
        X_transformed = self._pca.transform(X_cp)
        return cp.asnumpy(X_transformed)


def build_feature_dataset(args, tokenizer, model):
    """
    Iterate over the dataset, extract the average hidden state corresponding to the candidate answer as a feature,
    and save it together with the corresponding correctness label.

    Args:
        args: Command-line configuration parameters.
        tokenizer: Pre-trained tokenizer.
        model: Pre-trained language model.

    Returns:
        X (np.ndarray): Feature matrix, shape is (num_samples, feature_dim).
        y (np.ndarray): Corresponding label array.
    """
    X, y = [], []
    ds = load_dataset("Windy0822/ultrainteract_math_rollout", split="train")

    total_samples = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
    processed_samples = 0

    print(f"Processing up to {total_samples} samples...")

    for i, sample in tqdm(enumerate(ds), total=total_samples, desc="Processing samples"):
        prompt = sample.get("prompt", "")
        raw_paths = sample.get("steps", [])
        correctness = sample.get("correctness", [])

        if not isinstance(raw_paths, list) or not isinstance(correctness, list):
            warnings.warn(f"Sample {i} is malformed, skipping.")
            continue

        num_candidates = min(len(raw_paths), len(correctness))
        for j in range(num_candidates):
            feature = extract_avg_response_feature(
                prompt,
                raw_paths[j],
                args.separator,
                args.layers,
                tokenizer,
                model,
                args.apply_norm
            )

            X.append(feature)
            y.append(correctness[j])

        processed_samples += 1
        if processed_samples >= total_samples:
            break

    X = np.array(X)
    y = np.array(y)

    print(f"Extracted features for {X.shape[0]} candidate responses. Feature shape: {X.shape[1:]}")
    return X, y


def main():
    parser = argparse.ArgumentParser(
        description="Extract average Hidden State features from candidate answers and train a PCA+LDA pipeline, "
                    "supporting GPU acceleration of PCA via --use_gpu."
    )
    # Model and feature extraction parameters.
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained language model name or path (e.g., 'gpt2').")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="List of hidden state layer indices to use; if None, use all layers.")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator used to join the reasoning steps (default: two newlines).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process from the dataset (default: all samples).")
    parser.add_argument("--apply_norm", action="store_true",
                        help="Normalize the concatenated hidden state.")

    # Pipeline parameters.
    parser.add_argument("--pca_components", type=int, default=50,
                        help="Number of components to keep during PCA dimensionality reduction (default: 50).")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Enable GPU acceleration for PCA calculation (based on cuML).")

    # Output parameters.
    parser.add_argument("--output_model_file", type=str, default=None,
                        help="Path to save the trained PCA+LDA pipeline (e.g., 'model/pipeline_model.pkl').")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")

    args = parser.parse_args()

    set_seed(args.seed)

    if not args.output_model_file:
        norm_part = "norm_" if args.apply_norm else ""
        args.output_model_file = f"model/pca_lda_pipeline_{norm_part}{args.model_name.replace('/', '_')}.pkl"

    output_dir = os.path.dirname(args.output_model_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model.eval()

    X, y = build_feature_dataset(args, tokenizer, model)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.use_gpu:
        print("Using GPU-accelerated PCA (cuML).")
        pca_component = GPUPCAWrapper(n_components=args.pca_components)
    else:
        from sklearn.decomposition import PCA
        print("Using CPU-based PCA (scikit-learn).")
        pca_component = PCA(n_components=args.pca_components)

    pipeline = Pipeline([
        ('pca', pca_component),
        ('lda', LinearDiscriminantAnalysis())
    ])

    print("Training the PCA+LDA pipeline...")
    pipeline.fit(X, y)
    print("Training completed.")

    output_dir = os.path.dirname(args.output_model_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    joblib.dump(pipeline, args.output_model_file)
    print(f"Trained pipeline saved to {args.output_model_file}")


if __name__ == "__main__":
    main()
