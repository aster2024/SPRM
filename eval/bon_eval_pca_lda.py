#!/usr/bin/env python
import argparse
import json
import warnings
import os
import sys
import time
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
import joblib

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *

from sklearn.base import BaseEstimator, TransformerMixin

class GPUPCAWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for cuML PCA to perform PCA on GPU. This wrapper ensures that the
    output of the PCA transform is a numpy array (as required by the scikit-learn
    pipeline downstream, e.g., for LDA).
    """
    def __init__(self, n_components=100):
        self.n_components = n_components
        try:
            from cuml.decomposition import PCA as cumlPCA
        except ImportError as e:
            raise ImportError("cuML is required for GPU accelerated PCA. Please install RAPIDS cuML.") from e
        self._pca = cumlPCA(n_components=self.n_components)

    def fit(self, X, y=None):
        try:
            import cupy as cp
        except ImportError as e:
            raise ImportError("Cupy is required for GPU accelerated PCA. Please install cupy.") from e
        X_cp = cp.asarray(X)
        self._pca.fit(X_cp)
        return self

    def transform(self, X):
        try:
            import cupy as cp
        except ImportError as e:
            raise ImportError("Cupy is required for GPU accelerated PCA. Please install cupy.") from e
        X_cp = cp.asarray(X)
        X_transformed = self._pca.transform(X_cp)
        return cp.asnumpy(X_transformed)

    # def fit_transform(self, X, y=None):
    #     self.fit(X, y)
    #     return self.transform(X)


def evaluate_all_models(args, reward_model_files):
    """
    Evaluate candidate outputs using multiple reward pipelines.

    For each candidate output:
      - In "orm" mode, compute a single averaged hidden state from the full response
        and pass it through the loaded pipeline.
      - In "prm" mode (one of "min", "mean", "max", "product"), process each reasoning
        step—computing the average hidden state for the partial response—and then aggregate
        the per-step pipeline scores according to the chosen aggregation method.

    The function then computes pass@k metrics and writes them to JSON files.
    """
    print("\n========== Starting Evaluation ==========")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    ds = load_data(args.dataset_file)
    print(f"Loaded {len(ds)} candidate outputs from {args.dataset_file}.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    model_lm.eval()

    sample0 = ds[0]
    if len(sample0.get("steps", [])) == 0:
        print("No reasoning steps in the first sample, exiting evaluation.")
        return None

    feature0 = extract_avg_response_feature(
        sample0["prompt"],
        sample0["steps"],
        args.separator,
        args.layers,
        tokenizer,
        model_lm,
        apply_norm=args.apply_norm,
    )
    feature_dim = feature0.shape[0]
    print(f"Pipeline feature dimension: {feature_dim}")

    reward_models = {}
    for method, pipeline_file in reward_model_files.items():
        pipeline = joblib.load(pipeline_file)
        if args.use_gpu:
            if hasattr(pipeline, "named_steps") and "pca" in pipeline.named_steps:
                # If the PCA step is not already a GPUPCAWrapper, attempt conversion.
                if not isinstance(pipeline.named_steps["pca"], GPUPCAWrapper):
                    try:
                        cpu_pca = pipeline.named_steps["pca"]
                        gpu_pca = GPUPCAWrapper(n_components=cpu_pca.n_components)
                        # Transfer fitted parameters from the CPU PCA to GPU PCA.
                        # Note: This transfer assumes compatibility between the parameters.
                        gpu_pca._pca.components_ = cpu_pca.components_
                        if hasattr(cpu_pca, "mean_"):
                            gpu_pca._pca.mean_ = cpu_pca.mean_
                        steps = pipeline.steps
                        for i, (name, transformer) in enumerate(steps):
                            if name == "pca":
                                steps[i] = (name, gpu_pca)
                                break
                        pipeline.steps = steps
                        print(f"Converted PCA step for method [{method}] to GPU accelerated version.")
                    except Exception as e:
                        warnings.warn(f"Failed to convert PCA to GPU for method [{method}]: {e}")
            else:
                warnings.warn("Pipeline does not have a 'pca' step; cannot apply GPU acceleration.")
        reward_models[method] = pipeline
        print(f"Loaded pipeline for method [{method}] from {pipeline_file}")

    groups_dict = defaultdict(list)
    for sample in ds:
        groups_dict[sample["idx"]].append(sample)
    if args.max_samples:
        groups_dict = dict(list(groups_dict.items())[:args.max_samples])

    results = {method: [] for method in reward_models.keys()}
    total_time = 0.0

    for idx, candidate_list in tqdm(groups_dict.items(), desc="Processing candidate groups"):
        group_results = {method: [] for method in reward_models.keys()}
        for candidate in candidate_list:
            prompt = candidate["prompt"]
            reference = candidate.get("reference", "")
            detailed_info = extract_detailed_info_for_reasoning_path(
                prompt,
                candidate.get("steps", []),
                args.separator,
                args.layers,
                tokenizer,
                model_lm,
                apply_norm=args.apply_norm,
                to_cpu=True
            )
            token_features = get_token_features(detailed_info)
            if token_features is None:
                continue
            seq_len = token_features.size(0)
            boundaries = detailed_info.get("boundaries", None)
            if boundaries is not None and len(boundaries) >= 2:
                step_boundaries = boundaries[1:]
            else:
                warnings.warn("No valid boundaries found, using the whole sequence.")
                step_boundaries = [(0, seq_len)]
            candidate_text = args.separator.join(candidate.get("steps", []))
            correct = bool(candidate["correctness"])

            for method, pipeline in reward_models.items():
                if args.eval_mode == "orm":
                    overall_avg = torch.mean(token_features, dim=0).cpu().numpy()
                    t0 = time.time()
                    score = pipeline.decision_function(overall_avg.reshape(1, -1))[0]
                    t1 = time.time()
                    total_time += (t1 - t0)
                elif args.eval_mode in ["min", "mean", "max", "product"]:
                    if step_boundaries is not None and len(step_boundaries) > 0:
                        step_scores = []
                        for b in step_boundaries:
                            avg_feature = torch.mean(token_features[0:b[1]], dim=0).cpu().numpy()
                            t0 = time.time()
                            score_i = pipeline.decision_function(avg_feature.reshape(1, -1))[0]
                            t1 = time.time()
                            total_time += (t1 - t0)
                            step_scores.append(score_i)
                        if args.eval_mode == "min":
                            score = float(min(step_scores))
                        elif args.eval_mode == "mean":
                            score = float(np.mean(step_scores))
                        elif args.eval_mode == "max":
                            score = float(max(step_scores))
                        elif args.eval_mode == "product":
                            prod = 1.0
                            for s in step_scores:
                                prod *= s
                            score = prod
                    else:
                        warnings.warn("No valid boundaries found; using overall average.")
                        overall_avg = torch.mean(token_features, dim=0).cpu().numpy()
                        t0 = time.time()
                        score = pipeline.decision_function(overall_avg.reshape(1, -1))[0]
                        t1 = time.time()
                        total_time += (t1 - t0)
                else:
                    raise ValueError(f"Invalid evaluation mode: {args.eval_mode}")

                candidate_result = {
                    "prompt": prompt,
                    "reference": reference,
                    "extracted_output": candidate.get("extracted_output", candidate_text),
                    "correctness": int(correct),
                    "reward": score
                }
                group_results[method].append(candidate_result)

        for method in reward_models.keys():
            if group_results[method]:
                results[method].append(group_results[method])

    print(f"Total time taken for reward computation: {total_time:.4f} seconds")

    # Compute and print pass@k metrics for each reward model.
    all_metrics = {}
    for method, groups in results.items():
        if len(groups) == 0:
            print(f"No valid candidate results for method {method}")
            continue
        metrics = compute_metrics(groups, args.k_vals)
        print(f"\n======== Pass@k Evaluation Results for pipeline [{method}] ========")
        for k, val in metrics.items():
            print(f"Pass@{k}: {val}%")
        print("===========================================")
        all_metrics[method] = metrics
        if args.metrics_file is None:
            norm_part = "norm_" if args.apply_norm else ""
            metrics_file = f"res/eval_metrics_pca_lda_{method}_{args.eval_mode}_{norm_part}{args.model_name.replace('/', '_')}.json"
        else:
            base, ext = os.path.splitext(args.metrics_file)
            metrics_file = f"{base}_{method}{ext}"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Evaluation metrics saved to {metrics_file}")

    if len(all_metrics) > 1:
        print("\n===== Overall Evaluation Metrics =====")
        for method, m in all_metrics.items():
            print(f"Method [{method}]:")
            for k, v in m.items():
                print(f"   Pass@{k}: {v}%")
        print("=======================================")
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate candidate reasoning paths using multiple trained reward pipelines."
    )
    # Model and dataset arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained LM model name or path (e.g., 'gpt2').")
    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the dataset JSON file containing candidate outputs.")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator used to join reasoning steps (default: two newlines).")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Hidden layer indices to extract; if empty, extract all layers.")
    # Reward model arguments
    parser.add_argument("--reward_model_load", type=str, default=None,
                        help="Path to a saved reward model checkpoint. Optional if --methods is provided.")
    parser.add_argument("--disable_gate", action="store_true",
                        help="Disable gating mechanism. (Not used in the pipeline version)")
    parser.add_argument("--apply_norm", action="store_true",
                        help="Apply normalization to hidden states before reward computation.")
    parser.add_argument("--use_dim_reduction", action="store_true",
                        help="(Not used in the pipeline version)")
    parser.add_argument("--dim_reduction_dim", type=int, default=128,
                        help="(Not used in the pipeline version)")
    # Pass@k evaluation arguments
    parser.add_argument("--k_vals", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64],
                        help="List of candidate numbers for pass@k evaluation (default: 1 2 4 8 16 32 64).")
    parser.add_argument("--eval_mode", type=str, default="min", choices=["min", "mean", "max", "orm", "product"],
                        help="Evaluation mode: aggregation mode for step rewards ('min', 'mean', 'max', 'product') or 'orm' for full response mode (default: min).")
    # New parameter: enable GPU acceleration for PCA inference.
    parser.add_argument("--use_gpu", action="store_true",
                        help="Enable GPU acceleration for PCA inference using cuML (requires cuML and cupy).")
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--metrics_file", type=str, default=None,
                        help="If set, save evaluation metrics to this JSON file. For multiple evaluations, the method name will be appended.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (default: None).")
    args = parser.parse_args()

    reward_model_files = {}

    if not args.reward_model_load:
        norm_part = "norm_" if args.apply_norm else ""
        args.reward_model_load = f"model/pca_lda_pipeline_{norm_part}{args.model_name.replace('/', '_')}.pkl"

    reward_model_files["single"] = args.reward_model_load

    evaluate_all_models(args, reward_model_files)


if __name__ == "__main__":
    main()
