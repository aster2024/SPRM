#!/usr/bin/env python
import argparse
import warnings
import os
import sys
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import *


class ORMRewardDataset(Dataset):
    """
    Dataset for overall reward (ORM) mode. Each example corresponds to a single candidate rollout.
    The candidate is extracted on-the-fly using the summary and detailed data.
    We assume that the summary sample contains a list of correctness labels (one per candidate).
    In ORM mode, we use the candidate’s correctness (converted to float) as the external reward.
    """

    def __init__(self, summary_data, detailed_data):
        self.summary_data = summary_data
        self.detailed_data = detailed_data
        self.index_pairs = []
        for i, sum_sample in enumerate(summary_data):
            correctness = sum_sample.get("correctness", [])
            for j in range(len(correctness)):
                self.index_pairs.append((i, j))

    def __len__(self):
        return len(self.index_pairs)

    def __getitem__(self, idx):
        i, j = self.index_pairs[idx]
        summary_sample = self.summary_data[i]
        detailed_sample = self.detailed_data[i]
        candidate = extract_candidate_features(summary_sample, detailed_sample, j)
        if candidate is None:
            return None
        features, label, length = candidate
        ext_reward = summary_sample.get("score", [])[j]
        return (features, length, ext_reward, float(label))


class PRMRewardDataset(Dataset):
    """
    Dataset for per-step reward (PRM) mode. Here each candidate rollout is segmented into
    individual steps based on boundaries computed in detailed data.

    We assume:
      - The summary sample's "steps" field is a list of rollouts (each rollout is a list of steps).
      - The summary sample's "step_scores" field is a list of lists; each inner list holds a reward
        score for each step in that rollout.
      - The detailed sample's "detailed_paths" contains, for each candidate rollout, a "boundaries" key.
        The boundaries list is produced by extract_detailed_info_for_reasoning_path and its first element
        corresponds to the prompt; subsequent elements mark the boundaries of each reasoning step.

    The label (target) is taken from the overall candidate correctness (same for all steps).
    """

    def __init__(self, summary_data, detailed_data):
        self.summary_data = summary_data
        self.detailed_data = detailed_data
        # Build index triples: (sample index, candidate index, step index)
        self.index_triples = []
        for i, sum_sample in enumerate(summary_data):
            rollouts = sum_sample.get("steps", [])
            step_scores_all = sum_sample.get("step_scores", [])
            assert len(rollouts) == len(step_scores_all)
            num_candidates = len(step_scores_all)
            for j in range(num_candidates):
                step_scores = step_scores_all[j]
                assert len(rollouts[j]) == len(step_scores)
                for k in range(len(step_scores)):
                    self.index_triples.append((i, j, k))

    def __len__(self):
        return len(self.index_triples)

    def __getitem__(self, idx):
        i, j, k = self.index_triples[idx]
        summary_sample = self.summary_data[i]
        detailed_sample = self.detailed_data[i]
        candidate = extract_candidate_features(summary_sample, detailed_sample, j)
        if candidate is None:
            return None
        features, label, _ = candidate
        try:
            detailed_candidate = detailed_sample.get("detailed_paths", [])[j]
        except IndexError:
            warnings.warn(f"Candidate index {j} not found in detailed_paths for sample {i}.")
            return None
        boundaries_all = detailed_candidate.get("boundaries", None)
        if boundaries_all is None or len(boundaries_all) < 2:
            warnings.warn(f"Missing or insufficient boundaries for sample {i}, candidate {j}.")
            return None
        # In extract_candidate_features, features were computed as hidden_states[boundaries_all[1][0]:boundaries_all[-1][1]]
        # Therefore, relative index 0 corresponds to boundaries_all[1][0].
        reward_start = boundaries_all[1][0]
        # Ensure that step index k is within the steps available.
        if k >= (len(boundaries_all) - 1):
            warnings.warn(f"Step index {k} out of range for sample {i}, candidate {j}.")
            return None
        step_boundary = boundaries_all[k + 1]
        # Convert absolute boundaries (with respect to full token sequence) to relative indices.
        relative_start = step_boundary[0] - reward_start
        relative_end = step_boundary[1] - reward_start
        if relative_end <= relative_start:
            warnings.warn(f"Invalid boundary for sample {i}, candidate {j}, step {k}.")
            return None
        step_features = features[relative_start: relative_end]
        step_length = relative_end - relative_start
        # External reward for this step is taken from the summary sample's step_scores.
        try:
            ext_reward = float(summary_sample.get("step_scores", [])[j][k])
        except (IndexError, TypeError):
            warnings.warn(f"Missing step score for sample {i}, candidate {j}, step {k}.")
            return None
        return (step_features, step_length, ext_reward, float(summary_sample.get("correctness", [])[j]))


def collate_fn_reward(batch):
    """
    Collate function for both ORM and PRM datasets.
    Each sample is a tuple: (features, length, ext_reward, label).
    Features are padded along the token (sequence) dimension.
    """
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        return None
    features_list, lengths, ext_rewards, labels = zip(*batch)
    padded_features = pad_sequence(features_list, batch_first=True, padding_value=0)
    ext_rewards_tensor = torch.tensor(ext_rewards, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    return padded_features, lengths, ext_rewards_tensor, labels_tensor


def train_reward_model(reward_model, optimizer, summary_data, detailed_data, args, device):
    """
    Train the reward model using either ORM or PRM mode.
    The dataset returns:
      - features: token features (padded sequences)
      - lengths: list of sequence lengths
      - ext_reward: external reward (r1)
      - label: ground truth label (0 or 1)
    The reward model outputs a final reward used along with a chosen loss (CE or hinge).
    """
    if args.train_mode == "orm":
        dataset = ORMRewardDataset(summary_data, detailed_data)
    elif args.train_mode == "prm":
        dataset = PRMRewardDataset(summary_data, detailed_data)
    else:
        raise ValueError(f"Unknown train_mode: {args.train_mode}")

    collate_function = collate_fn_reward

    if args.train_val_split_ratio is not None and 0 < args.train_val_split_ratio < 1:
        total = len(dataset)
        train_size = int(total * (1 - args.train_val_split_ratio))
        val_size = total - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.reward_batch_size, shuffle=True,
                                      collate_fn=collate_function)
        val_dataloader = DataLoader(val_dataset, batch_size=args.reward_batch_size, shuffle=False,
                                    collate_fn=collate_function)
    else:
        train_dataloader = DataLoader(dataset, batch_size=args.reward_batch_size, shuffle=True,
                                      collate_fn=collate_function)
        val_dataloader = None

    # Choose criterion based on method.
    if args.method == "hinge":
        def hinge_loss(outputs, labels):
            """
            Hinge loss.
            Transforms labels from {0,1} to {-1, +1} and computes:
              loss = mean( clamp(1 - transformed_labels * outputs, min=0) )
            """
            labels_transformed = labels * 2 - 1
            return torch.mean(torch.clamp(1 - labels_transformed * outputs, min=0))

        criterion = hinge_loss
    elif args.method == "ce":
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported method: {args.method}. Only 'ce' and 'hinge' are supported.")

    reward_model.train()
    patience = 3
    best_val_loss = float('inf')
    patience_counter = 0
    best_reward_model = None

    for epoch in range(args.reward_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.reward_epochs}", leave=False)
        for batch in progress_bar:
            if batch is None:
                continue
            features, lengths, ext_rewards, labels = batch
            features = features.to(device)
            ext_rewards = ext_rewards.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = reward_model(features, lengths, ext_rewards)  # shape: (batch_size,)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_train_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        print(f"Epoch {epoch + 1}/{args.reward_epochs}, Train Loss: {avg_train_loss:.4f}")

        if val_dataloader is not None:
            reward_model.eval()
            val_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    if batch is None:
                        continue
                    features, lengths, ext_rewards, labels = batch
                    features = features.to(device)
                    ext_rewards = ext_rewards.to(device)
                    labels = labels.to(device)
                    outputs = reward_model(features, lengths, ext_rewards)
                    loss = criterion(outputs, labels)
                    batch_size = features.size(0)
                    total_samples += batch_size
                    val_loss += loss.item() * batch_size
            avg_val_loss = val_loss / (total_samples if total_samples > 0 else 1)
            print(f"Epoch {epoch + 1}/{args.reward_epochs}, Val Loss: {avg_val_loss:.4f}")
            reward_model.train()
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_reward_model = reward_model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}.")
                    break
    return best_reward_model if best_reward_model is not None else reward_model


def process_data_and_train(args):
    ds = load_data(args.input_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model_lm.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_samples = []
    detailed_samples = []
    feature_dim = None

    total_samples = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
    sample_count = 0
    with tqdm(total=total_samples, desc="Extracting samples") as pbar:
        for i, sample in enumerate(ds):
            prompt = sample.get("prompt", "")
            raw_paths = sample.get("steps", [])
            if not isinstance(raw_paths, list):
                print(f"Skipping sample {i}: invalid steps format.")
                pbar.update(1)
                continue
            detailed_info_paths = []
            for reasoning_steps in raw_paths:
                if not isinstance(reasoning_steps, list):
                    raise ValueError("Reasoning steps format error.")
                info = extract_detailed_info_for_reasoning_path(
                    prompt, reasoning_steps, args.separator, args.layers, tokenizer, model_lm, args.apply_norm
                )
                detailed_info_paths.append(info)

            if args.train_mode == "orm":
                assert "scores" in sample, "ORM mode requires 'scores' field in the sample."
            elif args.train_mode == "prm":
                assert "step_scores" in sample, "PRM mode requires 'step_scores' field in the sample."
            else:
                raise ValueError(f"Unknown train_mode: {args.train_mode}")

            output_sample = {
                "prompt": prompt,
                "reference": sample.get("reference", ""),
                "dataset": sample.get("dataset", ""),
                "completions": sample.get("completions", []),
                "correctness": sample.get("correctness", []),
                "steps": raw_paths,
                "scores": sample.get("scores", []),
                "step_scores": sample.get("step_scores", [])
            }
            summary_samples.append(output_sample)

            detailed_sample = {
                "prompt": prompt,
                "steps": raw_paths,
                "detailed_paths": detailed_info_paths
            }
            detailed_samples.append(detailed_sample)

            if feature_dim is None and len(detailed_info_paths) > 0:
                hs = detailed_info_paths[0]["hidden_states"]
                sorted_layers = sorted(hs.keys(), key=lambda x: int(x))
                feature_dim = sum(hs[layer].shape[-1] for layer in sorted_layers if hs[layer] is not None)
                print(f"Detected feature dimension: {feature_dim}")

            sample_count += 1
            pbar.update(1)
            if args.max_samples and sample_count >= args.max_samples:
                break

    del model_lm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    if feature_dim is None:
        raise ValueError("No valid samples with detailed hidden states found.")

    print(f"\nTraining mode: {args.train_mode}")
    print(f"Training reward model on {len(summary_samples)} samples...")

    if args.output_model_file is None:
        norm_part = "norm_" if args.apply_norm else ""
        output_file = f"model/combined_reward_model_{args.train_mode}_{args.method}_{norm_part}{args.model_name.replace('/', '_')}.pt"
    else:
        output_file = args.output_model_file

    reward_model = CombinedRewardModel(feature_dim).to(device)
    reward_optimizer = optim.AdamW(reward_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    reward_model = train_reward_model(reward_model, reward_optimizer, summary_samples, detailed_samples, args, device)
    torch.save(reward_model.state_dict(), output_file)
    print(f"Training completed. Reward model saved to {output_file}.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a CombinedRewardModel (using ORM or PRM mode) for reward prediction."
    )
    # Extraction-related arguments.
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained model name or path (e.g., 'gpt2').")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input JSON file containing the samples.")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to save hidden states; if not provided, save all layers.")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator between prompt and reasoning steps (default: two newlines).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process.")

    # Reward model training arguments.
    parser.add_argument("--reward_epochs", type=int, default=40,
                        help="Number of training epochs (default: 40).")
    parser.add_argument("--reward_batch_size", type=int, default=64,
                        help="Batch size for training (default: 64).")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4).")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay (default: 1e-5).")
    parser.add_argument("--apply_norm", action="store_true",
                        help="Apply layer normalization to hidden states before reward model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--output_model_file", type=str, default=None,
                        help="Path to save the trained reward model.")

    # New arguments.
    parser.add_argument("--train_val_split_ratio", type=float, default=0.2,
                        help="Fraction of data for validation (default: 0.2).")
    parser.add_argument("--method", type=str, required=True,
                        choices=["ce", "hinge"],
                        help="Loss method to use ('ce' for cross-entropy or 'hinge').")
    parser.add_argument("--train_mode", type=str, required=True,
                        choices=["orm", "prm"],
                        help="Training mode: 'orm' for overall reward, 'prm' for per-step reward.")


    args = parser.parse_args()

    if args.output_model_file is not None:
        output_dir = os.path.dirname(args.output_model_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    set_seed(args.seed)
    process_data_and_train(args)


if __name__ == "__main__":
    main()
