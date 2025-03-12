#!/usr/bin/env python

import argparse
import warnings
import os
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import set_seed, extract_detailed_info_for_reasoning_path


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


def precompute_ic_features(info, lm_model, tokenizer, layers, use_tuned_lens=False):
    """
    Precompute the internal consistency (IC) features for a given sample's detailed info.
    Instead of storing full hidden states, we use a lens conversion (either 'logit lens' or 'tuned lens')
    to transform hidden states into logits, compute softmax probabilities, and then extract the probability
    of the target token at each token position. We also include one extra token preceding the response
    (so the first response token's prediction is computed from the prompt's last token).

    Returns:
      ic_features: A tensor of shape (L, num_layers), where L = extended sequence length - 1.
                   Each row corresponds to the probability (from one lens method) for each selected layer.
    """
    boundaries = info.get("boundaries", None)
    hidden_states = info.get("hidden_states", {})  # dict of layer -> tensor of shape (1, seq_length, hidden_dim)
    inputs = info.get("inputs", {})
    full_token_ids = inputs.get("input_ids", None)
    if full_token_ids is None:
        warnings.warn("No input ids found in detailed info.")
        return None

    if boundaries is not None and len(boundaries) >= 2:
        reward_start = boundaries[1][0]
        reward_end = boundaries[-1][1]
    else:
        first_layer_key = sorted(hidden_states.keys(), key=lambda x: int(x))[0]
        token_seq_length = len(hidden_states[first_layer_key][0])
        reward_start = 0
        reward_end = token_seq_length

    # Extend the response segment by including the token immediately preceding reward_start (if available)
    new_reward_start = reward_start - 1 if reward_start > 0 else reward_start

    # Extended token ids include the additional preceding token.
    extended_token_ids = full_token_ids[0][new_reward_start:reward_end]
    L_extended = len(extended_token_ids)
    if L_extended < 2:
        # Not enough tokens to compute prediction (need at least 1 pair)
        warnings.warn("Not enough tokens to compute IC features.")
        return None

    if use_tuned_lens:
        try:
            from tuned_lens.nn.unembed import Unembed
        except ImportError:
            raise ImportError("Please install tuned-lens (pip install tuned-lens) to use this option.")
        unembed = Unembed(lm_model)

        def compute_layer_logits(hidden):
            return unembed(hidden)
    else:
        norm_fn, head_fn = get_norm_and_head(lm_model)

        def compute_layer_logits(hidden):
            hidden_norm = norm_fn(hidden)
            return head_fn(hidden_norm)

    ic_features_matrix_list = []
    for layer in layers:
        hs_tensor = hidden_states.get(str(layer), None)
        if hs_tensor is None:
            ic_features_matrix_list.append(torch.zeros(L_extended - 1))
            continue
        hs_tensor = hs_tensor[0].to(torch.float32)  # shape: (seq_length, hidden_dim)
        extended_hs = hs_tensor[new_reward_start:reward_end]  # shape: (L_extended, hidden_dim)
        # For each token prediction, use the hidden state at previous position
        preceding_states = extended_hs[:-1]  # shape: (L_extended-1, hidden_dim)
        with torch.no_grad():
            logits = compute_layer_logits(preceding_states)  # shape: (L_extended-1, vocab_size)
            probs = torch.softmax(logits, dim=-1)  # shape: (L_extended-1, vocab_size)
        target_ids = torch.tensor(extended_token_ids[1:], dtype=torch.long)  # shape: (L_extended-1,)
        target_ids = target_ids.unsqueeze(1)  # shape: (L_extended-1, 1)
        token_probs = torch.gather(probs, 1, target_ids).squeeze(1)  # shape: (L_extended-1,)
        ic_features_matrix_list.append(token_probs)
    # Stack per-layer probabilities to get a feature matrix of shape (L_extended-1, num_layers)
    ic_features = torch.stack(ic_features_matrix_list, dim=-1)
    return ic_features


class InMemoryICRewardDataset(Dataset):
    """
    A dataset class that builds examples from in-memory summary and detailed samples.
    Instead of storing full hidden states, IC features (precomputed using logit lens/tuned lens)
    are stored. Each sample returns a tensor of shape (T, num_layers) and a correctness label.
    """

    def __init__(self, summary_data, detailed_data):
        self.summary_data = summary_data
        self.detailed_data = detailed_data
        self.example_idx = []
        for i, summary_sample in enumerate(summary_data):
            labels = summary_sample.get("correctness", [])
            for j in range(len(labels)):
                self.example_idx.append((i, j))

    def __len__(self):
        return len(self.example_idx)

    def __getitem__(self, idx):
        summary_idx, reasoning_idx = self.example_idx[idx]
        summary_sample = self.summary_data[summary_idx]
        detail_sample = self.detailed_data[summary_idx]

        reasoning_paths = detail_sample.get("detailed_paths", [])
        labels = summary_sample.get("correctness", [])
        if reasoning_idx >= len(reasoning_paths) or reasoning_idx >= len(labels):
            warnings.warn(f"Index mismatch for sample {summary_idx}, reasoning index {reasoning_idx}")
            return None

        path = reasoning_paths[reasoning_idx]
        label = labels[reasoning_idx]

        ic_features = path.get("ic_features", None)
        if ic_features is None:
            warnings.warn(
                f"No precomputed IC features found for sample {summary_idx}, reasoning index {reasoning_idx}. Skipping.")
            return None

        return ic_features, label


def collate_ic_fn(batch):
    """
    Collate function for IC dataset.
    Pads variable-length IC feature sequences.
    Each sample: (ic_features [T x num_layers], label)
    Returns: padded_features [B x max_T x num_layers], lengths, and labels tensor.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    ic_features_list, labels_list = zip(*batch)
    lengths = [feat.shape[0] for feat in ic_features_list]
    padded_features = pad_sequence(ic_features_list, batch_first=True, padding_value=0)
    labels = torch.tensor(labels_list, dtype=torch.float32)
    return padded_features, lengths, labels


class ICRewardModel(nn.Module):
    """
    A lightweight reward model that takes precomputed IC feature matrices as input.
    For each token (i.e. each row in the matrix), the model produces a scalar reward.

    In training mode (is_eval=False), the model computes a single averaged reward over the entire token sequence.

    In evaluation mode (is_eval=True), the model segments the token sequence into reasoning steps (each defined as a segment),
    computes a gating weighted average reward for each step (if gating is enabled) or a simple average (if disabled),
    and then aggregates these step rewards using a specified reward_mode (e.g., "min", "mean", or "max").
    """

    def __init__(self, feature_dim, type, disable_gate):
        super(ICRewardModel, self).__init__()
        self.disable_gate = disable_gate
        out_features = 2 if not disable_gate else 1
        if type in ["linear", "svm"]:
            self.fused_layer = nn.Linear(feature_dim, out_features)
        elif type == "mlp":
            self.fused_layer = nn.Sequential(
                nn.Linear(feature_dim, 4096),
                nn.ReLU(),
                nn.Linear(4096, out_features)
            )
        else:
            raise ValueError(f"Invalid model type: {type}")

    def forward(self, x, lengths, is_eval=False, boundaries=None, reward_mode="min"):
        """
        x: Tensor of shape (batch_size, max_seq_length, feature_dim).
        lengths: List of actual sequence lengths per example.
        is_eval: Boolean indicating whether to use evaluation mode with token segmentation.
        boundaries: (Optional) For evaluation mode only. A list (one per sample) of segment boundaries.
                    Each element is a list of (start, end) tuples, representing indices within x for each reasoning step.
        reward_mode: Aggregation mode for evaluation mode; options are "min", "mean", or "max".

        Returns:
            If is_eval=False: Averaged reward score per sample (one value per candidate).
            If is_eval=True: An aggregated reward computed from per-step rewards (aggregated by reward_mode).
        """
        batch_size, max_seq_len, _ = x.size()
        device = x.device

        if not is_eval:
            # Training (or standard) mode: compute reward over the full sequence.
            mask = torch.zeros((batch_size, max_seq_seq_len := max_seq_len), dtype=torch.float32, device=device)
            for i, length in enumerate(lengths):
                mask[i, :length] = 1.0

            if not self.disable_gate:
                fused_output = self.fused_layer(x)  # (batch_size, seq_len, 2)
                gates = torch.sigmoid(fused_output[..., 0])  # (batch_size, seq_len)
                rewards = fused_output[..., 1]  # (batch_size, seq_len)
                weighted_scores = gates * rewards * mask
                sum_weighted_scores = torch.sum(weighted_scores, dim=1)
                # sum_gates = torch.sum(gates * mask, dim=1)
                # avg_scores = sum_weighted_scores / sum_gates.clamp(min=1e-8)
            else:
                rewards = self.reward_layer(x).squeeze(-1)  # (batch_size, seq_len)
                masked_rewards = rewards * mask
                sum_weighted_scores = torch.sum(masked_rewards, dim=1)
                # avg_scores = torch.sum(masked_rewards, dim=1) / torch.sum(mask, dim=1).clamp(min=1)
            # return avg_scores
            return sum_weighted_scores
        else:
            # Evaluation mode: segmentation-based token reward computation.
            if boundaries is None:
                raise ValueError("Boundaries must be provided for evaluation mode.")

            agg_rewards = []
            for i in range(batch_size):
                sample_boundaries = boundaries[i]  # list of (start, end) for each reasoning step in this candidate
                step_rewards = []
                for (seg_start, seg_end) in sample_boundaries:
                    if seg_end <= seg_start:
                        warnings.warn(f"Skipping empty segment: {seg_start} -> {seg_end}")
                        continue
                    segment_x = x[i, seg_start:seg_end, :]  # shape: (segment_length, feature_dim)
                    if not self.disable_gate:
                        fused_output = self.fused_layer(segment_x)  # (segment_length, 2)
                        gates = torch.sigmoid(fused_output[..., 0])  # (segment_length)
                        rewards = fused_output[..., 1]  # (segment_length)
                        weighted_sum = torch.sum(gates * rewards)
                        # sum_gates = torch.sum(gates)
                        # seg_reward = weighted_sum / (sum_gates.clamp(min=1e-8))
                    else:
                        rewards = self.reward_layer(segment_x).squeeze(-1)  # (segment_length)
                        weighted_sum = torch.sum(rewards)
                        # seg_reward = torch.mean(rewards)
                    # step_rewards.append(seg_reward)
                    step_rewards.append(weighted_sum)

                if len(step_rewards) == 0:
                    aggregated = torch.tensor(0.0, device=device)
                else:
                    step_rewards_tensor = torch.stack(step_rewards)  # shape: (num_segments,)
                    if reward_mode == "min":
                        aggregated = torch.min(step_rewards_tensor)
                    elif reward_mode == "mean":
                        aggregated = torch.mean(step_rewards_tensor)
                    elif reward_mode == "max":
                        aggregated = torch.max(step_rewards_tensor)
                    else:
                        raise ValueError(f"Unsupported reward_mode: {reward_mode}")
                agg_rewards.append(aggregated)
            return torch.stack(agg_rewards)


def train_ic_reward_model(reward_model, optimizer, summary_data, detailed_data, args, device):
    """
    Train the IC reward model using the InMemoryICRewardDataset.
    Uses BCEWithLogitsLoss or hinge loss (if SVM mode is selected).
    Early stopping is applied if the validation loss does not improve.
    """
    dataset = InMemoryICRewardDataset(summary_data, detailed_data)

    if args.train_val_split_ratio is not None and 0 < args.train_val_split_ratio < 1:
        total = len(dataset)
        train_size = int(total * (1 - args.train_val_split_ratio))
        val_size = total - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.reward_batch_size, shuffle=True,
                                      collate_fn=collate_ic_fn)
        val_dataloader = DataLoader(val_dataset, batch_size=args.reward_batch_size, shuffle=False,
                                    collate_fn=collate_ic_fn)
    else:
        train_dataloader = DataLoader(dataset, batch_size=args.reward_batch_size, shuffle=True,
                                      collate_fn=collate_ic_fn)
        val_dataloader = None

    # Choose loss function based on model_type.
    if args.model_type == "svm":
        def hinge_loss(outputs, labels):
            # Transform labels from {0,1} to {-1, +1}.
            labels_transformed = labels * 2 - 1
            loss = torch.mean(torch.clamp(1 - labels_transformed * outputs, min=0))
            return loss

        criterion = hinge_loss
    else:
        criterion = nn.BCEWithLogitsLoss()

    reward_model.train()
    patience = 3  # epochs to wait for validation improvement
    best_val_loss = float('inf')
    patience_counter = 0
    best_reward_model = None

    for epoch in range(args.reward_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"IC Reward Model Training Epoch {epoch + 1}/{args.reward_epochs}",
                            leave=False)
        for batch in progress_bar:
            if batch is None:
                continue
            features, lengths, labels = batch
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = reward_model(features, lengths)  # shape: (batch_size,)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        avg_train_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
        print(f"Epoch {epoch + 1}/{args.reward_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation step.
        if val_dataloader is not None:
            reward_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    if batch is None:
                        continue
                    features, lengths, labels = batch
                    features = features.to(device)
                    labels = labels.to(device)
                    outputs = reward_model(features, lengths)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * features.size(0)
            avg_val_loss = val_loss / len(val_dataset) if len(val_dataset) > 0 else 0.0
            print(f"Epoch {epoch + 1}/{args.reward_epochs}, Val Loss: {avg_val_loss:.4f}")
            reward_model.train()
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_reward_model = reward_model
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1} due to no improvement in validation loss.")
                    break

    return best_reward_model if best_reward_model is not None else reward_model


def process_data_and_train(args):
    ds = load_dataset("Windy0822/ultrainteract_math_rollout", split="train")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_lm = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model_lm.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary_samples = []  # For reward training (labels, etc.)
    detailed_samples = []  # For detailed extraction info (will include precomputed IC features)

    reward_model = None
    reward_optimizer = None

    sample_count = 0
    total_samples = len(ds) if args.max_samples is None else min(args.max_samples, len(ds))
    layer_list = args.layers

    with tqdm(total=total_samples, desc="Extracting samples") as pbar:
        for i, sample in enumerate(ds):
            prompt = sample.get("prompt", "")
            raw_paths = sample.get("steps", [])
            if not isinstance(raw_paths, list):
                print(f"Skipping sample {i}: steps field format error.")
                pbar.update(1)
                continue

            detailed_info_paths = []
            for reasoning_steps in raw_paths:
                if not isinstance(reasoning_steps, list):
                    raise ValueError("Reasoning steps format is not a list.")
                info = extract_detailed_info_for_reasoning_path(
                    prompt, reasoning_steps, args.separator, args.layers, tokenizer, model_lm
                )
                if layer_list is None:
                    hs = info["hidden_states"]
                    sorted_layers = sorted(hs.keys(), key=lambda x: int(x))
                    layer_list = [int(x) for x in sorted_layers]
                ic_features = precompute_ic_features(info, model_lm, tokenizer, layer_list,
                                                     use_tuned_lens=args.use_tuned_lens)
                if ic_features is None:
                    raise ValueError("IC features extraction failed.")
                info["ic_features"] = ic_features
                # delete the hidden states to save memory
                del info["hidden_states"]

                detailed_info_paths.append(info)

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

            sample_count += 1
            pbar.update(1)

            if reward_model is None and len(detailed_info_paths) > 0:
                feature_dim = len(layer_list)
                print(f"Initializing IC Reward model with feature dimension: {feature_dim}")
                base_model = ICRewardModel(feature_dim, args.model_type, args.disable_gate).to(device)
                reward_model = base_model
                reward_optimizer = optim.AdamW(reward_model.parameters(),
                                               lr=args.learning_rate, weight_decay=args.weight_decay)

            if args.max_samples and sample_count >= args.max_samples:
                break

    # Free the LM and tokenizer resources.
    model_lm = None
    tokenizer = None
    torch.cuda.empty_cache()

    if len(summary_samples) > 0 and len(detailed_samples) > 0:
        print(f"\nFinal training on remaining {len(summary_samples)} samples...")
        reward_model = train_ic_reward_model(reward_model, reward_optimizer, summary_samples, detailed_samples, args,
                                             device)

    # Clear in-memory data and save the trained reward model.
    summary_samples = []
    detailed_samples = []
    torch.save(reward_model.state_dict(), args.output_model_file)
    print(f"Training completed. Reward model saved to {args.output_model_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract detailed data and train an Internal Consistency (IC) reward model based on Windy0822/ultrainteract_math_rollout."
    )
    # Extraction arguments.
    parser.add_argument("--model_name", type=str, required=True,
                        help="Pre-trained model name or path, e.g., 'gpt2'")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layer indices to use for IC; if not provided, all available layers are used")
    parser.add_argument("--separator", type=str, default="\n\n",
                        help="Separator used to join prompt and each reasoning step (default: two newlines)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (default: process all samples)")

    # Reward model training arguments.
    parser.add_argument("--reward_epochs", type=int, default=60,
                        help="Number of training epochs for the reward model (default: 60)")
    parser.add_argument("--reward_batch_size", type=int, default=64,
                        help="Batch size for reward model training (default: 64)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for reward model training (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer (default: 1e-5)")
    parser.add_argument("--disable_gate", action="store_true",
                        help="Disable gating mechanism in the reward model")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output_model_file", type=str, default=None,
                        help="Path to save the trained reward model")

    # New arguments for training validation split, model type and lens conversion.
    parser.add_argument("--train_val_split_ratio", type=float, default=0.2,
                        help="Fraction of data for validation during reward model training (default: 0.2)")
    parser.add_argument("--model_type", type=str, choices=["linear", "svm", "mlp"], default="linear",
                        help="Reward model type: 'linear' for standard, 'svm' for hinge loss based (default: linear), 'mlp' for multi-layer perceptron")
    parser.add_argument("--use_tuned_lens", action="store_true",
                        help="If set, use tuned lens (via tuned-lens) instead of logit lens for hidden state conversion")

    args = parser.parse_args()

    if args.output_model_file is None:
        args.output_model_file = f"model/reward_model_ic_{args.model_name.replace('/', '_')}_type{args.model_type}_{'tuned' if args.use_tuned_lens else 'logit'}_lens.pt"

    set_seed(args.seed)
    process_data_and_train(args)


if __name__ == "__main__":
    main()
