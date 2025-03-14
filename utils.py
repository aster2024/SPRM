import torch
import torch.nn as nn
import random
import numpy as np
import warnings
import json

def segment_token_ids(full_token_id_list, boundaries):
    """
    Segment token id sequence based on boundaries.
    """
    segments = []
    for start, end in boundaries:
        segments.append(full_token_id_list[start:end])
    return segments


def extract_detailed_info_for_reasoning_path(prompt, reasoning_steps, separator, layers, tokenizer, model,
                                             apply_norm=False, to_cpu=True):
    """
    For a given prompt and list of reasoning steps, construct a dialogue, obtain model outputs,
    and determine token boundaries using robust conversation prefix processing.

    If apply_norm is True, apply the model's normalization function (retrieved using get_norm_and_head)
    to all hidden states before they are saved.
    """
    # Construct the full assistant answer by joining all reasoning steps.
    full_assistant_answer = separator.join(reasoning_steps)
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": full_assistant_answer}
    ]
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    full_token_ids = inputs["input_ids"][0].tolist()

    boundaries = []

    conversation_prefix = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""}
    ]
    conv = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False)
    conv = conv.strip()

    if conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
        conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
    if conv.endswith("<|eot_id|>"):
        conv = conv[:-len("<|eot_id|>")]
    prompt_ids = tokenizer.encode(conv, add_special_tokens=False)
    boundaries.append((0, len(prompt_ids)))
    last_length = len(prompt_ids)

    for i in range(len(reasoning_steps)):
        partial_answer = separator.join(reasoning_steps[: i + 1])
        conversation_prefix = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": partial_answer}
        ]
        conv = tokenizer.apply_chat_template(conversation_prefix, tokenize=False, add_generation_prompt=False)
        conv = conv.strip()
        # If this is NOT the final reasoning step, remove trailing special tokens.
        if (i + 1) != len(reasoning_steps) and conv.endswith("<|start_header_id|>assistant<|end_header_id|>"):
            conv = conv[:-len("<|start_header_id|>assistant<|end_header_id|>")]
        if (i + 1) != len(reasoning_steps) and conv.endswith("<|eot_id|>"):
            conv = conv[:-len("<|eot_id|>")]
        if (i + 1) != len(reasoning_steps):
            conv += separator
        current_ids = tokenizer.encode(conv, add_special_tokens=False)
        current_length = len(current_ids)
        boundaries.append((last_length, current_length))
        last_length = current_length

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states

    if apply_norm:
        norm_fn = get_norm(model)
        hidden_states = tuple(norm_fn(state) for state in hidden_states)

    if layers is None:
        layers_to_save = list(range(len(hidden_states)))
    else:
        layers_to_save = layers

    selected_hidden_states = {}
    if to_cpu:
        for layer in layers_to_save:
            if layer < len(hidden_states):
                selected_hidden_states[layer] = hidden_states[layer].detach().cpu()
            else:
                selected_hidden_states[layer] = None

        segments = segment_token_ids(full_token_ids, boundaries)

        detailed_info = {
            "conversations": conversation,
            "inputs": {k: v.detach().cpu().tolist() for k, v in inputs.items()},
            "hidden_states": selected_hidden_states,
            "boundaries": boundaries,
            "segments": segments
        }
    else:
        for layer in layers_to_save:
            if layer < len(hidden_states):
                selected_hidden_states[layer] = hidden_states[layer].detach()
            else:
                selected_hidden_states[layer] = None

        segments = segment_token_ids(full_token_ids, boundaries)

        detailed_info = {
            "conversations": conversation,
            "inputs": {k: v.detach().tolist() for k, v in inputs.items()},
            "hidden_states": selected_hidden_states,
            "boundaries": boundaries,
            "segments": segments
        }
    return detailed_info

def get_norm(model):
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

    return norm_fn


class LinearRewardModel(nn.Module):
    """
    A simple linear layer model for reward prediction using averaging.

    In training mode (is_eval=False), the model computes a single averaged reward over the entire token sequence.

    In evaluation mode (is_eval=True), the model segments the token sequence into reasoning steps (each defined as a segment),
    computes a gating weighted average reward for each step (if gating is enabled) or a simple average (if disabled),
    and then aggregates these step rewards using a specified reward_mode (e.g., "min", "mean", or "max").
    """

    def __init__(self, feature_dim, disable_gate=False):
        super(LinearRewardModel, self).__init__()
        self.disable_gate = disable_gate
        if not disable_gate:
            # Fused layer that predicts both a gate value and a reward.
            self.fused_layer = nn.Linear(feature_dim, 2)
        else:
            # Simple single-layer for reward prediction.
            self.reward_layer = nn.Linear(feature_dim, 1)

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
                # Sigmoid may cause vanishing gradients, so we use a softplus activation instead.
                # gates = torch.nn.functional.softplus(fused_output[..., 0])
                rewards = fused_output[..., 1]  # (batch_size, seq_len)
                sum_weighted_scores = torch.sum(gates * rewards * mask, dim=1)  # (batch_size)
                sum_gates = torch.sum(gates * mask, dim=1)  # (batch_size)
                avg_scores = sum_weighted_scores / sum_gates.clamp(min=1e-8)
            else:
                rewards = self.reward_layer(x).squeeze(-1)  # (batch_size, seq_len)
                masked_rewards = rewards * mask
                avg_scores = torch.sum(masked_rewards, dim=1) / torch.sum(mask, dim=1).clamp(min=1)
            return avg_scores
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
                    # segment_x = x[i, seg_start:seg_end, :]  # shape: (segment_length, feature_dim)
                    segment_x = x[i, :seg_end, :] # shape: (segment_length, feature_dim) # Use all steps before the current one
                    if not self.disable_gate:
                        fused_output = self.fused_layer(segment_x)  # (segment_length, 2)
                        gates = torch.sigmoid(fused_output[..., 0])  # (segment_length)
                        # Sigmoid may cause vanishing gradients, so we use a softplus activation instead.
                        # gates = torch.nn.functional.softplus(fused_output[..., 0])
                        rewards = fused_output[..., 1]  # (segment_length)
                        weighted_sum = torch.sum(gates * rewards)
                        sum_gates = torch.sum(gates)
                        seg_reward = weighted_sum / (sum_gates.clamp(min=1e-8))
                    else:
                        rewards = self.reward_layer(segment_x).squeeze(-1)  # (segment_length)
                        seg_reward = torch.mean(rewards)
                    step_rewards.append(seg_reward)

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
                    elif reward_mode == "all":
                        aggregated = step_rewards_tensor
                    else:
                        raise ValueError(f"Unsupported reward_mode: {reward_mode}")
                agg_rewards.append(aggregated)
            return torch.stack(agg_rewards)


class ICRewardModel(nn.Module):
    """
    Internal Consistency Reward Model.

    For each sample (i.e. a token sequence from the response and the last token of the prompt):
      - For each token:
          - For each chosen layer, compute the probability (via softmax)
            of the actual token (at this position) using the previous token's hidden state.
      - The vector (one scalar per layer) is fed through a light linear layer.
      - The token-level scores are averaged to yield a single reward score for the sample.

    The model can use either a logit lens (using the model’s own LM head) or a tuned lens
    (using a separate Unembed module) to transform hidden states into logits.
    """

    def __init__(self, lm_model, tokenizer, layers, use_tuned_lens=False):
        """
        lm_model: the underlying language model (e.g. GPT2)
        tokenizer: the corresponding tokenizer (needed if further processing is required)
        layers: a list of layer numbers to use (integers)
        use_tuned_lens: if True, use tuned lens (i.e. an external unembed module) rather than logit lens.
                       (Note: the flag use_tuned_lens is re-used here in place of a dedicated dim reduction flag.)
        """
        super(ICRewardModel, self).__init__()
        self.lm_model = lm_model
        self.tokenizer = tokenizer
        self.layers = layers
        self.use_tuned_lens = use_tuned_lens

        if not use_tuned_lens:
            self.norm_fn, self.head_fn = self.get_norm_and_head(lm_model)
        else:
            try:
                from tuned_lens.nn.unembed import Unembed
            except ImportError:
                raise ImportError("Please install tuned-lens (pip install tuned-lens) to use this option.")
            self.unembed = Unembed(lm_model)

        # A linear layer that maps a vector of per-layer probabilities (length=len(layers)) to a reward scalar.
        self.reward_linear = nn.Linear(len(layers), 1)

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

    def forward(self, batch):
        """
        Forward pass for a batch of samples.

        batch: a list of samples produced by InMemoryICRewardDataset.
          Each sample is a tuple: (hidden_states, response_input_ids, label)

        For each sample:
          - For each token position,
            compute a per-layer probability of the target token at position i.
          - Feed the vector of probabilities (one per chosen layer) through self.reward_linear.
          - Average the resulting token-level rewards to yield a final reward score.
        Returns a tensor of shape (batch_size,) with the predicted rewards.
        """
        device = self.reward_linear.weight.device
        batch_rewards = []
        for sample in batch:
            hidden_states_dict, response_ids, label = sample
            L = len(response_ids)

            token_reward_list = []
            # Loop over token positions from 1 to L-1.
            for i in range(1, L):
                per_layer_prob = []
                for layer in self.layers:
                    # If a layer is missing, use probability 0.
                    if layer not in hidden_states_dict or hidden_states_dict[layer] is None:
                        per_layer_prob.append(torch.tensor(0.0, device=device))
                        continue
                    # Get the hidden state for (i-1) from the current layer.
                    hidden = hidden_states_dict[layer].to(device)  # shape: (seq_length, hidden_dim)
                    prev_hidden = hidden[i - 1].unsqueeze(0)  # shape: (1, hidden_dim)
                    if not self.use_tuned_lens:
                        norm_hidden = self.norm_fn(prev_hidden)  # shape: (1, hidden_dim)
                        logits = self.head_fn(norm_hidden)  # shape: (1, vocab_size)
                    else:
                        logits = self.unembed(prev_hidden)  # shape: (1, vocab_size)
                    # Compute probability distribution (softmax) over vocabulary.
                    probs = torch.softmax(logits, dim=-1)[0]  # shape: (vocab_size,)
                    target_id = response_ids[i]
                    # In case the target id is out of range, default to 0.
                    if target_id >= probs.size(0):
                        token_prob = torch.tensor(0.0, device=device)
                    else:
                        token_prob = probs[target_id]
                    per_layer_prob.append(token_prob)
                # Stack the per-layer probabilities into a feature vector, shape: (num_layers,)
                token_feature = torch.stack(per_layer_prob)
                # Pass through the simple linear layer to get a scalar reward for this token.
                token_reward = self.reward_linear(token_feature.unsqueeze(0)).squeeze(0)
                token_reward_list.append(token_reward)
            # Aggregate the token-level rewards (here, by average).
            if len(token_reward_list) > 0:
                sample_reward = torch.stack(token_reward_list).mean()
            else:
                sample_reward = torch.tensor(0.0, device=device)
            batch_rewards.append(sample_reward)
        return torch.stack(batch_rewards)


class DimReduction(nn.Module):
    """
    A fixed linear projection for dimensionality reduction.
    This module ensures that both training and testing use a consistent projection.
    """

    def __init__(self, input_dim, output_dim):
        raise NotImplementedError("This class is not ready for use.")
        super(DimReduction, self).__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        raise NotImplementedError("This class is not ready for use.")
        # x: (batch_size, seq_length, input_dim)
        return self.proj(x)


class RewardModelWithDimReduction(nn.Module):
    """
    A wrapper that first applies a fixed dimension reduction,
    then applies the base reward model.
    """

    def __init__(self, base_model, dim_reduction):
        raise NotImplementedError("This class is not ready for use.")
        super(RewardModelWithDimReduction, self).__init__()
        self.dim_reduction = dim_reduction
        self.base_model = base_model

    def forward(self, x, lengths):
        raise NotImplementedError("This class is not ready for use.")
        # Reduce the feature dimension before feeding to the base model.
        x_reduced = self.dim_reduction(x)
        return self.base_model(x_reduced, lengths)


class CombinedRewardModel(nn.Module):
    """
    A simplified linear layer model for reward prediction with gating-based averaging
    and integration of an external overall reward signal.

    The model assumes inputs from an LLM's hidden states and computes a token-level gating
    and reward value. The overall model reward (r2) is computed as a gating-weighted average over tokens:
      r2 = sum(gate * reward * mask) / sum(gate * mask)
    where the mask indicates valid tokens based on the provided sequence lengths and gate is computed
    using a sigmoid activation.

    The final reward prediction is a weighted combination of the external overall reward (r1) and
    the model prediction (r2) modulated by the average gate value (v):
      final_reward = sigmoid(alpha) * r1 + (1 - sigmoid(alpha)) * v * r2
    where alpha is a learnable parameter (after sigmoid, stays in the (0,1) range).
    """

    def __init__(self, feature_dim):
        """
        Args:
            feature_dim (int): Dimensionality of the input features (hidden state size).
        """
        super(CombinedRewardModel, self).__init__()
        self.fused_layer = nn.Linear(feature_dim, 2)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, lengths, ext_reward):
        """
        Args:
            x (Tensor): Input features of shape (batch_size, max_seq_length, feature_dim).
            lengths (list or Tensor): Sequence lengths for each sample in the batch.
            ext_reward (Tensor): External overall reward of shape (batch_size,).

        Returns:
            Tensor: Final aggregated reward for each sample in the batch, of shape (batch_size,).
        """
        ext_reward.requires_grad = False  # Ensure that the external reward is not used for gradient computation.
        batch_size, max_seq_len, _ = x.size()
        device = x.device

        mask = torch.zeros((batch_size, max_seq_len), dtype=torch.float32, device=device)
        for i, length in enumerate(lengths):
            mask[i, :length] = 1.0

        fused_output = self.fused_layer(x)  # shape: (batch_size, seq_len, 2)
        gates = torch.sigmoid(fused_output[..., 0])  # shape: (batch_size, seq_len)
        rewards_pred = fused_output[..., 1]  # shape: (batch_size, seq_len)

        sum_gates = torch.sum(gates * mask, dim=1)  # shape: (batch_size,)
        count_tokens = torch.sum(mask, dim=1).clamp(min=1e-8)
        v = sum_gates / count_tokens  # shape: (batch_size,)

        weighted_reward = torch.sum(gates * rewards_pred * mask, dim=1)
        r2 = weighted_reward / sum_gates.clamp(min=1e-8)  # shape: (batch_size,)

        alpha_weight = torch.sigmoid(self.alpha)
        final_reward = alpha_weight * ext_reward + (1 - alpha_weight) * v * r2

        return final_reward


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def extract_candidate_features(summary_sample, detailed_sample, reasoning_idx):
    """
    Extract token features, candidate label, and sequence length for a given candidate
    (reasoning path) from a sample.

    Returns:
        token_features (Tensor): Concatenated hidden state features (seq_length, feature_dim)
        label: correctness label for the candidate.
        candidate_length (int): Sequence length of the candidate.
    """
    reasoning_paths = detailed_sample.get("detailed_paths", [])
    labels = summary_sample.get("correctness", [])
    if reasoning_idx >= len(reasoning_paths) or reasoning_idx >= len(labels):
        warnings.warn(f"Index mismatch for sample, reasoning index {reasoning_idx}")
        return None
    path = reasoning_paths[reasoning_idx]
    label = labels[reasoning_idx]

    hidden_states = path.get("hidden_states", {})
    if not hidden_states:
        warnings.warn(f"No hidden states found for reasoning index {reasoning_idx}. Skipping.")
        return None

    boundaries = path.get("boundaries", None)
    if boundaries is not None and len(boundaries) >= 2:
        # Use the reward segment from the start of the first reasoning step to the end of the last step.
        reward_start = boundaries[1][0]
        reward_end = boundaries[-1][1]
    else:
        reward_start = 0
        first_layer_key = sorted(hidden_states.keys(), key=lambda x: int(x))[0]
        token_seq_length = len(hidden_states[first_layer_key])
        reward_end = token_seq_length

    # Concatenate hidden states from selected layers.
    sorted_layers = sorted(hidden_states.keys(), key=lambda x: int(x))
    layer_tensors = []
    for layer in sorted_layers:
        tensor = hidden_states[layer][0].clone().detach().to(torch.float32)  # shape: (seq_length, hidden_dim)
        tensor = tensor[reward_start:reward_end]
        layer_tensors.append(tensor)

    seq_lengths = [t.size(0) for t in layer_tensors]
    if len(set(seq_lengths)) != 1:
        warnings.warn(f"Sequence lengths do not match for reasoning index {reasoning_idx}. Skipping.")
        return None

    token_features = torch.cat(layer_tensors, dim=-1)
    candidate_length = token_features.size(0)
    return token_features, label, candidate_length

def load_data(file_name):
    """
    Load the dataset from a JSON file that contains a list of candidate outputs.
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
    Extract token features for the candidate portion from detailed_info.
    Uses the boundaries returned in detailed_info to select the tokens
    corresponding to the assistant’s (i.e. reasoning) part.
    """
    hidden_states = detailed_info["hidden_states"]
    if not hidden_states:
        return None
    boundaries = detailed_info.get("boundaries", None)
    if boundaries is not None and len(boundaries) >= 2:
        # Use tokens from the start of the first reasoning step to the end of the last step.
        reward_start = boundaries[1][0]
        reward_end = boundaries[-1][1]
    else:
        warnings.warn("No boundaries found in detailed_info, using all tokens.")
        first_layer_key = sorted(hidden_states.keys(), key=lambda x: int(x))[0]
        reward_end = len(hidden_states[first_layer_key][0])
        reward_start = 0
    # Extract and concatenate hidden states from all saved layers.
    sorted_layers = sorted(hidden_states.keys(), key=lambda x: int(x))
    layer_tensors = []
    for layer in sorted_layers:
        tensor = hidden_states[layer][0].clone().detach().to(torch.float32)
        tensor = tensor[reward_start:reward_end]
        layer_tensors.append(tensor)
    if len(layer_tensors) == 0:
        warnings.warn("No valid token features extracted, skipping.")
        return None
    token_features = torch.cat(layer_tensors, dim=-1)
    return token_features


def compute_metrics(groups, k_vals, ext_reward_mode="none"):
    """
    Calculate pass@k accuracy for each k value.

    Depending on the chosen external reward integration mode, the candidate selection is modified.
    Modes:
      - "none": Use only the reward model's reward (original behavior).
      - "ranking": For each group, compute separate rankings for the reward and the external reward,
                   then compute a combined ranking (the worse, i.e., larger, of the two).
                   Then select the candidate with the lowest combined rank among the top k.
      - "scaling": For each group, scale the reward and external reward to [0,1] and average them.
                   Then select the candidate with the highest combined score among the top k.
    """
    metrics = {}
    for k in k_vals:
        correct_count = 0
        total = len(groups)
        for group in groups:
            if ext_reward_mode == "none":
                topk = group[:k]
                if topk:
                    best_candidate = max(topk, key=lambda x: x["reward"])
                    if best_candidate["correctness"]:
                        correct_count += 1
            elif ext_reward_mode == "ranking":
                sorted_by_reward = sorted(group, key=lambda x: x["reward"], reverse=True)
                reward_ranks = {id(c): idx for idx, c in enumerate(sorted_by_reward)}
                # Compute ranking from the external reward (higher is better)
                sorted_by_ext = sorted(group, key=lambda x: x["ext_reward"], reverse=True)
                ext_ranks = {id(c): idx for idx, c in enumerate(sorted_by_ext)}
                # Compute combined ranking: the worse (larger index) of the two rankings for each candidate
                for candidate in group:
                    candidate["combined_rank"] = max(reward_ranks[id(candidate)], ext_ranks[id(candidate)])
                # Sort candidates by combined ranking in ascending order (lower rank is better)
                sorted_candidates = sorted(group, key=lambda x: x["combined_rank"])
                topk = sorted_candidates[:k]
                if topk:
                    best_candidate = topk[0]
                    if best_candidate["correctness"]:
                        correct_count += 1
            elif ext_reward_mode == "scaling":
                # Combined scaling integration method
                rewards = [c["reward"] for c in group]
                ext_rewards = [c["ext_reward"] for c in group]
                r_min, r_max = min(rewards), max(rewards)
                er_min, er_max = min(ext_rewards), max(ext_rewards)
                for candidate in group:
                    # Scale the reward model's reward to [0,1]
                    if r_max - r_min != 0:
                        norm_reward = (candidate["reward"] - r_min) / (r_max - r_min)
                    else:
                        norm_reward = 1.0
                    # Scale the external reward to [0,1]
                    if er_max - er_min != 0:
                        norm_ext = (candidate["ext_reward"] - er_min) / (er_max - er_min)
                    else:
                        norm_ext = 1.0
                    candidate["integrated_score"] = (norm_reward + norm_ext) / 2.0
                # Sort candidates by integrated score in descending order (higher is better)
                sorted_candidates = sorted(group, key=lambda x: x["integrated_score"], reverse=True)
                topk = sorted_candidates[:k]
                if topk:
                    best_candidate = topk[0]
                    if best_candidate["correctness"]:
                        correct_count += 1
            else:
                raise ValueError(f"Invalid external reward integration mode: {ext_reward_mode}")
        metrics[k] = round((correct_count / total) * 100, 1) if total > 0 else 0.0
    return metrics


def extract_avg_response_feature(prompt, reasoning_steps, separator, layers, tokenizer, model, apply_norm=False):
    """
    Extract an averaged hidden state feature vector for the response segment.

    For a given prompt and candidate reasoning steps, this function:
      1. Constructs a conversation by joining the reasoning steps with the provided separator.
      2. Tokenizes the conversation (using the custom chat template of the tokenizer).
      3. Obtains the hidden states from the model.
      4. Concatenates the hidden states from the specified layers.
      5. Determines the token boundaries so that only the assistant's response is considered.
      6. Averages the concatenated hidden states over the token dimension for the response part.

    Args:
        prompt (str): The input prompt.
        reasoning_steps (list): List of reasoning steps (strings) representing the candidate response.
        separator (str): String used to join the reasoning steps.
        layers (list or None): List of layer indices to use. If None, use all layers.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        model (transformers.PreTrainedModel): The language model.
        apply_norm (bool): If True, apply layer normalization on the concatenated hidden states.

    Returns:
        np.ndarray: Averaged hidden state feature vector (1D numpy array) for the response.
    """
    full_answer = separator.join(reasoning_steps)
    conversation = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": full_answer}
    ]

    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    full_token_ids = inputs["input_ids"][0].tolist()

    model_output = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = model_output.hidden_states

    if layers is None:
        layers_to_use = list(range(len(hidden_states)))
    else:
        layers_to_use = layers

    layer_features = []
    for layer in layers_to_use:
        if layer < len(hidden_states) and hidden_states[layer] is not None:
            layer_features.append(hidden_states[layer].squeeze(0))  # shape: (seq_len, hidden_dim)

    concatenated_features = torch.cat(layer_features, dim=-1)

    if apply_norm:
        concatenated_features = torch.nn.functional.layer_norm(
            concatenated_features, concatenated_features.shape[-1:]
        )

    conversation_prefix = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": ""}
    ]

    conv_prefix = tokenizer.apply_chat_template(
        conversation_prefix,
        tokenize=False,
        add_generation_prompt=False
    ).strip()

    prefix_ids = tokenizer.encode(conv_prefix, add_special_tokens=False)
    prefix_length = len(prefix_ids)

    response_features = concatenated_features[prefix_length:]

    if response_features.size(0) == 0:
        warnings.warn("No response tokens detected. Returning a zero vector.")
        avg_feature = torch.zeros(concatenated_features.size(1))
    else:
        avg_feature = torch.mean(response_features, dim=0)

    return avg_feature.cpu().detach().float().numpy()

