import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    """
    Mixture of Experts Feed-Forward Network.
    Each expert is an independent FFN (Linear -> ReLU -> Linear).
    Uses a learned gating network with Top-K selection.
    """
    def __init__(self, d_model, dim_feedforward, num_experts=4, top_k=2, dropout=0.1,
                 routing_mode='learned'):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model
        if routing_mode not in {'learned', 'random'}:
            raise ValueError(f'Unknown routing mode: {routing_mode}')
        if not 1 <= top_k <= num_experts:
            raise ValueError('top_k must be between 1 and num_experts')
        self.routing_mode = routing_mode
        self.routing_generator = None
        self.collect_routing_stats = False
        self.register_buffer('routing_counts', torch.zeros(num_experts, dtype=torch.long),
                             persistent=False)

        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Expert FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        Args:
            x: [*, d_model]  (any leading dims, e.g. [B, T, D] or [B, D])
        Returns:
            output: same shape as x
            aux_loss: scalar, load balancing loss
        """
        orig_shape = x.shape
        # Flatten to 2D: [N, D]
        x_flat = x.reshape(-1, self.d_model)
        N = x_flat.shape[0]

        if self.routing_mode == 'random':
            # iid continuous scores give a uniform subset without replacement.
            scores = torch.rand(N, self.num_experts, device=x.device,
                                generator=self.routing_generator)
            top_k_indices = scores.topk(self.top_k, dim=-1).indices
            top_k_probs = x_flat.new_full((N, self.top_k), 1.0 / self.top_k)
            gate_probs = x_flat.new_full((N, self.num_experts), 1.0 / self.num_experts)
        else:
            gate_logits = self.gate(x_flat)
            gate_probs = F.softmax(gate_logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
            top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        if self.collect_routing_stats:
            self.routing_counts.add_(torch.bincount(top_k_indices.reshape(-1),
                                                   minlength=self.num_experts))

        # Compute expert outputs (only for selected experts)
        output = torch.zeros_like(x_flat)  # [N, D]
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, k]    # [N]
            weight = top_k_probs[:, k:k+1]      # [N, 1]
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += weight[mask] * expert_output

        # Load balancing auxiliary loss (Switch Transformer style)
        # f_i: fraction of tokens routed to expert i
        # P_i: average gating probability for expert i
        f = torch.zeros(self.num_experts, device=x.device)
        for k in range(self.top_k):
            one_hot = F.one_hot(top_k_indices[:, k], self.num_experts).float()  # [N, E]
            f += one_hot.mean(dim=0)
        f = f / self.top_k  # average across top-k slots

        P = gate_probs.mean(dim=0)  # [num_experts]
        aux_loss = (f * P).sum() * self.num_experts

        output = output.reshape(orig_shape)
        return output, aux_loss


class MoETransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with MoE-FFN.
    Shares the self-attention across all experts, but uses MoE for the FFN part.
    Interface-compatible with nn.TransformerEncoderLayer.
    """
    def __init__(self, d_model, nhead, dim_feedforward, num_experts=4, top_k=2,
                 dropout=0.1, norm_first=False, batch_first=True, routing_mode='learned'):
        super().__init__()
        self.norm_first = norm_first

        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)

        # MoE FFN
        self.moe_ffn = MoEFFN(d_model, dim_feedforward, num_experts, top_k, dropout,
                             routing_mode=routing_mode)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        """
        Args:
            src: [B, T, D] (batch_first=True)
            src_mask: attention mask
            src_key_padding_mask: key padding mask
        Returns:
            output: [B, T, D]
            aux_loss: scalar
        """
        if self.norm_first:
            # Pre-norm
            src2 = self.norm1(src)
            src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src = src + self.dropout(src2)

            src2 = self.norm2(src)
            ffn_out, aux_loss = self.moe_ffn(src2)
            src = src + ffn_out
        else:
            # Post-norm
            src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
            src = self.norm1(src + self.dropout(src2))

            ffn_out, aux_loss = self.moe_ffn(src)
            src = self.norm2(src + ffn_out)

        return src, aux_loss


class MoETransformerEncoder(nn.Module):
    """
    Stack of MoETransformerEncoderLayer.
    Collects aux_loss from all layers.
    """
    def __init__(self, d_model, nhead, dim_feedforward, num_layers,
                 num_experts=4, top_k=2, dropout=0.1, norm_first=False,
                 routing_mode='learned'):
        super().__init__()
        self.layers = nn.ModuleList([
            MoETransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                num_experts=num_experts, top_k=top_k, dropout=dropout,
                norm_first=norm_first, batch_first=True, routing_mode=routing_mode
            )
            for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Returns:
            output: [B, T, D]
            total_aux_loss: scalar (sum over all layers)
        """
        total_aux_loss = 0.0
        output = src
        for layer in self.layers:
            output, aux_loss = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            total_aux_loss = total_aux_loss + aux_loss
        return output, total_aux_loss
