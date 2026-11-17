import numpy as np
import torch
import torch.nn as nn


from models.utils import polyline_encoder
from models.context_encoder.mtr_encoder import SinusoidalPosEmb
from models.mds import MoETransformerEncoder
from einops import rearrange
import math


class SocialTransformer(nn.Module):
    def __init__(self, in_dim=48, hidden_dim=256, out_dim=128):
        super(SocialTransformer, self).__init__()
        self.encode_past = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2, dim_feedforward=hidden_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)
        self.mlp_out = nn.Linear(hidden_dim, out_dim)


    def forward(self, past_traj, mask):
        """
        @param past_traj: [B, A, P, D]
        @param mask:      [B, A] or None
        """
        B, A, P, D = past_traj.shape
        past_traj = rearrange(past_traj, 'b a p d -> b a (p d)')

        h_feat = self.encode_past(past_traj)    # [B, A, D]

        h_feat_ = self.transformer_encoder(h_feat, mask=mask)

        h_feat = h_feat + h_feat_
        h_feat = self.mlp_out(h_feat)           # [B, A, D]

        return h_feat


class ETHEncoder(nn.Module):
    def __init__(self, config, use_pre_norm):
        super().__init__()
        self.model_cfg = config
        dim = self.model_cfg.D_MODEL

        ### build social encoder
        # in_dim = max_past_frames * 6 (abs + rel + vel, each 2D)
        social_in_dim = self.model_cfg.get('SOCIAL_IN_DIM', 48)
        self.agent_social_encoder = SocialTransformer(in_dim=social_in_dim, hidden_dim=256, out_dim=dim)

        # Positional encoding
        self.pos_encoding = nn.Sequential(
                SinusoidalPosEmb(dim, theta = 10000),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
        self.agent_query_embedding = nn.Embedding(self.model_cfg.AGENTS, dim)
        self.mlp_pe = nn.Sequential(
            nn.Linear(2*dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        # Check if MoE is enabled for encoder
        moe_cfg = self.model_cfg.get('MOE', None)
        use_moe = moe_cfg is not None and moe_cfg.get('ENCODER_MOE', False)

        if use_moe:
            num_experts = moe_cfg.get('NUM_EXPERTS', 4)
            top_k = moe_cfg.get('TOP_K', 2)
            self.transformer_encoder = MoETransformerEncoder(
                d_model=dim,
                nhead=self.model_cfg.NUM_ATTN_HEAD,
                dim_feedforward=self.model_cfg.get('FFN_DIM', dim * 4),
                num_layers=self.model_cfg.NUM_ATTN_LAYERS,
                num_experts=num_experts,
                top_k=top_k,
                dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                norm_first=use_pre_norm,
                routing_mode=moe_cfg.get('ROUTING_MODE', 'learned'),
            )
            self.use_moe = True
        else:
            # build standard transformer encoder layers
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim,
                                                    dropout=self.model_cfg.get('DROPOUT_OF_ATTN', 0.1),
                                                    nhead=self.model_cfg.NUM_ATTN_HEAD,
                                                    dim_feedforward=self.model_cfg.get('FFN_DIM', dim * 4),
                                                    norm_first=use_pre_norm,
                                                    batch_first=True)
            # Preserve legacy checkpoint names by default; ablation controls can
            # omit the unused source template from their parameter accounting.
            if self.model_cfg.get('REGISTER_ENCODER_TEMPLATE', True):
                self.layer = encoder_layer
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.model_cfg.NUM_ATTN_LAYERS)
            self.use_moe = False

        self.num_out_channels = dim

    ### polyline encoder MLP PointNet [B, A, D]
    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder


    def forward(self, past_traj):
        """
        Args: [Batch size, Number of agents, Number of time frames, 6]
        Returns:
            encoder_out: [B, A, D]
            aux_loss: scalar (0 if not using MoE)
        """

        B, A, P, D = past_traj.shape
        agent_feature = self.agent_social_encoder(past_traj, mask=None)  # [B, A, D]

        ### use positional encoding
        pos_encoding = self.pos_encoding(torch.arange(agent_feature.shape[1]).to(past_traj.device))         # [A, D]

        ### enforce positional encoding earlier here
        agent_query = self.agent_query_embedding(torch.arange(self.model_cfg.AGENTS).to(past_traj.device))  # [A, D]

        pos_encoding = self.mlp_pe(torch.cat([agent_query, pos_encoding], dim=-1)) # [A, D]

        agent_feature += pos_encoding.unsqueeze(0)                              # [B, A, D]

        if self.use_moe:
            encoder_out, aux_loss = self.transformer_encoder(agent_feature)
        else:
            encoder_out = self.transformer_encoder(agent_feature)
            aux_loss = 0.0

        return encoder_out, aux_loss
