import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryVQVAE(nn.Module):
    def __init__(self, input_dim, code_dim=64, num_codes=512, beta=0.25, mode='vq'):
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.num_codes = num_codes
        self.beta = beta
        if mode not in {'vq', 'continuous', 'frozen'}:
            raise ValueError(f'Unknown quantizer mode: {mode}')
        self.mode = mode

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, code_dim * 2),
            nn.ReLU(),
            nn.Linear(code_dim * 2, code_dim),
        )
        self.codebook = nn.Embedding(num_codes, code_dim) if mode != 'continuous' else None
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, code_dim * 2),
            nn.ReLU(),
            nn.Linear(code_dim * 2, input_dim),
        )

        if self.codebook is not None:
            nn.init.uniform_(self.codebook.weight, -1.0 / num_codes, 1.0 / num_codes)
            if mode == 'frozen':
                self.codebook.weight.requires_grad_(False)

    def _flatten(self, x):
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        return x_flat, orig_shape

    def _quantize(self, z_e):
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=1)
            - 2 * z_e @ self.codebook.weight.t()
        )
        encoding_indices = distances.argmin(dim=1)
        z_q = self.codebook(encoding_indices)

        commitment_loss = F.mse_loss(z_e, z_q.detach())
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        vq_loss = codebook_loss + self.beta * commitment_loss

        z_q_st = z_e + (z_q - z_e).detach()

        encodings = F.one_hot(encoding_indices, self.num_codes).float()
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())

        return z_q_st, vq_loss, perplexity, encoding_indices

    def forward(self, x):
        x_flat, orig_shape = self._flatten(x)
        z_e = self.encoder(x_flat)
        if self.mode == 'continuous':
            recon = self.decoder(z_e).reshape(orig_shape)
            indices = torch.full(orig_shape[:-1], -1, device=x.device, dtype=torch.long)
            return recon, z_e.new_zeros(()), z_e.new_zeros(()), indices
        z_q_st, vq_loss, perplexity, encoding_indices = self._quantize(z_e)
        recon = self.decoder(z_q_st).reshape(orig_shape)
        return recon, vq_loss, perplexity, encoding_indices.reshape(orig_shape[:-1])

