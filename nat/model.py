"""Transformer built from neuron-astrocyte attention blocks."""
import torch
import torch.nn.functional as F
from torch import nn

from .attention import AttentionNAT


class LayerNAT(nn.Module):
    """Pre-norm block: neuron-astrocyte attention + ReLU MLP, with residuals."""

    def __init__(self, d_model, m_hidden, gamma=0.95):
        super().__init__()
        self.attention = AttentionNAT(d_model, m_hidden, gamma=gamma)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, X):
        Z = self.attention(self.ln1(X)) + X
        return self.mlp(self.ln2(Z)) + Z


class NAT(nn.Module):
    """Neuron-Astrocyte Transformer for sequence classification.

    Token + learned positional embeddings, n_layers LayerNAT blocks, and one
    linear output head per output token, all read from the last position.

    Args:
        voc_size: vocabulary size (input tokens and output classes).
        pos_encoding: maximum sequence length.
        d_model: embedding dimension (d).
        m_hidden: number of random features in each attention block (m).
        n_layers: number of blocks.
        num_outputs: number of output tokens to predict.
        gamma: memory decay in each attention block.
    """

    def __init__(self, voc_size, pos_encoding, d_model, m_hidden, n_layers,
                 num_outputs=1, gamma=0.95):
        super().__init__()
        self.tokens_embed = nn.Embedding(voc_size, d_model)
        self.pos_embed = nn.Embedding(pos_encoding, d_model)

        self.layers = nn.ModuleList([
            LayerNAT(d_model, m_hidden, gamma=gamma) for _ in range(n_layers)
        ])

        self.output_heads = nn.ModuleList([
            nn.Linear(d_model, voc_size) for _ in range(num_outputs)
        ])

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)

    def get_hidden_states(self, x):
        """Residual stream after the last block, shape (batch, seq, d_model)."""
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        X = self.tokens_embed(x) + self.pos_embed(pos)

        for layer in self.layers:
            X = layer(X)

        return X

    def forward(self, x):
        out = self.get_hidden_states(x)[:, -1, :]
        return [head(out) for head in self.output_heads]

    @staticmethod
    def compute_loss(logits_list, targets):
        """Mean cross-entropy over heads; exact-match accuracy over all heads."""
        if targets.dim() == 1:
            targets = targets.unsqueeze(1)

        total_loss = 0.0
        correct = torch.ones(targets.size(0), dtype=torch.bool, device=targets.device)

        for i in range(targets.size(1)):
            total_loss += F.cross_entropy(logits_list[i], targets[:, i])
            correct &= logits_list[i].argmax(dim=1) == targets[:, i]

        return total_loss / targets.size(1), correct.float().mean().item()