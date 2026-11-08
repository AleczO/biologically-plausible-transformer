"""Sequence-level neuron-astrocyte attention (linear-time, recurrent)."""
import torch
from torch import nn


class AttentionNAT(nn.Module):
    """Neuron-astrocyte attention over a whole sequence, computed recurrently.

    At step t the output is (1/p_t) * H_{t-1} phi_q(x_t) + x_t, i.e. a read
    from the memory built from tokens 0..t-1, after which x_t is written into
    H (with decay gamma) and phi_k(x_t) into g. Cost is linear in sequence
    length and the state (H, g) has a fixed size of d*m + m.
    """
    def __init__(self, d_model, m_hidden, gamma=0.95):
        super().__init__()
        self.d = d_model
        self.m = m_hidden
        self.gamma = gamma

        self.W_q = nn.Parameter(torch.randn(self.m, self.d) * 0.02)
        self.W_k = nn.Parameter(torch.randn(self.m, self.d) * 0.02)
        
        self.register_buffer('b', torch.empty(self.m).uniform_(0, 2 * torch.pi))

    def _phi_q(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return torch.cos(x @ self.W_q.t() + self.b) 

    def _phi_k(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return torch.cos(x @ self.W_k.t() + self.b)

    def forward(self, X):
        if X.dim() == 2:
            X = X.unsqueeze(1)

        batch_size, seq_len, _ = X.shape

        H = torch.zeros(batch_size, self.d, self.m, dtype=X.dtype, device=X.device)
        g = torch.zeros(batch_size, self.m, dtype=X.dtype, device=X.device)

        outputs = []

        for t in range(seq_len):
            x_t = X[:, t, :]
            
            h_q = self._phi_q(x_t)
            h_k = self._phi_k(x_t)
            
            if t > 0:
                p = (g * h_q).sum(dim=-1) / self.m
                p = torch.clamp(p, min=1e-4).unsqueeze(-1)
                
                H_h = torch.bmm(H, h_q.unsqueeze(-1)).squeeze(2)
                out_t = (1 / p) * H_h + x_t
            else:
                out_t = x_t
                
            outputs.append(out_t.unsqueeze(1))
            
            outer = torch.bmm(x_t.unsqueeze(-1), h_k.unsqueeze(1)) / self.m
            H = H.clone() * self.gamma + outer
            g = g.clone() + h_k
            
        return torch.cat(outputs, dim=1)