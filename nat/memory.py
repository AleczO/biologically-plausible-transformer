"""Single neuron-astrocyte unit that processes one token at a time.

Minimal, stateful version of the network from Kozachkov et al. (2023):
a write phase stores tokens in the memory matrix H and the astrocytic
trace g, a read phase queries them.
"""
import torch
from torch import nn


class NANetwork(nn.Module):
    """Neuron-astrocyte network with an explicit read/write switch.

    Args:
        d_in_out: dimension of the input/output layer (d).
        m_hidden: number of hidden neurons / random features (m).
        gamma: exponential decay applied to H on every write.
    """

    def __init__(self, d_in_out, m_hidden, gamma=0.95):
        super().__init__()
        self.d = d_in_out
        self.m = m_hidden

        self.W = nn.Parameter(torch.randn(self.m, self.d))

        self.register_buffer("gamma", torch.tensor(gamma))
        self.register_buffer("b", torch.empty(self.m).uniform_(0, 2 * torch.pi))

        self.register_buffer("H", torch.zeros(self.d, self.m))  # synaptic memory
        self.register_buffer("g", torch.zeros(self.m))          # astrocytic trace

    def _phi(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return torch.cos(self.W @ x + self.b)

    def forward(self, x, r):
        """Process a single token x of shape (d,).

        r = 0 (write): update H and g with x, return the hidden activity h.
        r = 1 (read):  return (1/p) * H h + x, the attention-like readout.
        """
        h = self._phi(x)

        if r == 1:
            p = (1 / self.m) * (self.g @ h)
            return (1 / p) * (self.H @ h) + x

        if r == 0:
            self.H.mul_(self.gamma).add_(torch.outer(x, h) / self.m)
            self.g.add_(h)
            return h

        return h

    def reset_state(self):
        """Clear the memory before processing a new sequence."""
        self.H.zero_()
        self.g.zero_()