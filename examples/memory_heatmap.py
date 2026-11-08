"""Write two different token sequences into NANetwork and compare the memory H.

Run from the repository root:  python -m examples.memory_heatmap
"""
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn

from nat import NANetwork


def write_sequence(model, embed, token_ids):
    """Reset the memory, write every token, return a copy of H."""
    model.reset_state()
    for x in embed(token_ids):
        model(x, r=0)
    return model.H.detach().cpu().numpy().copy()


if __name__ == "__main__":
    torch.manual_seed(0)

    vocab_size = 1000
    d_model = 16     # d
    m_hidden = 128   # m

    embed = nn.Embedding(vocab_size, d_model)
    model = NANetwork(d_model, m_hidden)

    with torch.no_grad():
        h_a = write_sequence(model, embed, torch.tensor([10, 20, 30]))
        h_b = write_sequence(model, embed, torch.tensor([500, 510, 520]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, h, title in [(ax1, h_a, "Memory H: context 1"), (ax2, h_b, "Memory H: context 2")]:
        sns.heatmap(h, ax=ax, cmap="magma", cbar=False)
        ax.set_title(title)
        ax.set_ylabel("Input features (d)")
        ax.set_xlabel("Hidden neurons (m)")

    plt.tight_layout()
    plt.show()