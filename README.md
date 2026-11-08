# Biologically Plausible Transformer

PyTorch implementation of a transformer whose attention is computed by a
neuron-astrocyte network, following Kozachkov, Kastanenka and Krotov
(2023) [[1]](#references). Attention is evaluated recurrently: a synaptic
memory matrix `H` and an astrocytic tra ce `g` are updated token by token,
which makes the cost linear in sequence length and the state size independent
of it.

## Structure

```
.
├── nat/
│   ├── memory.py      # NANetwork: single unit, one token at a time, explicit read/write
│   ├── attention.py   # AttentionNAT: the same mechanism over a batch of sequences
│   └── model.py       # LayerNAT block and the full NAT model
├── examples/
│   └── memory_heatmap.py   # writes two contexts into NANetwork and plots H
├── tests/              # pytest checks (shapes, causality, training step)
└── docs/
    └── description_draft.pdf   # notes on the model (Polish, draft)
```

## Model

**Feature map.** Inputs are L2-normalized and passed through a random-Fourier-
feature-style map [[2]](#references):

$$\phi(x) = \cos\left(W \frac{x}{\lVert x \rVert} + b\right), \qquad W \in \mathbb{R}^{m \times d},\; b \sim \mathcal{U}(0, 2\pi)$$

`b` is fixed; `W` is a trainable parameter. `NANetwork` uses one `W` for keys
and queries (weight tying, as in [[1]](#references)); `AttentionNAT` uses
separate `W_k` and `W_q`. Values are the inputs themselves (`W_V = I`).

**Write** (token `x_t` stored in memory, with decay `γ`):

$$H_t = \gamma H_{t-1} + \tfrac{1}{m}\, x_t\, \phi_k(x_t)^\top, \qquad g_t = g_{t-1} + \phi_k(x_t)$$

**Read** (query with the next token):

$$y_t = \frac{1}{p_t} H_{t-1}\, \phi_q(x_t) + x_t, \qquad p_t = \tfrac{1}{m}\, g_{t-1}^\top \phi_q(x_t)$$

`H` (shape `d × m`) plays the role of the numerator of linear attention and
`p` the role of the normalizing denominator. In `AttentionNAT`, each position
first reads from the memory of the previous tokens and then writes itself,
so the layer is causal; the first position returns its input unchanged.

**Full model (`NAT`).** Token and learned positional embeddings, `n_layers`
pre-norm blocks (`AttentionNAT` + ReLU MLP, with residual connections), and
one linear output head per predicted token, read from the last position.

Implementation notes:

- `γ` decays `H` but not `g`.
- Because `cos` features can be negative, `p` can be ≤ 0; `AttentionNAT`
  clamps it to `1e-4`, `NANetwork` does not.
- The memory state has a fixed size of `d·m + m` regardless of sequence length.

## Quick start

```bash
git clone https://github.com/AleczO/biologically-plausible-transformer.git
cd biologically-plausible-transformer
pip install -r requirements.txt

pytest                              # run the tests
python -m examples.memory_heatmap   # plot the memory H for two contexts
```

```python
import torch
from nat import NAT

model = NAT(voc_size=100, pos_encoding=8, d_model=64, m_hidden=128,
            n_layers=2, num_outputs=1)
x = torch.randint(0, 100, (32, 8))        # batch of token sequences
logits = model(x)                         # list with one (32, 100) tensor per output head
loss, acc = model.compute_loss(logits, torch.randint(0, 100, (32,)))
```

## References

[1] L. Kozachkov, K. V. Kastanenka, and D. Krotov, "Building transformers from neurons and astrocytes," *Proc. Natl. Acad. Sci. U.S.A.*, vol. 120, no. 34, p. e2219150120, 2023. doi: [10.1073/pnas.2219150120](https://doi.org/10.1073/pnas.2219150120)

[2] A. Rahimi and B. Recht, "Random features for large-scale kernel machines," in *Advances in Neural Information Processing Systems*, vol. 20, 2007.