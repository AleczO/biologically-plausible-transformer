"""Shape, causality and training-step checks. Run `pytest` from the repo root."""
import math

import torch

from nat import NAT, AttentionNAT, NANetwork

D, M = 16, 32


def test_attention_shape():
    attn = AttentionNAT(D, M)
    X = torch.randn(3, 5, D)
    assert attn(X).shape == (3, 5, D)


def test_attention_is_causal():
    """Changing the last token must not change outputs at earlier positions."""
    torch.manual_seed(0)
    attn = AttentionNAT(D, M)
    X = torch.randn(2, 6, D)
    X2 = X.clone()
    X2[:, -1] += 1.0
    assert torch.allclose(attn(X)[:, :-1], attn(X2)[:, :-1])


def test_model_forward_and_backward():
    torch.manual_seed(0)
    model = NAT(voc_size=11, pos_encoding=4, d_model=D, m_hidden=M,
                n_layers=2, num_outputs=3)
    x = torch.randint(0, 11, (8, 4))
    y = torch.randint(0, 11, (8, 3))

    logits = model(x)
    assert len(logits) == 3 and all(l.shape == (8, 11) for l in logits)

    loss, acc = model.compute_loss(logits, y)
    loss.backward()
    assert math.isfinite(loss.item()) and 0.0 <= acc <= 1.0


def test_memory_unit_write_and_read():
    torch.manual_seed(0)
    net = NANetwork(D, M)
    with torch.no_grad():
        for x in torch.randn(3, D):
            net(x, r=0)
        assert net.H.abs().sum() > 0
        assert net(torch.randn(D), r=1).shape == (D,)
        net.reset_state()
        assert net.H.abs().sum() == 0 and net.g.abs().sum() == 0