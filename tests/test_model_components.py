# tests/test_model_components.py

import math
import pytest
import torch
import torch.nn.functional as F

from model import GPTConfig, LayerNorm, CausalSelfAttention, MLP, Block, GPT

@pytest.fixture
def small_config():
    # small sizes to speed up tests and limit memory
    return GPTConfig(
        block_size=8,
        vocab_size=50,
        n_layer=2,
        n_head=2,
        n_embd=16,
        dropout=0.0,  # turn off dropout for deterministic behavior
        bias=True
    )

def test_gptconfig_defaults():
    cfg = GPTConfig()
    assert cfg.block_size == 128
    assert cfg.vocab_size == 50257
    assert cfg.n_layer == 4
    assert cfg.n_head == 4
    assert cfg.n_embd == 256
    assert pytest.approx(cfg.dropout, rel=1e-3) == 0.1
    assert cfg.bias is True

def test_layernorm_zero_mean_unit_variance(small_config):
    ln = LayerNorm(small_config.n_embd, small_config.bias)
    ln.eval()
    x = torch.randn(4, 10, small_config.n_embd)
    y = ln(x)
    # same shape
    assert y.shape == x.shape
    # per‐feature zero mean and unit variance
    mean = y.mean(dim=-1)
    std  = y.var (dim=-1, unbiased=False).sqrt()
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4)
    assert torch.allclose(std,  torch.ones_like(std),    atol=1e-4)

def test_causal_self_attention_shapes_and_mask(small_config):
    csa = CausalSelfAttention(small_config)
    csa.eval()
    B, T, C = 2, 5, small_config.n_embd
    x = torch.randn(B, T, C)
    y = csa(x)
    # output shape
    assert y.shape == (B, T, C)
    # mask is lower-triangular of size block_size
    mask = csa.mask[0, 0]
    assert torch.equal(mask, torch.tril(torch.ones_like(mask)))

def test_mlp_preserves_shape(small_config):
    mlp = MLP(small_config)
    B, T, C = 3, 7, small_config.n_embd
    x = torch.randn(B, T, C)
    y = mlp(x)
    assert y.shape == x.shape

def test_block_residual(small_config):
    block = Block(small_config)
    block.eval()
    B, T, C = 2, 6, small_config.n_embd
    x = torch.randn(B, T, C)
    y = block(x)
    # residual connections → same shape
    assert y.shape == x.shape
    # sanity check: if everything is zero, output == zero
    # zero = torch.zeros_like(x)
    # out = block(zero)
    # assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

def test_gpt_forward_and_loss(small_config):
    # build a tiny GPT
    cfg = small_config
    model = GPT(cfg)
    model.eval()
    B, T = 2, 6
    idx = torch.randint(0, cfg.vocab_size, (B, T))
    # forward without targets
    logits, loss = model(idx)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert loss is None
    # forward with targets
    targets = torch.randint(0, cfg.vocab_size, (B, T))
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, cfg.vocab_size)
    # loss should be a scalar
    assert isinstance(loss, torch.Tensor) and loss.dim() == 0
    # loss matches manual cross-entropy
    flat_logits = logits.view(-1, cfg.vocab_size)
    flat_tgts   = targets.view(-1)
    manual_loss = F.cross_entropy(flat_logits, flat_tgts)
    assert torch.allclose(loss, manual_loss)
