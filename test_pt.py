# tests/test_pt.py

import numpy as np
import pytest
import torch

import dinov3_pt as dinov3


@pytest.fixture(scope="function")
def vits(vit_paths):
    ref_name, (ref_path, pt_path, _) = vit_paths
    vit_ref = torch.hub.load(
        "facebookresearch/dinov3", ref_name, source="github", weights=ref_path
    ).eval()
    vit_pt = dinov3.load(ref_name, pt_path)

    return vit_ref, vit_pt


@pytest.mark.parametrize("h,w", [(2, 2), (4, 4), (16, 16), (8, 12)])
def test_rope_fwd(h, w, vits):
    vit_ref, vit_pt = vits
    out_ref_h, out_ref_w = vit_ref.rope_embed(H=h, W=w)
    out_pt_h, out_pt_w = vit_pt.rope_embed(h=h, w=w)

    # Check shape
    assert out_ref_h.shape == out_pt_h.shape
    assert out_ref_w.shape == out_pt_w.shape

    # Check values
    np.testing.assert_allclose(out_ref_h, out_pt_h, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(out_ref_w, out_pt_w, rtol=1e-6, atol=1e-6)


def test_patch_embed_fwd(vits):
    vit_ref, vit_pt = vits
    torch.manual_seed(14)

    x_1chw = torch.rand((1, 3, vit_pt.cfg.img_size, vit_pt.cfg.img_size))

    # PyTorch forward pass
    with torch.no_grad():
        out_pt = vit_pt.patch_embed(x_1chw)
        out_ref = vit_ref.patch_embed(x_1chw)

    # Check shapes
    assert out_ref.shape == out_pt.shape

    # Check values
    np.testing.assert_allclose(out_ref, out_pt, rtol=1e-6, atol=1e-4)


def test_self_attn_fwd(vits):
    vit_ref, vit_pt = vits
    torch.manual_seed(14)

    h = w = vit_pt.cfg.img_size // vit_pt.cfg.patch_size
    n_tok = h * w + vit_pt.cfg.n_storage_tokens + 1
    x_1nd = torch.rand((1, n_tok, vit_pt.cfg.embed_dim))

    rope_sincos_ref = vit_ref.rope_embed(H=h, W=w)
    rope_sincos_pt = vit_pt.rope_embed(h=h, w=w)

    # PyTorch forward pass
    with torch.no_grad():
        out_pt = vit_pt.blocks[0].attn(x_1nd, rope=rope_sincos_pt)
        out_ref = vit_ref.blocks[0].attn(x_1nd, rope=rope_sincos_ref)

    # Check shapes
    assert out_ref.shape == out_pt.shape

    # Check values
    np.testing.assert_allclose(out_ref, out_pt, rtol=1e-6, atol=1e-4)


def test_self_attn_no_rope_fwd(vits):
    vit_ref, vit_pt = vits
    torch.manual_seed(12)

    h = w = vit_pt.cfg.img_size // vit_pt.cfg.patch_size
    n_tok = h * w + vit_pt.cfg.n_storage_tokens + 1
    x_1nd = torch.rand((1, n_tok, vit_pt.cfg.embed_dim))

    # PyTorch forward pass
    with torch.no_grad():
        out_pt = vit_pt.blocks[0].attn(x_1nd)
        out_ref = vit_ref.blocks[0].attn(x_1nd)

    # Check shapes
    assert out_ref.shape == out_pt.shape

    # Check values
    np.testing.assert_allclose(out_ref, out_pt, rtol=1e-6, atol=1e-4)
