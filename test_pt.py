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
    vit_pt = dinov3.load(pt_path)

    return vit_ref, vit_pt


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
