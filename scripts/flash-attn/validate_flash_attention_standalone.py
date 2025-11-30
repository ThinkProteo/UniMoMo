#!/usr/bin/env python
"""
Standalone Validation Script for Flash Attention Integration in EPTAttentionMoT

This script directly imports only the EPT module to avoid dependency issues.

Usage:
    python validate_flash_attention_standalone.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

# Direct import to avoid biotite dependency
import importlib.util
spec = importlib.util.spec_from_file_location("ept", Path(__file__).parent / "models/modules/EPT/ept.py")
ept_module = importlib.util.module_from_spec(spec)

# Mock dependencies that EPT needs
class MockRegister:
    """Mock utils.register module"""
    @staticmethod
    def register(name):
        def decorator(cls):
            return cls
        return decorator

class MockNNUtils:
    """Mock utils.nn_utils module"""
    @staticmethod
    def stable_norm(*args, **kwargs):
        pass

    @staticmethod
    def std_conserve_scatter_sum(x, index, dim=0):
        """Simplified scatter sum"""
        from torch_scatter import scatter_sum
        return scatter_sum(x, index, dim=dim)

    @staticmethod
    def graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8):
        """Convert graph to batch format"""
        device = H.device
        bs = batch_to_nodes.max().item() + 1

        # Count nodes per batch
        counts = torch.bincount(batch_to_nodes, minlength=bs)
        max_n = counts.max().item()

        # Round up to factor_req
        max_n = ((max_n + factor_req - 1) // factor_req) * factor_req

        # Create batched tensor
        H_batch = torch.zeros(bs, max_n, H.shape[-1], dtype=H.dtype, device=device)
        mask = torch.zeros(bs, max_n, dtype=torch.bool, device=device)

        # Fill in data
        offsets = torch.zeros(bs, dtype=torch.long, device=device)
        for i, batch_idx in enumerate(batch_to_nodes):
            idx = offsets[batch_idx]
            H_batch[batch_idx, idx] = H[i]
            mask[batch_idx, idx] = True
            offsets[batch_idx] += 1

        return H_batch, mask

class MockGETTools:
    """Mock GET.tools module"""
    @staticmethod
    def _unit_edges_from_block_edges(block_id, edges, Z, k=None):
        """Simplified edge processing"""
        # Return dummy values for this test
        n_edges = edges.shape[0]
        unit_row = edges[:, 0]
        unit_col = edges[:, 1]
        block_edge_id = torch.arange(n_edges, device=edges.device)
        unit_edge_src_start = torch.zeros(n_edges, dtype=torch.long, device=edges.device)
        unit_edge_src_id = torch.zeros(n_edges, dtype=torch.long, device=edges.device)
        return (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id)

# Inject mocks
sys.modules['utils.register'] = type('module', (), {'R': MockRegister})()
sys.modules['utils.nn_utils'] = type('module', (), {
    'stable_norm': MockNNUtils.stable_norm,
    'std_conserve_scatter_sum': MockNNUtils.std_conserve_scatter_sum,
    'graph_to_batch_nx': MockNNUtils.graph_to_batch_nx,
})()
sys.modules['..GET.tools'] = type('module', (), {
    '_unit_edges_from_block_edges': MockGETTools._unit_edges_from_block_edges,
})()

# Now load the EPT module
try:
    spec.loader.exec_module(ept_module)
    EPTAttentionMoT = ept_module.EPTAttentionMoT
    print("✅ Successfully loaded EPTAttentionMoT")
except Exception as e:
    print(f"❌ Failed to load EPT module: {e}")
    sys.exit(1)


def create_test_inputs(
    batch_size=2,
    n_vae=16,
    l_text=8,
    d_hidden=128,
    n_heads=4,
    num_kv_groups=4,
    device="cuda",
):
    """Create synthetic test inputs for EPTAttentionMoT."""

    # Model parameters
    n_kv_heads = n_heads
    n_q_heads = n_kv_heads * num_kv_groups
    d_head = d_hidden // n_q_heads

    # VAE inputs
    H = torch.randn(batch_size, n_vae, d_hidden, device=device)
    V = torch.randn(batch_size, n_vae, 3, d_hidden, device=device)

    # Text inputs
    text_k = torch.randn(batch_size, l_text, n_kv_heads, d_head, device=device)
    text_v = torch.randn(batch_size, l_text, n_kv_heads, d_head, device=device)
    mask_text = torch.ones(batch_size, l_text, dtype=torch.bool, device=device)

    # Cached info (geometric features)
    D_batch = -torch.rand(batch_size, n_vae, n_vae, device=device) * 10.0
    rbf_feat_batch = torch.randn(1, batch_size, n_heads, n_vae, n_vae, device=device)
    H_mask = torch.ones(batch_size, n_vae, dtype=torch.bool, device=device)

    # Randomly mask some positions
    H_mask[0, n_vae//2:] = False
    mask_text[1, l_text//2:] = False

    cached_info = (D_batch, rbf_feat_batch, H_mask)

    return H, V, text_k, text_v, mask_text, cached_info


def test_numerical_equivalence(tolerance=1e-5, device="cuda"):
    """Test that Flash Attention produces same results as vanilla attention."""

    print("=" * 80)
    print("Flash Attention Validation Test")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Tolerance: {tolerance}")
    print()

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Model configuration
    d_hidden = 128
    n_heads = 4
    num_kv_groups = 4
    layer_idx = 0

    print(f"Model Configuration:")
    print(f"  d_hidden: {d_hidden}")
    print(f"  n_heads (KV): {n_heads}")
    print(f"  num_kv_groups: {num_kv_groups}")
    print(f"  n_q_heads: {n_heads * num_kv_groups}")
    print()

    # Create test inputs
    H, V, text_k, text_v, mask_text, cached_info = create_test_inputs(
        batch_size=2,
        n_vae=16,
        l_text=8,
        d_hidden=d_hidden,
        n_heads=n_heads,
        num_kv_groups=num_kv_groups,
        device=device,
    )

    print(f"Test Input Shapes:")
    print(f"  H (VAE scalar):   {tuple(H.shape)}")
    print(f"  V (VAE vector):   {tuple(V.shape)}")
    print(f"  text_k:           {tuple(text_k.shape)}")
    print(f"  text_v:           {tuple(text_v.shape)}")
    print(f"  mask_text:        {tuple(mask_text.shape)}")
    print()

    # Create two models
    try:
        model_flash = EPTAttentionMoT(
            d_hidden=d_hidden,
            d_ffn=256,
            n_heads=n_heads,
            layer_idx=layer_idx,
            num_kv_groups=num_kv_groups,
            use_flash_attn=True,
            qk_norm=False,
        ).to(device)

        model_vanilla = EPTAttentionMoT(
            d_hidden=d_hidden,
            d_ffn=256,
            n_heads=n_heads,
            layer_idx=layer_idx,
            num_kv_groups=num_kv_groups,
            use_flash_attn=False,
            qk_norm=False,
        ).to(device)

        print("✅ Models created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Copy weights
    model_vanilla.load_state_dict(model_flash.state_dict())

    # Set to eval mode
    model_flash.eval()
    model_vanilla.eval()

    print("Running forward passes...")

    # Run both models
    try:
        with torch.no_grad():
            H_out_flash, V_out_flash = model_flash(
                H, V, cached_info, text_k, text_v, mask_text
            )

            H_out_vanilla, V_out_vanilla = model_vanilla(
                H, V, cached_info, text_k, text_v, mask_text
            )
        print("✅ Forward passes completed")
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 80)
    print("Numerical Equivalence Check")
    print("=" * 80)

    # Compare outputs
    h_diff = torch.abs(H_out_flash - H_out_vanilla)
    v_diff = torch.abs(V_out_flash - V_out_vanilla)

    h_max_diff = h_diff.max().item()
    h_mean_diff = h_diff.mean().item()
    v_max_diff = v_diff.max().item()
    v_mean_diff = v_diff.mean().item()

    print(f"\nScalar Output (H) Differences:")
    print(f"  Max absolute difference:  {h_max_diff:.2e}")
    print(f"  Mean absolute difference: {h_mean_diff:.2e}")
    print(f"  Output shape: {tuple(H_out_flash.shape)}")

    print(f"\nVector Output (V) Differences:")
    print(f"  Max absolute difference:  {v_max_diff:.2e}")
    print(f"  Mean absolute difference: {v_mean_diff:.2e}")
    print(f"  Output shape: {tuple(V_out_flash.shape)}")

    # Check if within tolerance
    h_pass = h_max_diff < tolerance
    v_pass = v_max_diff < tolerance

    print()
    print("=" * 80)
    print("Validation Results")
    print("=" * 80)
    print(f"Scalar output (H): {'✅ PASS' if h_pass else '❌ FAIL'}")
    print(f"Vector output (V): {'✅ PASS' if v_pass else '❌ FAIL'}")
    print()

    if h_pass and v_pass:
        print("🎉 SUCCESS: Flash Attention produces numerically equivalent results!")
        print(f"   All differences are within tolerance ({tolerance:.1e})")
        return True
    else:
        print("❌ FAILURE: Outputs differ beyond tolerance")
        print(f"   Tolerance: {tolerance:.1e}")
        if not h_pass:
            print(f"   H max diff: {h_max_diff:.2e} (exceeds tolerance)")
        if not v_pass:
            print(f"   V max diff: {v_max_diff:.2e} (exceeds tolerance)")
        return False


def main():
    """Run validation test."""

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")
        print()

    # Run test
    success = test_numerical_equivalence(tolerance=1e-5, device=device)

    print("\n" + "=" * 80)
    if success:
        print("🎉 VALIDATION SUCCESSFUL!")
        print("\nFlash Attention integration is working correctly.")
        return 0
    else:
        print("❌ VALIDATION FAILED")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
