#!/usr/bin/env python
"""
Validation Script for Flash Attention Integration in EPTAttentionMoT

Tests numerical equivalence between vanilla attention and Flash Attention paths.
Tolerance: 1e-5 for floating point comparison.

Usage:
    python validate_flash_attention.py
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Add UniMoMo to path
UNIMOMO_ROOT = Path(__file__).parent
sys.path.insert(0, str(UNIMOMO_ROOT))

from models.modules.EPT.ept_mot import EPTAttentionMoT


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
    D_batch = -torch.rand(batch_size, n_vae, n_vae, device=device) * 10.0  # Distance matrix
    rbf_feat_batch = torch.randn(1, batch_size, n_heads, n_vae, n_vae, device=device)  # RBF features
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

    # Create two models: one with Flash Attention, one with vanilla
    model_flash = EPTAttentionMoT(
        d_hidden=d_hidden,
        d_ffn=256,
        n_heads=n_heads,
        layer_idx=layer_idx,
        num_kv_groups=num_kv_groups,
        use_flash_attn=True,  # Flash Attention enabled
        qk_norm=False,
    ).to(device)

    model_vanilla = EPTAttentionMoT(
        d_hidden=d_hidden,
        d_ffn=256,
        n_heads=n_heads,
        layer_idx=layer_idx,
        num_kv_groups=num_kv_groups,
        use_flash_attn=False,  # Vanilla attention
        qk_norm=False,
    ).to(device)

    # Copy weights from Flash model to vanilla model for fair comparison
    model_vanilla.load_state_dict(model_flash.state_dict())

    # Set both to eval mode
    model_flash.eval()
    model_vanilla.eval()

    print("Running forward passes...")

    # Run both models
    with torch.no_grad():
        H_out_flash, V_out_flash = model_flash(
            H, V, cached_info, text_k, text_v, mask_text
        )

        H_out_vanilla, V_out_vanilla = model_vanilla(
            H, V, cached_info, text_k, text_v, mask_text
        )

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


def test_fallback_mechanism(device="cuda"):
    """Test that fallback to vanilla attention works correctly."""

    print("\n" + "=" * 80)
    print("Testing Fallback Mechanism")
    print("=" * 80)

    # Create a model with Flash Attention
    model = EPTAttentionMoT(
        d_hidden=128,
        d_ffn=256,
        n_heads=4,
        layer_idx=0,
        use_flash_attn=True,
    ).to(device)

    # Create inputs
    H, V, text_k, text_v, mask_text, cached_info = create_test_inputs(device=device)

    model.eval()

    # Try forward pass - should work with Flash Attention
    with torch.no_grad():
        try:
            H_out, V_out = model(H, V, cached_info, text_k, text_v, mask_text)
            print("✅ Flash Attention forward pass successful")
            print(f"   Output shapes: H={tuple(H_out.shape)}, V={tuple(V_out.shape)}")
        except Exception as e:
            print(f"❌ Flash Attention forward pass failed: {e}")
            return False

    return True


def test_backward_compatibility(device="cuda"):
    """Test that model can be created with default parameters."""

    print("\n" + "=" * 80)
    print("Testing Backward Compatibility")
    print("=" * 80)

    # Create model with minimal parameters (use_flash_attn defaults to True)
    try:
        model = EPTAttentionMoT(
            d_hidden=128,
            d_ffn=256,
            n_heads=4,
        ).to(device)
        print("✅ Model creation with default parameters successful")
        print(f"   use_flash_attn: {model.use_flash_attn}")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False


def main():
    """Run all validation tests."""

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        print(f"PyTorch version: {torch.__version__}")
        print()

    # Run tests
    results = {}

    # Test 1: Numerical equivalence
    results["numerical_equivalence"] = test_numerical_equivalence(
        tolerance=1e-5,
        device=device
    )

    # Test 2: Fallback mechanism
    results["fallback"] = test_fallback_mechanism(device=device)

    # Test 3: Backward compatibility
    results["backward_compat"] = test_backward_compatibility(device=device)

    # Summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s}: {status}")

    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nFlash Attention integration is working correctly.")
        print("You can now use EPTAttentionMoT with use_flash_attn=True for improved performance.")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease review the failures above before using Flash Attention in production.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
