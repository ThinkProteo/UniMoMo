#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
EPT-MoT: Mixture-of-Thought extensions for EPT (Equivariant Protein Transformer).

This module extends the original EPT with cross-modal attention capabilities,
allowing the protein structure model to attend to text embeddings from an LLM.

Classes:
    - EPTLayerMoT: MoT-enhanced transformer layer with text cross-attention
    - EPTAttentionMoT: Multi-head attention with text K/V conditioning
    - TransformerMoT: Full transformer stack with per-layer text conditioning
    - XTransEncoderActMoT: Top-level encoder wrapper for MoT
"""

import math
from typing import Optional, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_mean

from utils.nn_utils import std_conserve_scatter_sum, graph_to_batch_nx

from ..GET.tools import _unit_edges_from_block_edges
from .radial_basis import RadialBasis

# Import base classes from original ept.py
from .ept import GVPFFNLayer, SubLayerWrapper

try:
    from xformers.ops import memory_efficient_attention as attn_func
    xformers_enable = True
except:
    xformers_enable = False


# ==============================================================================
# Rotary Position Embedding (RoPE) for EPT-MoT
# ==============================================================================

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding for EPT-MoT attention.
    
    This enables position-aware attention in the structure-to-text cross-attention,
    where:
    - Text tokens have sequential positions (0, 1, 2, ...)
    - VAE (structure) tokens all share position = text_length + 1
    
    This design allows structure tokens to "see" the full text context
    while being positioned after the text sequence.
    """
    
    def __init__(self, dim: int, base: float = 100000.0, max_position_embeddings: int = 4096):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        
        # Compute inverse frequencies for rotation
        # RoPE operates on pairs of elements, so we need dim/2 frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Cache for cos/sin (lazily computed)
        self._cos_cached = None
        self._sin_cached = None
        self._cached_max_pos = 0
    
    def _update_cache(self, max_position: int, device: torch.device, dtype: torch.dtype):
        """Update the cos/sin cache if needed."""
        if max_position <= self._cached_max_pos and self._cos_cached is not None:
            return
        
        # Compute for all positions up to max_position
        self._cached_max_pos = max(max_position, self.max_position_embeddings)
        seq = torch.arange(self._cached_max_pos, device=device, dtype=torch.float32)
        
        # freqs: [max_pos, dim/2]
        freqs = torch.outer(seq, self.inv_freq.to(device))
        
        # Duplicate for full dimension: [max_pos, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        """
        Get cos/sin embeddings for given position IDs.
        
        Args:
            x: Input tensor to determine device and dtype [B, n_heads, seq_len, dim]
            position_ids: Position indices [B, seq_len]
            
        Returns:
            cos, sin tensors both of shape [B, seq_len, dim]
        """
        max_pos = int(position_ids.max().item()) + 1
        self._update_cache(max_pos, x.device, x.dtype)
        
        # Gather cos/sin for each position: [B, seq_len, dim]
        cos = self._cos_cached[position_ids]
        sin = self._sin_cached[position_ids]
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half of the features by 90 degrees.
    Splits the last dimension in half and swaps with negation.
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_embedding(
    x: torch.Tensor, 
    cos: torch.Tensor, 
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> torch.Tensor:
    """
    Apply Rotary Position Embedding to input tensor.
    
    Args:
        x: Input tensor [B, n_heads, seq_len, dim]
        cos: Cosine embeddings [B, seq_len, dim]
        sin: Sine embeddings [B, seq_len, dim]
        unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting to heads
        
    Returns:
        Tensor with RoPE applied, same shape as input
    """
    # Unsqueeze to broadcast over heads: [B, 1, seq_len, dim]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    
    return x * cos + rotate_half(x) * sin


# ==============================================================================


class EPTLayerMoT(nn.Module):
    """
    MoT-enhanced EPT layer with optional text cross-attention.
    
    This layer wraps EPTAttentionMoT for self-attention + text cross-attention,
    followed by GVPFFNLayer for feed-forward processing.
    """
    
    def __init__(
        self,
        d_hidden,
        d_ffn,
        n_heads,
        layer_idx=-1,
        act_fn=nn.SiLU(),
        layer_norm="pre",
        residual=True,
        efficient=False,
        vector_act="none",
        attn_bias=True,
        num_kv_groups=4,
    ):
        super(EPTLayerMoT, self).__init__()
        self.attn_layer = SubLayerWrapper(
            EPTAttentionMoT(
                d_hidden=d_hidden,
                d_ffn=d_ffn,
                n_heads=n_heads,
                layer_idx=layer_idx,
                act_fn=act_fn,
                layer_norm=layer_norm,
                residual=residual,
                vector_act=vector_act,
                attn_bias=attn_bias,
                num_kv_groups=num_kv_groups,
            ),
            d_hidden,
            layer_norm,
            residual,
        )
        self.ffn_layer = SubLayerWrapper(
            GVPFFNLayer(d_hidden, d_ffn, act_fn, vector_act=vector_act),
            d_hidden,
            layer_norm,
            residual,
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        H,  # [B, N_vae_max, d_hidden]
        V,  # [B, N_vae_max, 3, d_hidden]
        cached_info=None,
        text_k: Optional[torch.Tensor] = None,  # [B, L_text_max, n_kv_heads, d_attn]
        text_v: Optional[torch.Tensor] = None,  # [B, L_text_max, n_kv_heads, d_head]
        mask_text: Optional[torch.Tensor] = None,  # [B, L_text_max]
        text_lengths: Optional[torch.Tensor] = None,  # [B], actual text lengths for RoPE
    ):
        """
        Forward pass with optional text cross-attention.
        
        Args:
            H: Node scalar features [B, N, d_hidden]
            V: Node vector features [B, N, 3, d_hidden]
            cached_info: Tuple of (D_batch, rbf_feat_batch, H_mask)
            text_k: Text key embeddings from LLM [B, L, n_kv_heads, d_head]
            text_v: Text value embeddings from LLM [B, L, n_kv_heads, d_head]
            mask_text: Boolean mask for valid text positions [B, L]
            text_lengths: Actual text lengths per sample for RoPE positions [B]
            
        Returns:
            H: Updated scalar features
            V: Updated vector features
        """
        H, V = self.attn_layer(
            H,
            V,
            cached_info=cached_info,
            text_k=text_k,
            text_v=text_v,
            mask_text=mask_text,
            text_lengths=text_lengths,
        )
        H, V = self.ffn_layer(H, V)
        return H, V


class EPTAttentionMoT(nn.Module):
    """
    MoT-style EPT attention with BATCHED computation.
    
    This attention module supports:
    - Self-attention among protein structure nodes
    - Cross-attention to text embeddings from an LLM
    - Grouped Query Attention (GQA) for efficiency
    
    Architecture notes:
    - Queries and Keys are projected to 4 * d_head (matching original EPT)
    - Text Keys are projected to match this expanded dimension
    - Values concatenate scalar (d_head) and vector (3*d_head) components
    """
    
    # Class-level debug settings
    _debug_text_attn = True  # Set to True to print text attention stats
    _debug_step_counter = 0
    _debug_print_interval = 50  # Print every N forward passes

    def __init__(
        self,
        d_hidden: int,
        d_ffn: int,
        n_heads: int,
        layer_idx: int = -1,
        act_fn=nn.SiLU(),
        layer_norm: str = "pre",
        residual: bool = True,
        vector_act: str = "none",
        attn_bias: bool = True,
        qk_norm: bool = True,
        num_kv_groups: int = 4,
        rope_theta: float = 100000.0, #different from qwen3 
        max_position_embeddings: int = 4096,
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.layer_idx = layer_idx

        # GQA configuration
        self.num_kv_groups = num_kv_groups
        self.n_kv_heads = n_heads
        self.n_q_heads = self.n_kv_heads * self.num_kv_groups
        
        if d_hidden % self.n_q_heads != 0:
            raise ValueError(
                f"d_hidden ({d_hidden}) must be divisible by n_q_heads ({self.n_q_heads})"
            )
        
        # Standard head dimension
        self.d_head = d_hidden // self.n_q_heads
        
        # Original EPT uses 4x expansion for Q/K
        # This matches the dimensionality of V (1 scalar + 3 vectors = 4 components)
        self.d_qk_head = 4 * self.d_head

        # Scale factor (matches original EPT: 0.5 / sqrt(d_head))
        self.scale_factor = 0.5 / math.sqrt(self.d_head)

        # VAE branch projections (Q/K use expanded d_qk_head)
        self.scaler_q = nn.Linear(d_hidden, self.n_q_heads * self.d_qk_head, bias=attn_bias)
        self.scaler_k = nn.Linear(d_hidden, self.n_kv_heads * self.d_qk_head, bias=attn_bias)
        
        # VAE Value projections (scalar + vector = 4 * d_head total)
        self.scaler_v = nn.Linear(d_hidden, self.n_kv_heads * self.d_head, bias=attn_bias)
        self.vector_v = nn.Linear(d_hidden, self.n_kv_heads * self.d_head, bias=False)

        # Text Key projection: d_head -> 4*d_head to match VAE K dimension
        # diffusion match to text; not the other way around!!
        # self.text_k_proj = nn.Linear(self.d_head, self.d_qk_head, bias=False)
        
        # Text Value projection: maps Qwen's head_dim to EPT's scalar d_head
        # Text values are pure scalar embeddings (no geometric vector component)
        # We project them to d_head and pad with zeros for the vector portion
        # Qwen3-4B head_dim=128, EPT d_head varies based on config
        self.text_v_proj = nn.Linear(self.d_qk_head, self.d_head, bias=False)

        # Output projections
        self.scaler_o = nn.Linear(self.n_q_heads * self.d_head, d_hidden)
        self.vector_o = nn.Linear(self.n_q_heads * self.d_head, d_hidden, bias=False)

        # Optional Q/K LayerNorm (applied on expanded 4x dimension)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.d_qk_head)
            self.k_norm = nn.LayerNorm(self.d_qk_head)
            self.text_k_norm = nn.LayerNorm(self.d_qk_head)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()
            self.text_k_norm = nn.Identity()
        
        # Rotary Position Embedding for position-aware attention
        # Applied on expanded Q/K dimension (d_qk_head = 4 * d_head)
        self.rotary_emb = RotaryEmbedding(
            dim=self.d_qk_head,
            base=rope_theta,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(
        self,
        H: torch.Tensor,  # [B, N_vae_max, d_hidden]
        V: torch.Tensor,  # [B, N_vae_max, 3, d_hidden]
        cached_info,      # (D_batch, rbf_feat_batch, H_mask)
        text_k: Optional[torch.Tensor] = None,  # [B, L_text_max, n_kv_heads, d_head]
        text_v: Optional[torch.Tensor] = None,  # [B, L_text_max, n_kv_heads, d_head]
        mask_text: Optional[torch.Tensor] = None,  # [B, L_text_max], 1=valid
        text_lengths: Optional[torch.Tensor] = None,  # [B], actual text lengths per sample
    ):
        B, N_vae_max, _ = H.shape
        device = H.device

        D_batch, rbf_feat_batch, H_mask = cached_info

        # Handle text inputs
        if text_k is None or text_v is None:
            L_text_max = 0
            text_v = H.new_zeros(B, 0, self.n_kv_heads, self.d_head)
            mask_text = H.new_ones(B, 0, dtype=torch.bool)
            text_lengths = torch.zeros(B, dtype=torch.long, device=device)
        else:
            assert text_k.shape[0] == B, "Batch size mismatch"
            L_text_max = text_k.shape[1]
            
            if mask_text is None:
                mask_text = torch.ones(B, L_text_max, dtype=torch.bool, device=device)
            
            # Derive text_lengths from mask_text if not provided
            if text_lengths is None:
                text_lengths = mask_text.sum(dim=1).long()  # [B]

        # Compute VAE Q/K/V projections
        H_q_vae = self.scaler_q(H).view(B, N_vae_max, self.n_q_heads, self.d_qk_head)
        H_k_vae = self.scaler_k(H).view(B, N_vae_max, self.n_kv_heads, self.d_qk_head)
        
        # VAE Values (Scalar part)
        H_v_vae = self.scaler_v(H).view(B, N_vae_max, self.n_kv_heads, self.d_head)

        # Vector features: [B, N, 3, n_kv_heads, d_head] -> [B, N, n_kv_heads, 3*d_head]
        V_v_vae = self.vector_v(V).view(B, N_vae_max, 3, self.n_kv_heads, self.d_head)
        V_v_vae = V_v_vae.transpose(-2, -3).flatten(start_dim=-2)

        # Concatenate scalar + vector for VAE values: [B, N, n_kv_heads, 4*d_head]
        V_attn_vae = torch.cat([H_v_vae, V_v_vae], dim=-1)

        # Apply Q/K normalization
        H_q_vae = self.q_norm(H_q_vae)
        H_k_vae = self.k_norm(H_k_vae)
        if text_k is not None:
            text_k = self.text_k_norm(text_k)

        # ========== BATCHED ATTENTION ==========

        # Concatenate text and VAE K/V along sequence dimension
        # Handle None text_k (e.g., when using text injection mode)
        if text_k is not None:
            K_full = torch.cat([text_k, H_k_vae], dim=1)  # [B, L_total, n_kv, d_qk_head]
        else:
            K_full = H_k_vae  # No text keys
        
        # CRITICAL FIX: Text values are pure scalar embeddings (Qwen head_dim), but EPT 
        # expects values with [scalar(d_head) + 3*vector(3*d_head)] = 4*d_head structure.
        # We project text_v to scalar portion only, padding vector portion with zeros.
        # This prevents text embeddings from being incorrectly interpreted as 3D vectors.
        if L_text_max > 0 and text_v is not None:
            # text_v: [B, L, n_kv, qwen_head_dim] -> project to [B, L, n_kv, d_head] for scalar
            # Then pad with zeros for vector portion: [B, L, n_kv, 4*d_head]
            text_v_scalar = self.text_v_proj(text_v)  # Learned projection to d_head
            text_v_vector_pad = torch.zeros(
                B, L_text_max, self.n_kv_heads, 3 * self.d_head, 
                device=device, dtype=text_v.dtype
            )  # Zero vector contribution
            text_v_structured = torch.cat([text_v_scalar, text_v_vector_pad], dim=-1)
            V_full = torch.cat([text_v_structured, V_attn_vae], dim=1)  # [B, L_total, n_kv, 4*d_head]
        else:
            V_full = V_attn_vae  # No text values
        
        L_total = L_text_max + N_vae_max

        # Reshape for multi-head attention: [B, n_heads, L, d]
        q = H_q_vae.transpose(1, 2)  # [B, n_q, N_vae, d_qk_head]
        k = K_full.transpose(1, 2)   # [B, n_kv, L_total, d_qk_head]
        v = V_full.transpose(1, 2)   # [B, n_kv, L_total, 4*d_head]

        # ========== ROTARY POSITION EMBEDDING ==========
        # Position scheme:
        # - Text tokens (keys): sequential positions 0, 1, 2, ..., text_len-1
        # - VAE tokens (queries and keys): all share position = text_length
        # 
        # This allows VAE tokens to "see" the full text context while being
        # positioned at the end of the sequence. All VAE tokens share the same
        # relative position to all text tokens.
        
        # Position IDs for queries (VAE tokens only)
        # All VAE tokens have position = text_length for their sample
        # tok_pos_q: [B, N_vae_max]
        tok_pos_q = text_lengths.unsqueeze(1).expand(B, N_vae_max)  # [B, N_vae]
        
        # Position IDs for keys (text + VAE tokens)
        # Text: 0, 1, 2, ..., L_text_max-1 (padding positions will be masked)
        # VAE: all at text_length for each sample
        # tok_pos_k: [B, L_total]
        text_positions = torch.arange(L_text_max, device=device).unsqueeze(0).expand(B, -1)  # [B, L_text]
        vae_positions = text_lengths.unsqueeze(1).expand(B, N_vae_max)  # [B, N_vae]
        tok_pos_k = torch.cat([text_positions, vae_positions], dim=1)  # [B, L_total]
        
        # Get cos/sin embeddings and apply RoPE
        cos_q, sin_q = self.rotary_emb(q, tok_pos_q)
        cos_k, sin_k = self.rotary_emb(k, tok_pos_k)
        
        q = apply_rotary_embedding(q, cos_q, sin_q)
        k = apply_rotary_embedding(k, cos_k, sin_k)


        # Expand K/V for grouped query attention
        k = k.repeat_interleave(self.num_kv_groups, dim=1)  # [B, n_q, L_total, d_qk_head]
        v = v.repeat_interleave(self.num_kv_groups, dim=1)  # [B, n_q, L_total, 4*d_head]

        # Build attention bias (geometric bias only for VAE-VAE interactions)
        bias_geom = rbf_feat_batch[self.layer_idx] + D_batch.unsqueeze(1)

        if bias_geom.shape[1] > 1:
            bias_vae_scalar = bias_geom.mean(dim=1)
        else:
            bias_vae_scalar = bias_geom.squeeze(1)

        bias_full = torch.zeros(B, self.n_q_heads, N_vae_max, L_total, device=device, dtype=H.dtype)
        bias_full[:, :, :, L_text_max:] = bias_vae_scalar.unsqueeze(1)

        combined_mask = torch.cat([mask_text, H_mask], dim=1)
        bias_full = bias_full.masked_fill(~combined_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        # Compute attention (vanilla implementation)
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k)
        attn = F.softmax(attn_scores * self.scale_factor + bias_full, dim=-1)
        
        # DEBUG: Print text attention stats for GT seq debugging
        if L_text_max > 0 and EPTAttentionMoT._debug_text_attn:
            EPTAttentionMoT._debug_step_counter += 1
            if EPTAttentionMoT._debug_step_counter % EPTAttentionMoT._debug_print_interval == 1:
                # attn shape: [B, n_q, N_vae, L_total]
                # Text tokens are at positions 0:L_text_max
                attn_to_text = attn[:, :, :, :L_text_max].sum(dim=-1)  # [B, n_q, N_vae]
                attn_to_vae = attn[:, :, :, L_text_max:].sum(dim=-1)   # [B, n_q, N_vae]
                
                # Compute stats across batch and heads
                text_attn_mean = attn_to_text.mean().item()
                text_attn_max = attn_to_text.max().item()
                text_attn_min = attn_to_text.min().item()
                vae_attn_mean = attn_to_vae.mean().item()
                
                print(f"📊 [Layer {self.layer_idx}] Text Attention Stats (step {EPTAttentionMoT._debug_step_counter}):")
                print(f"   Attn to TEXT: mean={text_attn_mean:.4f}, max={text_attn_max:.4f}, min={text_attn_min:.4f}")
                print(f"   Attn to VAE:  mean={vae_attn_mean:.4f}")
                print(f"   Text tokens: {L_text_max}, VAE tokens: {N_vae_max}")
        
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)  # [B, n_q, N_vae, 4*d_head]

        # Reshape output: [B, n_q, N_vae, 4*d_head] -> [B, N_vae, n_q, 4*d_head]
        out = out.transpose(1, 2)

        # Split scalar and vector parts
        H_out = out[..., :self.d_head].reshape(B, N_vae_max, self.n_q_heads * self.d_head)
        
        V_out_flat = out[..., self.d_head:].reshape(B, N_vae_max, self.n_q_heads, 3, self.d_head)
        
        # RESCALE VECTOR OUTPUT: Text tokens contribute zeros to vector portion, but they
        # still consume attention mass (softmax sums to 1). This dilutes VAE vector contributions.
        # We rescale by 1/(attention_to_VAE) to compensate.
        # attn shape: [B, n_q, N_vae, L_total], VAE positions start at L_text_max
        if L_text_max > 0:
            # Sum attention going to VAE tokens: [B, n_q, N_vae]
            attn_to_vae = attn[:, :, :, L_text_max:].sum(dim=-1)
            # Clamp to avoid division by zero (happens when all attention goes to text)
            attn_to_vae = attn_to_vae.clamp(min=1e-6)
            # Rescale: [B, N_vae, n_q, 3, d_head] / [B, n_q, N_vae] -> need to align dims
            # V_out_flat is [B, N_vae, n_q, 3, d_head], attn_to_vae is [B, n_q, N_vae]
            scale = 1.0 / attn_to_vae.transpose(1, 2)  # [B, N_vae, n_q]
            V_out_flat = V_out_flat * scale.unsqueeze(-1).unsqueeze(-1)  # [B, N_vae, n_q, 3, d_head]
        
        V_out = V_out_flat.transpose(-2, -3).reshape(B, N_vae_max, 3, self.n_q_heads * self.d_head)

        # Output projections
        H_final = self.scaler_o(H_out)
        V_final = self.vector_o(V_out)

        return H_final, V_final


class TransformerMoT(nn.Module):
    """
    Equivariant Adaptive Block Transformer with MoT-style cross-attention.
    
    This is the main transformer backbone that processes protein structure
    while optionally attending to text embeddings. Supports per-layer
    text conditioning via dictionary inputs.
    """

    def __init__(
        self,
        d_hidden,
        d_ffn,
        n_heads,
        n_layers,
        n_rbf,
        d_edge,
        cutoff=7.0,
        act_fn=nn.SiLU(),
        layer_norm="pre",
        residual=True,
        use_edge_feat=False,
        local_mask=False,
        attn_bias=True,
        sparse_k=None,
        efficient=False,
        vector_act="none",
        num_kv_groups=4,
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.use_edge_feat = use_edge_feat
        self.sparse_k = sparse_k
        self.efficient = efficient
        self._local_mask = local_mask
        self.num_kv_groups = num_kv_groups
        
        if self.efficient and not xformers_enable:
            print(
                "xformers are not downloaded, change into custom attention mechanism. "
                "Please install xformers via 'pip3 install -U xformers "
                "--index-url https://download.pytorch.org/whl/cu121', "
                "or see 'https://github.com/facebookresearch/xformers'."
            )
            self.efficient = False

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge + n_rbf, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden * 2),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            act_fn,
            nn.Linear(d_hidden, d_hidden),
        )

        self.final_v = GVPFFNLayer(d_hidden, d_ffn, act_fn, d_output=1)

        self.n_rbf = n_rbf
        if n_rbf > 1:
            self.rbf = RadialBasis(num_radial=n_rbf, cutoff=cutoff)

        if self.use_edge_feat:
            self.rbf_mapping = nn.Linear(n_rbf + d_edge, n_layers)
        else:
            self.rbf_mapping = nn.Linear(n_rbf, n_layers)

        if self.layer_norm == "pre":
            self.ln = nn.LayerNorm(d_hidden)

        # Use EPTLayerMoT for cross-modal attention support
        for i in range(0, n_layers):
            self.add_module(
                f"layer_{i}",
                EPTLayerMoT(
                    d_hidden,
                    d_ffn,
                    n_heads,
                    i,
                    act_fn,
                    layer_norm,
                    residual,
                    self.efficient,
                    vector_act,
                    attn_bias,
                    num_kv_groups,
                ),
            )

    def forward(
        self,
        H,  # [N, d_hidden]
        Z,  # [N, 3]
        block_id,  # [N]
        batch_id,  # [N_block]
        edges,  # [2, E]
        edge_attr=None,  # [E, d_edge]
        topo_edges=None,  # [2, E_topo]
        topo_edge_attr=None,  # [E_topo, d_edge]
        attn_mask=None,  # [B, N_max, N_max]
        text_k: Optional[Union[torch.Tensor, Dict[int, torch.Tensor]]] = None,
        text_v: Optional[Union[torch.Tensor, Dict[int, torch.Tensor]]] = None,
        mask_text: Optional[torch.Tensor] = None,  # [B, L_text_max]
        text_lengths: Optional[torch.Tensor] = None,  # [B], actual text lengths for RoPE
    ):
        """
        Forward pass with optional text cross-attention.
        
        Args:
            H: Node features [N, d_hidden]
            Z: Node coordinates [N, 3]
            block_id: Block (residue) assignments [N]
            batch_id: Batch assignments [N_block]
            edges: Edge indices [2, E]
            edge_attr: Edge features [E, d_edge]
            topo_edges: Topology (bond) edges [2, E_topo]
            topo_edge_attr: Topology edge features [E_topo, d_edge]
            attn_mask: Attention mask [B, N_max, N_max]
            text_k: Text keys - single tensor or dict mapping layer_idx to tensor
            text_v: Text values - single tensor or dict mapping layer_idx to tensor
            mask_text: Text mask [B, L_text_max]
            text_lengths: Actual text lengths per sample for RoPE positions [B]
            
        Returns:
            H_graph: Updated node features [N, d_hidden]
            V_graph: Updated vector features [N, 1, d_hidden]
        """
        with torch.no_grad():
            if topo_edges is not None:
                not_self_loop = edges[0] != edges[1]
                edges = edges.T[not_self_loop].T
                if edge_attr is not None:
                    edge_attr = edge_attr[not_self_loop]
            (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = _unit_edges_from_block_edges(
                block_id, edges.T, Z, k=self.sparse_k
            )

        if edge_attr is not None:
            edge_attr = edge_attr[block_edge_id]

        # Concat 3D and 2D edges
        if topo_edges is not None:
            unit_row = torch.cat([unit_row, topo_edges[0]], dim=0)
            unit_col = torch.cat([unit_col, topo_edges[1]], dim=0)
        if topo_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, topo_edge_attr], dim=0)

        # Vector init
        Z = Z.view(-1, 3)
        edge_vec = Z[unit_row] - Z[unit_col]
        edge_dis = torch.norm(edge_vec, dim=-1)
        dis_feat = self.rbf(edge_dis)
        edge_feat = torch.cat([H[unit_row], H[unit_col], dis_feat, edge_attr], dim=-1)
        edge_scaler = self.edge_mlp(edge_feat)
        inv_feat, equiv_feat = torch.split(edge_scaler, self.d_hidden, dim=-1)
        edge_scas = H[unit_col] * inv_feat
        edge_vecs = edge_vec.unsqueeze(-1) * equiv_feat.unsqueeze(-2)
        H = self.node_mlp(torch.cat([H, scatter_sum(edge_scas, unit_row, dim_size=H.shape[0], dim=0)], dim=-1))
        V = scatter_mean(edge_vecs, unit_row, dim_size=H.shape[0], dim=0)

        # Convert graph representation to batch format
        batch_to_nodes = batch_id[block_id]
        H_batch, H_mask = graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8)
        bs, max_n = H_batch.shape[0], H_batch.shape[1]
        V_batch = torch.zeros((bs, max_n, *V.shape[1:]), dtype=V.dtype, device=V.device)
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, max_n, *Z.shape[1:]), dtype=Z.dtype, device=Z.device)
        Z_batch[H_mask] = Z

        # Map RBF features to all layers & heads
        if self.use_edge_feat:
            dis_feat = torch.cat([dis_feat, edge_attr], dim=-1)
        rbf_feat = self.rbf_mapping(dis_feat)

        # Compute position indices for batched RBF features
        lengths = torch.zeros(bs, dtype=batch_id.dtype, device=batch_id.device)
        lengths[1:] = torch.cumsum(scatter_sum(torch.ones_like(batch_to_nodes), batch_to_nodes), dim=-1)[:-1]
        lengths = lengths[batch_to_nodes]
        tot_idx = torch.cumsum(torch.ones_like(batch_to_nodes), dim=-1) - 1
        self_idx = tot_idx - lengths

        # Build batched RBF feature tensor with masking
        if self._local_mask:
            rbf_feat_batch = torch.ones(
                (bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device
            ) * float("-inf")
            rbf_feat_batch[~H_mask] = 0.0
        else:
            rbf_feat_batch = torch.zeros(
                (bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device
            )
        if attn_mask is not None:
            rbf_feat_batch[~attn_mask] = float("-inf")
        rbf_feat_batch[batch_to_nodes[unit_row], self_idx[unit_row], self_idx[unit_col]] = rbf_feat
        rbf_feat_batch = rbf_feat_batch.reshape(bs, max_n, max_n, self.n_layers, -1).permute(
            3, 0, 4, 1, 2
        )

        # Compute distance matrix for geometric bias
        D_batch = torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1)
        D_batch = -D_batch
        cached_info = (D_batch.detach(), rbf_feat_batch, H_mask)

        # Apply EPTLayerMoT layers with optional text cross-attention
        for i in range(self.n_layers):
            # Get layer-specific text conditioning if dict format is used
            if isinstance(text_k, dict) and isinstance(text_v, dict):
                layer_text_k = text_k.get(i, None)
                layer_text_v = text_v.get(i, None)
            else:
                layer_text_k = text_k
                layer_text_v = text_v

            H_batch, V_batch = self._modules[f"layer_{i}"](
                H_batch,
                V_batch,
                cached_info,
                text_k=layer_text_k,
                text_v=layer_text_v,
                mask_text=mask_text,
                text_lengths=text_lengths,
            )

        if self.layer_norm == "pre":
            H_batch = self.ln(H_batch)

        # Convert back to graph representation
        H_graph = H_batch[H_mask]
        V_graph = V_batch[H_mask]

        # Normalize and process vector features
        V_graph = V_graph / (V_graph.norm(dim=-2, keepdim=True) + 1e-5)
        _, V_graph = self.final_v(H_graph, V_graph)
        return H_graph, V_graph


class XTransEncoderActMoT(nn.Module):
    """
    MoT-enhanced XTransEncoderAct that wraps TransformerMoT.

    This class provides the same interface as XTransEncoderAct but uses 
    TransformerMoT internally, supporting text_k, text_v, and mask_text 
    parameters for cross-modal attention.
    """

    def __init__(
        self,
        hidden_size,
        ffn_size,
        n_rbf,
        cutoff=7.0,
        z_requires_grad=False,
        edge_size=16,
        n_layers=3,
        n_head=4,
        pre_norm=False,
        use_edge_feat=False,
        sparse_k=3,
        local_mask=False,
        attn_bias=True,
        efficient=False,
        vector_act="none",
        num_kv_groups=4,
    ) -> None:
        super().__init__()

        self.encoder = TransformerMoT(
            d_hidden=hidden_size,
            d_ffn=ffn_size,
            n_heads=n_head,
            n_layers=n_layers,
            n_rbf=n_rbf,
            d_edge=edge_size,
            cutoff=cutoff,
            use_edge_feat=use_edge_feat,
            local_mask=local_mask,
            attn_bias=attn_bias,
            layer_norm="pre" if pre_norm else "post",
            sparse_k=sparse_k,
            efficient=efficient,
            vector_act=vector_act,
            num_kv_groups=num_kv_groups,
        )

    def forward(
        self,
        H,
        Z,
        block_id,
        batch_id,
        edges,
        edge_attr=None,
        topo_edges=None,
        topo_edge_attr=None,
        attn_mask=None,
        text_k: Optional[torch.Tensor] = None,
        text_v: Optional[torch.Tensor] = None,
        mask_text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with optional text cross-attention.

        Args:
            H: Node features [N, hidden_size]
            Z: Coordinates [N, 3]
            block_id: Block assignments [N]
            batch_id: Batch assignments [N_block]
            edges: Edge indices [2, E]
            edge_attr: Edge features [E, edge_size]
            topo_edges: Topology edges [2, E_topo]
            topo_edge_attr: Topology edge features [E_topo, edge_size]
            attn_mask: Attention mask [B, N_max, N_max]
            text_k: Text keys [B, L_text, n_heads, d_head] (optional)
            text_v: Text values [B, L_text, n_heads, d_head] (optional)
            mask_text: Text mask [B, L_text] (optional)
            text_lengths: Actual text lengths per sample for RoPE [B] (optional)

        Returns:
            H: Updated node features [N, hidden_size]
            V: Updated coordinates [N, 3]
        """
        H, V = self.encoder(
            H,
            Z,
            block_id,
            batch_id,
            edges,
            edge_attr,
            topo_edges,
            topo_edge_attr,
            attn_mask,
            text_k=text_k,
            text_v=text_v,
            mask_text=mask_text,
            text_lengths=text_lengths,
        )
        return H, V.reshape(Z.shape) + Z

