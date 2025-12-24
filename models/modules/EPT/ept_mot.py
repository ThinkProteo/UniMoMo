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
        qk_norm: bool = False,
        num_kv_groups: int = 4,
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
        self.text_k_proj = nn.Linear(self.d_head, self.d_qk_head, bias=False)

        # Output projections
        self.scaler_o = nn.Linear(self.n_q_heads * self.d_head, d_hidden)
        self.vector_o = nn.Linear(self.n_q_heads * self.d_head, d_hidden, bias=False)

        # Optional Q/K LayerNorm (applied on expanded 4x dimension)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.d_qk_head)
            self.k_norm = nn.LayerNorm(self.d_qk_head)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

    def forward(
        self,
        H: torch.Tensor,  # [B, N_vae_max, d_hidden]
        V: torch.Tensor,  # [B, N_vae_max, 3, d_hidden]
        cached_info,      # (D_batch, rbf_feat_batch, H_mask)
        text_k: Optional[torch.Tensor] = None,  # [B, L_text_max, n_kv_heads, d_head]
        text_v: Optional[torch.Tensor] = None,  # [B, L_text_max, n_kv_heads, d_head]
        mask_text: Optional[torch.Tensor] = None,  # [B, L_text_max], 1=valid
    ):
        B, N_vae_max, _ = H.shape
        device = H.device

        D_batch, rbf_feat_batch, H_mask = cached_info

        # Handle text inputs
        if text_k is None or text_v is None:
            L_text_max = 0
            text_k_proj = H.new_zeros(B, 0, self.n_kv_heads, self.d_qk_head)
            text_v = H.new_zeros(B, 0, self.n_kv_heads, self.d_head)
            mask_text = H.new_ones(B, 0, dtype=torch.bool)
        else:
            assert text_k.shape[0] == B, "Batch size mismatch"
            L_text_max = text_k.shape[1]
            
            # Project Text Keys: d_head -> 4*d_head (d_qk_head)
            text_k_proj = self.text_k_proj(text_k)
            
            if mask_text is None:
                mask_text = torch.ones(B, L_text_max, dtype=torch.bool, device=device)

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

        # Text Values: pad vector part with zeros (text has no 3D vector channel)
        V_v_text_zeros = torch.zeros(B, L_text_max, self.n_kv_heads, 3 * self.d_head, device=device, dtype=H.dtype)
        V_attn_text = torch.cat([text_v, V_v_text_zeros], dim=-1)

        # Apply Q/K normalization
        H_q_vae = self.q_norm(H_q_vae)
        H_k_vae = self.k_norm(H_k_vae)
        text_k_proj = self.k_norm(text_k_proj)

        # ========== BATCHED ATTENTION ==========

        # Concatenate text and VAE K/V along sequence dimension
        K_full = torch.cat([text_k_proj, H_k_vae], dim=1)  # [B, L_total, n_kv, d_qk_head]
        V_full = torch.cat([V_attn_text, V_attn_vae], dim=1)  # [B, L_total, n_kv, 4*d_head]
        L_total = L_text_max + N_vae_max

        # Reshape for multi-head attention: [B, n_heads, L, d]
        q = H_q_vae.transpose(1, 2)  # [B, n_q, N_vae, d_qk_head]
        k = K_full.transpose(1, 2)   # [B, n_kv, L_total, d_qk_head]
        v = V_full.transpose(1, 2)   # [B, n_kv, L_total, 4*d_head]

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
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)  # [B, n_q, N_vae, 4*d_head]

        # Reshape output: [B, n_q, N_vae, 4*d_head] -> [B, N_vae, n_q, 4*d_head]
        out = out.transpose(1, 2)

        # Split scalar and vector parts
        H_out = out[..., :self.d_head].reshape(B, N_vae_max, self.n_q_heads * self.d_head)
        
        V_out_flat = out[..., self.d_head:].reshape(B, N_vae_max, self.n_q_heads, 3, self.d_head)
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
        )
        block_repr = std_conserve_scatter_sum(H, block_id, dim=0)
        graph_repr = std_conserve_scatter_sum(block_repr, batch_id, dim=0)
        return H, V.reshape(Z.shape) + Z

