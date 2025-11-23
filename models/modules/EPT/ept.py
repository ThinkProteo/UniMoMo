#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_mean, scatter_sum, scatter_std

import utils.register as R

from utils.nn_utils import stable_norm, std_conserve_scatter_sum, graph_to_batch_nx

from ..GET.tools import _unit_edges_from_block_edges
from .radial_basis import RadialBasis

try:
    from xformers.ops import memory_efficient_attention as attn_func
    xformers_enable = True
except:
    xformers_enable = False


class XTransEncoderAct(nn.Module):
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
        # use_ieconv=False, zero_conv=False, efficient_ieconv=False, ieconv_share_edge_feat=False
    ) -> None:
        super().__init__()

        self.encoder = Transformer(
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
        )

    def forward(
        self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None
    ):
        H, V = self.encoder(H, Z, block_id, batch_id, edges, edge_attr, topo_edges, topo_edge_attr, attn_mask)
        block_repr = std_conserve_scatter_sum(H, block_id, dim=0)
        graph_repr = std_conserve_scatter_sum(block_repr, batch_id, dim=0)
        # return H, block_repr, graph_repr, V.reshape(Z.shape) + Z
        return H, V.reshape(Z.shape) + Z


class Transformer(nn.Module):
    """Equivariant Adaptive Block Transformer"""

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
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.use_edge_feat = use_edge_feat
        self.sparse_k = sparse_k
        self.efficient = efficient
        self._local_mask = local_mask
        if self.efficient and not xformers_enable:
            print(
                "xformers are not downloaded, change into custom attention mechanism. "
                "Please install xformers via 'pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121',"
                "or seek 'https://github.com/facebookresearch/xformers' for more details."
            )
            self.efficient = False

        self.edge_mlp = nn.Sequential(
            nn.Linear(d_hidden * 2 + d_edge + n_rbf, d_hidden), act_fn, nn.Linear(d_hidden, d_hidden * 2)
        )

        self.node_mlp = nn.Sequential(nn.Linear(d_hidden * 2, d_hidden), act_fn, nn.Linear(d_hidden, d_hidden))

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

        for i in range(0, n_layers):
            self.add_module(
                f"layer_{i}",
                EPTLayer(
                    d_hidden, d_ffn, n_heads, i, act_fn, layer_norm, residual, self.efficient, vector_act, attn_bias
                ),
            )

    def forward(
        self, H, Z, block_id, batch_id, edges, edge_attr=None, topo_edges=None, topo_edge_attr=None, attn_mask=None
    ):
        with torch.no_grad():
            if topo_edges is not None:
                # first delete self-loop of 3D edges. Otherwise there might be two same atom-level edges overwriting each other
                not_self_loop = edges[0] != edges[1]
                edges = edges.T[not_self_loop].T
                if edge_attr is not None:
                    edge_attr = edge_attr[not_self_loop]
            (unit_row, unit_col), (block_edge_id, unit_edge_src_start, unit_edge_src_id) = _unit_edges_from_block_edges(
                block_id, edges.T, Z, k=self.sparse_k
            )  # [Eu], Eu = \sum_{i, j \in E} n_i * n_j

        if edge_attr is not None:
            edge_attr = edge_attr[block_edge_id]

        # concat 3D and 2D edges
        if topo_edges is not None:
            unit_row = torch.cat([unit_row, topo_edges[0]], dim=0)
            unit_col = torch.cat([unit_col, topo_edges[1]], dim=0)
        if topo_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, topo_edge_attr], dim=0)  # [E1 + E2, d]

        # vector init
        Z = Z.view(-1, 3)
        edge_vec = Z[unit_row] - Z[unit_col]  # [Ne, 3]
        edge_dis = torch.norm(edge_vec, dim=-1)
        dis_feat = self.rbf(edge_dis)
        edge_feat = torch.cat([H[unit_row], H[unit_col], dis_feat, edge_attr], dim=-1)
        edge_scaler = self.edge_mlp(edge_feat)  # [Ne, d_hidden]
        inv_feat, equiv_feat = torch.split(edge_scaler, self.d_hidden, dim=-1)
        edge_scas = H[unit_col] * inv_feat
        edge_vecs = edge_vec.unsqueeze(-1) * equiv_feat.unsqueeze(-2)  # [Ne, 3, d_hidden]
        H = self.node_mlp(torch.cat([H, scatter_sum(edge_scas, unit_row, dim_size=H.shape[0], dim=0)], dim=-1))
        V = scatter_mean(edge_vecs, unit_row, dim_size=H.shape[0], dim=0)

        # graph to batch
        batch_to_nodes = batch_id[block_id]
        H_batch, H_mask = graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8)
        bs, max_n = H_batch.shape[0], H_batch.shape[1]
        V_batch = torch.zeros((bs, max_n, *V.shape[1:]), dtype=V.dtype, device=V.device)
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, max_n, *Z.shape[1:]), dtype=Z.dtype, device=Z.device)
        Z_batch[H_mask] = Z

        # rbf to all layer & heads
        if self.use_edge_feat:
            dis_feat = torch.cat([dis_feat, edge_attr], dim=-1)
        rbf_feat = self.rbf_mapping(dis_feat)
        lengths = torch.zeros(bs, dtype=batch_id.dtype, device=batch_id.device)
        lengths[1:] = torch.cumsum(scatter_sum(torch.ones_like(batch_to_nodes), batch_to_nodes), dim=-1)[:-1]  # [bs]
        lengths = lengths[batch_to_nodes]
        tot_idx = torch.cumsum(torch.ones_like(batch_to_nodes), dim=-1) - 1
        self_idx = tot_idx - lengths
        if self._local_mask:
            rbf_feat_batch = torch.ones(
                (bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device
            ) * float("-inf")
            rbf_feat_batch[
                ~H_mask
            ] = 0.0  # to prevent nan in padding which will lead to 0 * nan = nan (broadcast to other positions)
        else:
            rbf_feat_batch = torch.zeros(
                (bs, max_n, max_n, rbf_feat.shape[-1]), dtype=rbf_feat.dtype, device=rbf_feat.device
            )
        if attn_mask is not None:
            rbf_feat_batch[~attn_mask] = float("-inf")
        rbf_feat_batch[batch_to_nodes[unit_row], self_idx[unit_row], self_idx[unit_col]] = rbf_feat
        rbf_feat_batch = rbf_feat_batch.reshape(bs, max_n, max_n, self.n_layers, -1).permute(
            3, 0, 4, 1, 2
        )  # [l, bs, h, n, n]

        # svd init
        D_batch = torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1)  # [bs, n, n]
        D_batch = -D_batch

        cached_info = (D_batch.detach(), rbf_feat_batch, H_mask)

        for i in range(self.n_layers):
            H_batch, V_batch = self._modules[f"layer_{i}"](H_batch, V_batch, cached_info)

        if self.layer_norm == "pre":
            H_batch = self.ln(H_batch)

        H_graph = H_batch[H_mask]
        V_graph = V_batch[H_mask]

        V_graph = V_graph / (V_graph.norm(dim=-2, keepdim=True) + 1e-5)

        _, V_graph = self.final_v(H_graph, V_graph)
        return H_graph, V_graph


class EPTLayer(nn.Module):
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
    ):
        super(EPTLayer, self).__init__()
        self.attn_layer = SubLayerWrapper(
            SelfAttnLayer(d_hidden, n_heads, layer_idx, efficient, attn_bias=attn_bias), d_hidden, layer_norm, residual
        )
        self.ffn_layer = SubLayerWrapper(
            GVPFFNLayer(d_hidden, d_ffn, act_fn, vector_act=vector_act), d_hidden, layer_norm, residual
        )
        self.layer_idx = layer_idx

    def forward(self, H, V, cached_info=None):
        H, V = self.attn_layer(H, V, cached_info=cached_info)
        H, V = self.ffn_layer(H, V)

        return H, V


class SelfAttnLayer(nn.Module):
    def __init__(self, d_hidden, n_heads, layer_idx=-1, efficient=False, attn_bias=True):
        super(SelfAttnLayer, self).__init__()

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.d_head = self.d_hidden // self.n_heads
        self.layer_idx = layer_idx
        self.factor = 0.5 / math.sqrt(self.d_head)
        self.efficient = efficient
        self.scaler_q = nn.Linear(d_hidden, d_hidden * 4, bias=attn_bias)
        self.scaler_k = nn.Linear(d_hidden, d_hidden * 4, bias=attn_bias)
        self.scaler_v = nn.Linear(d_hidden, d_hidden, bias=attn_bias)
        self.vector_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.scaler_o = nn.Linear(d_hidden, d_hidden)
        self.vector_o = nn.Linear(d_hidden, d_hidden, bias=False)

    def forward(self, H, V, cached_info=None):
        # H : [B, N, d_hidden]
        # V : [B, N, 3, d_hidden]

        batch_size, num_nodes = H.shape[0], H.shape[1]

        D_batch, rbf_feat_batch, H_mask = cached_info

        H_q = self.scaler_q(H).view(batch_size, num_nodes, self.n_heads, -1)
        H_k = self.scaler_k(H).view(batch_size, num_nodes, self.n_heads, -1)
        H_v = self.scaler_v(H).view(batch_size, num_nodes, self.n_heads, -1)
        V_v = self.vector_v(V).view(batch_size, num_nodes, 3, self.n_heads, -1).transpose(-2, -3).flatten(start_dim=-2)
        V_attn = torch.cat([H_v, V_v], dim=-1)

        bias = rbf_feat_batch[self.layer_idx] + D_batch.unsqueeze(1)
        mask = H_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
        bias = bias.masked_fill(mask == 0, float("-inf"))

        if not self.efficient:
            attn = torch.einsum("bnhd, bmhd -> bhnm", H_q, H_k)
            attn = F.softmax(attn * self.factor + bias, dim=-1)
            res = torch.einsum("bhnm, bmhd -> bnhd", attn, V_attn)

        else:
            res = attn_func(query=H_q, key=H_k, value=V_attn, attn_bias=bias.expand(-1, self.n_heads, -1, -1))

        H_res = res[:, :, :, : self.d_head].reshape(batch_size, num_nodes, self.d_hidden)
        V_res = (
            res[:, :, :, self.d_head :]
            .reshape(batch_size, num_nodes, self.n_heads, 3, self.d_head)
            .transpose(-2, -3)
            .reshape(batch_size, num_nodes, 3, self.d_hidden)
        )

        H_o = self.scaler_o(H_res)
        V_o = self.vector_o(V_res)

        return H_o, V_o


class GVPFFNLayer(nn.Module):
    def __init__(self, d_hidden, d_ffn, act_fn=nn.SiLU(), d_output=None, vector_act="none"):
        super(GVPFFNLayer, self).__init__()

        self.d_hidden = d_hidden
        self.d_ffn = d_ffn
        self.act_fn = act_fn
        self.d_output = d_hidden if d_output is None else d_output

        self.linear_v = nn.Linear(d_hidden, d_hidden + self.d_output, bias=False)
        self.ffn_mlp = nn.Sequential(nn.Linear(d_hidden * 2, d_ffn), act_fn, nn.Linear(d_ffn, d_hidden + self.d_output))

        self.vector_act = vector_act
        if self.vector_act == "layernorm":
            self.vector_layernorm = nn.LayerNorm(self.d_output)

    def vector_act_func(self, Vs):
        if self.vector_act == "none":
            return Vs
        elif self.vector_act == "sigmoid":
            return F.sigmoid(Vs)
        elif self.vector_act == "tanh":
            return F.tanh(Vs)
        elif self.vector_act == "layernorm":
            return self.vector_layernorm(Vs)
        elif self.vector_act == "one":
            return torch.ones_like(Vs)

    def forward(self, H, V):
        V_proj = self.linear_v(V)
        V1, V2 = V_proj[..., : self.d_hidden], V_proj[..., self.d_hidden :]
        scaler = torch.cat([H, V1.norm(dim=-2)], dim=-1)
        scaler_out = self.ffn_mlp(scaler)
        H_out, V_update = scaler_out[..., : self.d_hidden], scaler_out[..., self.d_hidden :]
        V_out = self.vector_act_func(V_update).unsqueeze(-2) * V2
        return H_out, V_out


class SubLayerWrapper(nn.Module):
    def __init__(self, sub_layer, d_hidden, layer_norm="pre", residual=True):
        super(SubLayerWrapper, self).__init__()
        self.sub_layer = sub_layer
        self.d_hidden = d_hidden
        self.layer_norm = layer_norm
        self.ln = nn.LayerNorm(d_hidden)
        self.residual = residual

    def forward(self, H, V, **kwargs):
        H0, V0 = H, V
        if self.layer_norm == "pre":
            H = self.ln(H0)
        H, V = self.sub_layer(H, V, **kwargs)
        if self.residual:
            H = H + H0
            V = V + V0
        if self.layer_norm == "post":
            H = self.ln(H)
        return H, V


class EPTLayerMoT(nn.Module):
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
        # Note: Even if text_k/text_v are not provided, defaults to None for backward compatibility
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
    MoT-style EPT attention (inference only).

    VAE branch input (same as original SelfAttnLayer):
        H: [B, N_vae_max, d_hidden]
        V: [B, N_vae_max, 3, d_hidden]

    Text branch provides K/V (scalar part only), computed externally:
        text_k: [B, L_text_max, n_kv_heads, d_attn]   (d_attn = 4 * d_head)
        text_v: [B, L_text_max, n_kv_heads, d_head]
        Note: text has no vector channel input, vector values are filled with 0.

    cached_info (same as original EPT SelfAttnLayer):
        D_batch:         [B, N_vae_max, N_vae_max]
        rbf_feat_batch:  List[Tensor], one per layer [B, n_heads, N_vae_max, N_vae_max]
        H_mask:          [B, N_vae_max], 1=valid, 0=pad

    Original geometric bias:
        bias = rbf_feat_batch[layer_idx] + D_batch.unsqueeze(1)
        bias = bias.masked_fill(H_mask == 0, -inf)

    Here, we only apply bias to the vae→vae sub-block:
        - Q: only from vae tokens
        - K/V: concat([text, vae]) along length dimension
        - bias only applied to K columns from vae, text columns have bias=0

    Returns:
        H_out: [B, N_vae_max, d_hidden]
        V_out: [B, N_vae_max, 3, d_hidden]
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
        qk_norm: bool = True,
        num_kv_groups: int = 4,  # original setting of Qwen3, n_q_head=32, n_kv_head=8
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.layer_idx = layer_idx

        self.num_kv_groups = num_kv_groups
        self.n_kv_heads = n_heads
        self.n_q_heads = self.n_kv_heads * self.num_kv_groups

        self.d_head = d_hidden // self.n_q_heads
        self.d_attn = 4 * self.d_head  # 与原 EPT SelfAttn 的 4 * d_head 对齐
        self.factor = 0.5 / math.sqrt(self.d_head)  # 保留原 EPT 的 scaling 因子风格

        # VAE branch projections: H -> Q,K,H_v; V -> V_v
        self.scaler_q = nn.Linear(d_hidden, self.n_q_heads * self.d_attn, bias=attn_bias)
        self.scaler_k = nn.Linear(d_hidden, self.n_kv_heads * self.d_attn, bias=attn_bias)
        self.scaler_v = nn.Linear(d_hidden, self.n_kv_heads * self.d_head, bias=attn_bias)
        self.vector_v = nn.Linear(d_hidden, self.n_kv_heads * self.d_head, bias=False)

        # Output projections for VAE tokens
        self.scalar_o = nn.Linear(self.n_q_heads * self.d_head, d_hidden)
        self.vector_o = nn.Linear(self.n_q_heads * self.d_head, d_hidden, bias=False)

        # LayerNorm for qkv of vae and text
        if qk_norm:
            self.q_norm_vae = nn.LayerNorm(self.d_attn)
            self.k_norm_vae = nn.LayerNorm(self.d_attn)
            self.k_norm_text = nn.LayerNorm(self.d_attn)
        else:
            self.q_norm_vae = nn.Identity()
            self.k_norm_vae = nn.Identity()
            self.k_norm_text = nn.Identity()

        self.efficient = False

    @staticmethod
    def _resize_head_axis(tensor: torch.Tensor, target_heads: int) -> torch.Tensor:
        """
        Resize the head axis of a tensor with shape [B, L, H, D] to target_heads using 1D interpolation.
        This keeps per-token information while deterministically combining/splitting heads.
        """
        if tensor.ndim != 4:
            raise ValueError(f"Expected tensor with 4 dims [B, L, H, D], got shape {tuple(tensor.shape)}")
        _, seq_len, head_count, _ = tensor.shape
        if head_count == target_heads or seq_len == 0:
            return tensor

        tensor_reshaped = tensor.permute(0, 1, 3, 2).contiguous()  # [B, L, D, H]
        tensor_reshaped = tensor_reshaped.view(-1, 1, head_count)  # treat heads as length dimension
        tensor_resized = F.interpolate(
            tensor_reshaped,
            size=target_heads,
            mode="linear",
            align_corners=False,
        )
        tensor_resized = tensor_resized.view(*tensor.shape[:2], tensor.shape[3], target_heads)
        tensor_resized = tensor_resized.permute(0, 1, 3, 2).contiguous()
        return tensor_resized

    @staticmethod
    def _match_last_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Pad or truncate the last dimension of a tensor to match target_dim.
        """
        current_dim = tensor.shape[-1]
        if current_dim == target_dim or tensor.shape[1] == 0:  # seq_len == 0
            return tensor
        if current_dim > target_dim:
            return tensor[..., :target_dim]
        pad = target_dim - current_dim
        return F.pad(tensor, (0, pad))

    def _align_text_heads_and_dim(self, tensor: torch.Tensor, target_heads: int, target_dim: int) -> torch.Tensor:
        """
        Ensure the incoming text tensor matches the head count and per-head dimensionality expected by EPT.
        """
        if tensor.shape[2] != target_heads:
            tensor = self._resize_head_axis(tensor, target_heads)
        if tensor.shape[-1] != target_dim:
            tensor = self._match_last_dim(tensor, target_dim)
        return tensor.contiguous()

    def forward(
        self,
        H: torch.Tensor,
        V: torch.Tensor,
        cached_info=None,
        text_k: Optional[torch.Tensor] = None,
        text_v: Optional[torch.Tensor] = None,
        mask_text: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass that wraps forward_inference.
        During training, text inputs should be None (backward compatibility).
        """
        return self._forward(H, V, cached_info, text_k, text_v, mask_text)

    def _forward(
        self,
        H: torch.Tensor,  # [B, N_vae_max, d_hidden]
        V: torch.Tensor,  # [B, N_vae_max, 3, d_hidden]
        cached_info,  # (D_batch, rbf_feat_batch, H_mask)
        text_k: Optional[torch.Tensor] = None,  # [B, L_text_max, n_kv_heads, d_attn]
        text_v: Optional[torch.Tensor] = None,  # [B, L_text_max, n_kv_heads, d_head]
        mask_text: Optional[torch.Tensor] = None,  # [B, L_text_max], 1=valid
    ):
        B, N_vae_max, _ = H.shape
        device = H.device

        D_batch, rbf_feat_batch, H_mask = cached_info  # H_mask: [B, N_vae_max]

        # Handle text inputs (VAE-only mode if text_k/text_v are None)
        if text_k is None or text_v is None:
            L_text_max = 0
            text_k = H.new_zeros(B, 0, self.n_kv_heads, self.d_attn)
            text_v = H.new_zeros(B, 0, self.n_kv_heads, self.d_head)
            mask_text = H.new_zeros(B, 0, dtype=torch.bool)
        else:
            assert text_k.shape[0] == B, "batch size mismatch for text_k"
            L_text_max = text_k.shape[1]
            if mask_text is None:
                mask_text = torch.ones(B, L_text_max, dtype=torch.bool, device=device)

            # Align text head count & dimensions with EPTMoT expectations
            text_k = self._align_text_heads_and_dim(text_k, self.n_kv_heads, self.d_attn)
            text_v = self._align_text_heads_and_dim(text_v, self.n_kv_heads, self.d_head)

        # Compute VAE Q/K/V projections
        H_q_vae_all = self.scaler_q(H).view(B, N_vae_max, self.n_q_heads, self.d_attn)  # [B, N, n_q_heads, d_attn]
        H_k_vae_all = self.scaler_k(H).view(B, N_vae_max, self.n_kv_heads, self.d_attn)  # [B, N, n_kv_heads, d_attn]
        H_v_vae_all = self.scaler_v(H).view(B, N_vae_max, self.n_kv_heads, self.d_head)  # [B, N, n_kv_heads, d_head]

        # Process vector features and concat with scalar values
        V_v_vae_all = self.vector_v(V).view(B, N_vae_max, 3, self.n_kv_heads, self.d_head)
        V_v_vae_all = V_v_vae_all.transpose(-2, -3).flatten(start_dim=-2)  # [B, N, n_kv_heads, 3*d_head]
        V_attn_vae_all = torch.cat([H_v_vae_all, V_v_vae_all], dim=-1)  # [B, N, n_kv_heads, 4*d_head]

        # Apply normalization to text keys
        k_text_all = self.k_norm_text(text_k)  # [B, L_text_max, n_kv_heads, d_attn]

        # Text has no vector channel, fill with zeros
        V_v_text_all = torch.zeros(B, L_text_max, self.n_kv_heads, 3 * self.d_head, device=device, dtype=H.dtype)
        V_attn_text_all = torch.cat([text_v, V_v_text_all], dim=-1)  # [B, L_text_max, n_kv_heads, 4*d_head]

        # Initialize output tensors
        H_vae_out = H.new_zeros(B, N_vae_max, self.d_hidden)  # [B, N, d_hidden]
        V_vae_out = V.new_zeros(B, N_vae_max, 3, self.d_hidden)  # [B, N, 3, d_hidden]

        # Process each batch element separately
        for b in range(B):
            # Get valid token indices
            text_valid_idx = mask_text[b].nonzero(as_tuple=False).squeeze(-1)  # [L_text_b]
            vae_valid_idx = H_mask[b].nonzero(as_tuple=False).squeeze(-1)  # [L_vae_b]

            L_text_b = int(text_valid_idx.numel())
            L_vae_b = int(vae_valid_idx.numel())
            if L_vae_b == 0:
                continue  # Skip if no VAE tokens

            # Extract VAE Q/K/V for valid tokens
            H_q_vae = H_q_vae_all[b, vae_valid_idx]  # [L_vae_b, n_q_heads, d_attn]
            H_k_vae = H_k_vae_all[b, vae_valid_idx]  # [L_vae_b, n_kv_heads, d_attn]
            H_v_vae = H_v_vae_all[b, vae_valid_idx]  # [L_vae_b, n_kv_heads, d_head]
            V_attn_vae = V_attn_vae_all[b, vae_valid_idx]  # [L_vae_b, n_kv_heads, 4*d_head]

            # Apply normalization
            q_vae = self.q_norm_vae(H_q_vae)  # [L_vae_b, n_q_heads, d_attn]
            k_vae = self.k_norm_vae(H_k_vae)  # [L_vae_b, n_kv_heads, d_attn]

            # Extract text K/V for valid tokens
            if L_text_b > 0:
                k_text = k_text_all[b, text_valid_idx]  # [L_text_b, n_kv_heads, d_attn]
                V_attn_text = V_attn_text_all[b, text_valid_idx]  # [L_text_b, n_kv_heads, 4*d_head]
            else:
                k_text = k_text_all.new_zeros(0, self.n_kv_heads, self.d_attn)
                V_attn_text = V_attn_text_all.new_zeros(0, self.n_kv_heads, 4 * self.d_head)

            # Concatenate text and VAE K/V
            K_cat = torch.cat([k_text, k_vae], dim=0)  # [L_k, n_kv_heads, d_attn]
            V_cat = torch.cat([V_attn_text, V_attn_vae], dim=0)  # [L_k, n_kv_heads, 4*d_head]
            L_k = L_text_b + L_vae_b

            # Reshape for multi-head attention: [1, n_heads, L, d]
            q_b = q_vae.permute(1, 0, 2).unsqueeze(0)  # [1, n_q_heads, L_vae_b, d_attn]
            k_b = K_cat.permute(1, 0, 2).unsqueeze(0)  # [1, n_kv_heads, L_k, d_attn]
            v_b = V_cat.permute(1, 0, 2).unsqueeze(0)  # [1, n_kv_heads, L_k, 4*d_head]

            # Expand K/V for grouped query attention
            k_b_exp = k_b.repeat_interleave(self.num_kv_groups, dim=1)  # [1, n_q_heads, L_k, d_attn]
            v_b_exp = v_b.repeat_interleave(self.num_kv_groups, dim=1)  # [1, n_q_heads, L_k, 4*d_head]

            # Compute EPT geometric bias from RBF features
            bias_full = rbf_feat_batch[self.layer_idx][b] + D_batch[b].unsqueeze(0)  # [h_geom, N, N]
            mask_b = H_mask[b].unsqueeze(0).unsqueeze(0)  # [1, 1, N]
            bias_full = bias_full.masked_fill(mask_b == 0, float("-inf"))

            # Extract VAE-VAE bias and average over geometric heads if needed
            bias_vae_local = bias_full[:, vae_valid_idx][:, :, vae_valid_idx]  # [h_geom, L_vae_b, L_vae_b]
            h_geom = bias_vae_local.shape[0]
            if h_geom > 1:
                bias_vae_scalar = bias_vae_local.mean(dim=0)  # [L_vae_b, L_vae_b]
            else:
                bias_vae_scalar = bias_vae_local[0]  # [L_vae_b, L_vae_b]

            # Construct total bias: apply geometric bias only to VAE tokens, not text
            bias_total = q_b.new_zeros(1, self.n_q_heads, L_vae_b, L_k)  # [1, n_q_heads, L_vae_b, L_k]
            if L_vae_b > 0:
                # First L_text_b columns are text (no geometric bias), last L_vae_b columns are VAE
                bias_total[:, :, :, L_text_b:] = bias_vae_scalar.unsqueeze(0).unsqueeze(0)

            # Compute attention scores with scaling and bias
            attn_scores = torch.einsum("bhqd,bhkd -> bhqk", q_b, k_b_exp)  # [1, n_q_heads, L_vae_b, L_k]
            attn = attn_scores * (0.5 / math.sqrt(self.d_head)) + bias_total
            attn = F.softmax(attn, dim=-1)  # [1, n_q_heads, L_vae_b, L_k]
            res = torch.einsum("bhqk,bhkd -> bhqd", attn, v_b_exp)  # [1, n_q_heads, L_vae_b, 4*d_head]
            res = res.squeeze(0).permute(1, 0, 2)  # [L_vae_b, n_q_heads, 4*d_head]

            # Split result into scalar and vector parts
            H_res = res[:, :, : self.d_head].reshape(L_vae_b, self.n_q_heads * self.d_head)  # [L_vae_b, d_hidden]
            V_part = res[:, :, self.d_head:]  # [L_vae_b, n_q_heads, 3*d_head]
            V_part = V_part.view(L_vae_b, self.n_q_heads, 3, self.d_head)
            V_part = V_part.permute(0, 2, 1, 3).reshape(L_vae_b, 3, self.n_q_heads * self.d_head)
            V_res = V_part  # [L_vae_b, 3, n_q_heads * d_head]

            # Apply output projections and scatter back to padded tensor
            H_proj = self.scalar_o(H_res)  # [L_vae_b, d_hidden]
            V_proj = self.vector_o(V_res)  # [L_vae_b, 3, d_hidden]
            H_vae_out[b, vae_valid_idx] = H_proj
            V_vae_out[b, vae_valid_idx] = V_proj

        return H_vae_out, V_vae_out


class XTransEncoderActMoT(nn.Module):
    """
    MoT-enhanced XTransEncoderAct that wraps TransformerMoT.

    This class provides the same interface as XTransEncoderAct but uses TransformerMoT internally,
    which supports text_k, text_v, and mask_text parameters for cross-modal attention.
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
            text_k: Text keys [B, L_text, n_heads, d_attn] (optional)
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


class TransformerMoT(nn.Module):
    """Equivariant Adaptive Block Transformer with MoT-style EPT attention."""

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
    ):
        super().__init__()

        self.d_hidden = d_hidden
        self.n_layers = n_layers
        self.layer_norm = layer_norm
        self.use_edge_feat = use_edge_feat
        self.sparse_k = sparse_k
        self.efficient = efficient
        self._local_mask = local_mask
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

        # Use EPTLayerMoT instead of original EPTLayer to support cross-modal attention
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
        text_k: Optional[torch.Tensor] = None,  # [B, L_text_max, n_kv_heads, d_attn] or Dict[int, Tensor]
        text_v: Optional[torch.Tensor] = None,  # [B, L_text_max, n_kv_heads, d_head] or Dict[int, Tensor]
        mask_text: Optional[torch.Tensor] = None,  # [B, L_text_max]
    ):
        # Same as original Transformer.forward, but passes text_k/text_v/mask_text to layers

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

        # concat 3D and 2D edges
        if topo_edges is not None:
            unit_row = torch.cat([unit_row, topo_edges[0]], dim=0)
            unit_col = torch.cat([unit_col, topo_edges[1]], dim=0)
        if topo_edge_attr is not None:
            edge_attr = torch.cat([edge_attr, topo_edge_attr], dim=0)

        # vector init
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
        V = scatter_mean(edge_vecs, unit_row, dim_size=H.shape[0], dim=0)  # [N, 3, d]

        # Convert graph representation to batch format
        batch_to_nodes = batch_id[block_id]  # [N]
        H_batch, H_mask = graph_to_batch_nx(H, batch_to_nodes, mask_is_pad=False, factor_req=8)  # [B, N_max, d], [B, N_max]
        bs, max_n = H_batch.shape[0], H_batch.shape[1]
        V_batch = torch.zeros((bs, max_n, *V.shape[1:]), dtype=V.dtype, device=V.device)  # [B, N_max, 3, d]
        V_batch[H_mask] = V
        Z_batch = torch.zeros((bs, max_n, *Z.shape[1:]), dtype=Z.dtype, device=Z.device)  # [B, N_max, 3]
        Z_batch[H_mask] = Z

        # Map RBF features to all layers & heads
        if self.use_edge_feat:
            dis_feat = torch.cat([dis_feat, edge_attr], dim=-1)
        rbf_feat = self.rbf_mapping(dis_feat)  # [Ne, n_layers]

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
        )  # [n_layers, B, n_heads, N_max, N_max]

        # Compute distance matrix for geometric bias
        D_batch = torch.norm(Z_batch.unsqueeze(1) - Z_batch.unsqueeze(2), dim=-1)  # [B, N_max, N_max]
        D_batch = -D_batch
        cached_info = (D_batch.detach(), rbf_feat_batch, H_mask)

        # Apply EPTLayerMoT layers with optional text cross-attention
        # Support both single tensor and per-layer dict formats for text_k/text_v
        for i in range(self.n_layers):
            # Get layer-specific text conditioning if dict format is used
            if isinstance(text_k, dict) and isinstance(text_v, dict):
                # Layer-specific conditioning
                layer_text_k = text_k.get(i, None)
                layer_text_v = text_v.get(i, None)
            else:
                # Shared conditioning across all layers
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
        H_graph = H_batch[H_mask]  # [N, d_hidden]
        V_graph = V_batch[H_mask]  # [N, 3, d_hidden]

        # Normalize and process vector features
        V_graph = V_graph / (V_graph.norm(dim=-2, keepdim=True) + 1e-5)
        _, V_graph = self.final_v(H_graph, V_graph)
        return H_graph, V_graph  # [N, d_hidden], [N, 1, d_hidden]