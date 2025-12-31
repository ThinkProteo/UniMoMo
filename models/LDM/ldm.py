#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch_scatter import scatter_mean

# Disable TF32 to avoid CUBLAS errors on H100/H200 GPUs
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from data.bioparse import VOCAB

import utils.register as R
from utils.oom_decorator import oom_decorator
from utils.nn_utils import SinusoidalPositionEmbedding
from utils.gnn_utils import length_to_batch_id, std_conserve_scatter_mean

from .diffusion.dpm_full import FullDPM
from ..IterVAE.model import CondIterAutoEncoder
from ..modules.nn import GINEConv, MLP


@R.register('LDMMolDesign')
class LDMMolDesign(nn.Module):

    def __init__(
            self,
            autoencoder_ckpt,
            latent_deterministic,
            hidden_size,
            num_steps,
            h_loss_weight=None,
            std=10.0,
            is_aa_corrupt_ratio=0.1,
            diffusion_opt={},
            text_injection_mode=False,  # NEW: Direct text injection instead of attention
            text_embed_dim=2560,  # Qwen3-4B hidden_size for per-residue embeddings
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic
        self.text_injection_mode = text_injection_mode

        self.autoencoder: CondIterAutoEncoder = torch.load(
            autoencoder_ckpt, map_location='cpu', weights_only=False
        )
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()

        latent_size = self.autoencoder.latent_size
        self.hidden_size = hidden_size  # Save for text projection
        
        # Text injection: project per-residue text embedding to cond_embedding space
        # text_v shape: [B, L_text, hidden_dim] = [B, L, 2560] (Qwen hidden states)
        # Each token = one amino acid (1:1 mapping)
        # Project: [B, L, 2560] -> [B, L, hidden_size] (to match cond_embedding)
        if text_injection_mode:
            self.text_proj = nn.Sequential(
                nn.Linear(text_embed_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),  # Output to hidden_size for cond_embedding
            )
            print(f"📌 TEXT INJECTION MODE: Projecting text ({text_embed_dim}) → cond_embedding ({hidden_size})")

        # topo embedding
        self.bond_embed = nn.Embedding(5, hidden_size) # [None, single, double, triple, aromatic]
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size) # is or is not standard amino acid

        # condition embedding MLP
        self.cond_mlp = MLP(
            input_size=3 * hidden_size, # [position, topo, is_aa]
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        self.diffusion = FullDPM(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_steps=num_steps,
            **diffusion_opt
        )
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size  # make loss_X and loss_H about the same size
        else:
            self.h_loss_weight = h_loss_weight
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))
        self.is_aa_corrupt_ratio = is_aa_corrupt_ratio

    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            text_k=None,    # Optional: [B, L_text, n_kv_heads, d_head] text key features
            text_v=None,    # Optional: [B, L_text, n_kv_heads, d_head] text value features
            mask_text=None, # Optional: [B, L_text] text attention mask
            text_lengths=None,  # Optional: [B] actual text lengths for RoPE
            t=None,         # Optional: fixed timestep for debugging/overfitting
        ):
        '''
            Optional text conditioning via text_k, text_v, mask_text, text_lengths.
            When None, model behaves exactly as original UniMoMo.
            
            If text_injection_mode=True:
            - Pools text_v, projects to latent space, adds directly to H_0
            - No attention to text (text_k, text_v not passed to diffusion)
        '''

        # encode latent_H_0 (N*d) and latent_X_0 (N*3)
        with torch.no_grad():
            self.autoencoder.eval()
            # encoding
            Zh, Zx, _, _, _, _, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
            ) # [Nblock, d_latent], [Nblock, 3]

        position_embedding = self.position_encoding(position_ids)

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)

        # is aa embedding (sample 50% for generation part)
        corrupt_mask = generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))

        # TEXT INJECTION MODE: Add text embedding to cond_embedding (conditioning signal)
        # Per-residue embeddings with 1:1 token-residue mapping!
        # This conditions the denoising process without modifying the target H_0
        if self.text_injection_mode and text_v is not None and mask_text is not None:
            # text_v: [B, L_text, hidden_dim] - Qwen hidden states (per-residue)
            # Each token = one amino acid residue
            if text_v.dim() == 4:
                # Old format: [B, L_text, n_heads, head_dim] -> flatten
                B, L_text, n_heads, head_dim = text_v.shape
                text_v_flat = text_v.view(B, L_text, n_heads * head_dim)
            else:
                # New format: [B, L_text, hidden_dim] - direct hidden states
                B, L_text, text_hidden_dim = text_v.shape
                text_v_flat = text_v
            
            batch_size = lengths.shape[0]
            
            # Project each token to cond_embedding space: [B, L_text, hidden_size]
            # Cast to same dtype as projection layer (Qwen outputs bfloat16, proj is float32)
            proj_dtype = next(self.text_proj.parameters()).dtype
            text_v_flat = text_v_flat.to(dtype=proj_dtype)
            text_cond = self.text_proj(text_v_flat)  # [B, L_text, hidden_size]
            
            # Map tokens to residues 1:1
            # cond_embedding: [Nblock, hidden_size] where Nblock = sum(lengths)
            
            # DEBUG: Print text injection stats
            _debug_injection = getattr(self, '_debug_text_injection', False)
            if _debug_injection:
                print(f"\n🔍 TEXT INJECTION DEBUG - LDM (cond_embedding mode):")
                print(f"  text_v shape: {text_v.shape}, text_cond shape: {text_cond.shape}")
                print(f"  text_cond stats: mean={text_cond.mean():.4f}, std={text_cond.std():.4f}")
                print(f"  cond_embedding (before) stats: mean={cond_embedding.mean():.4f}, std={cond_embedding.std():.4f}")
            
            offset = 0
            for sample_idx in range(batch_size):
                sample_len = int(lengths[sample_idx].item())
                sample_mask = generate_mask[offset:offset + sample_len]  # [sample_len]
                
                # Count CDR residues in this sample
                n_cdr = int(sample_mask.sum().item())
                
                # Get valid text tokens for this sample
                if text_lengths is not None:
                    n_tokens = int(text_lengths[sample_idx].item())
                elif mask_text is not None:
                    n_tokens = int(mask_text[sample_idx].sum().item())
                else:
                    n_tokens = L_text
                
                # 1:1 mapping: tokens should match CDR residues
                n_to_add = min(n_cdr, n_tokens)
                
                if _debug_injection and sample_idx < 2:
                    print(f"  Sample {sample_idx}: n_cdr={n_cdr}, n_tokens={n_tokens}, n_to_add={n_to_add}")
                
                if n_to_add > 0:
                    # Get CDR positions in this sample
                    cdr_positions = sample_mask.nonzero(as_tuple=True)[0][:n_to_add]
                    
                    # Get text embeddings for this sample
                    text_embed = text_cond[sample_idx, :n_to_add]  # [n_to_add, hidden_size]
                    
                    # Add to cond_embedding at CDR positions (conditioning signal)
                    global_positions = offset + cdr_positions
                    cond_embedding[global_positions] = cond_embedding[global_positions] + text_embed
                
                offset += sample_len
            
            if _debug_injection:
                print(f"  cond_embedding (after) stats: mean={cond_embedding.mean():.4f}, std={cond_embedding.std():.4f}")
            
            # Disable attention to text (we're using direct injection)
            text_k, text_v, mask_text, text_lengths = None, None, None, None

        loss_dict = self.diffusion.forward(
            H_0=Zh,
            X_0=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            text_k=text_k,
            text_v=text_v,
            mask_text=mask_text,
            text_lengths=text_lengths,
            t=t,
        )

        # loss - RESTORED: Original UniMoMo formula with h_loss_weight
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']

        return loss_dict

    # def latent_geometry_guidance(self, X, generate_mask, batch_ids, tolerance=3, **kwargs):
    #     assert self.consec_dist_mean is not None and self.consec_dist_std is not None, \
    #            'Please run set_consec_dist(self, mean, std) to setup guidance parameters'
    #     return dist_energy(
    #         X, generate_mask, batch_ids,
    #         self.consec_dist_mean, self.consec_dist_std,
    #         tolerance=tolerance, **kwargs
    #     )

    def topo_embedding(self, A, bonds, block_ids, generate_mask):
        ctx_mask = ~generate_mask[block_ids]

        # only retain bonds in the context
        bond_select_mask = ctx_mask[bonds[:, 0]] & ctx_mask[bonds[:, 1]]
        bonds = bonds[bond_select_mask]

        # embed bond type
        edge_attr = self.bond_embed(bonds[:, 2])
        
        # embed atom type
        H = self.atom_embed(A)

        # get topo embedding
        topo_embedding = self.topo_gnn(H, bonds[:, :2].T, edge_attr) # [Natom]

        # aggregate to each block
        topo_embedding = std_conserve_scatter_mean(topo_embedding, block_ids, dim=0) # [Nblock]

        # set generation part to zero
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding

    def _normalize_position(self, X, batch_ids, center_mask):
        # TODO: pass in centers from dataset, which might be better for antibody (custom center)
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1) # [bs, 3]
        centers = centers[batch_ids] # [N, 3]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        X = X_norm * self.std + centers
        return X

    @torch.no_grad()
    def sample(
            self,
            X,              # [Natom, 3], atom coordinates     
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds, src-dst-type (single: 1, double: 2, triple: 3)
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for calculating complex mass center
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid (for determining the X_mask in inverse folding)
            text_k=None,    # Optional: [B, L_text, n_kv_heads, d_head] text key features
            text_v=None,    # Optional: [B, L_text, n_kv_heads, d_head] text value features
            mask_text=None, # Optional: [B, L_text] text attention mask
            text_lengths=None,  # Optional: [B] actual text lengths for RoPE
            sample_opt={
                'pbar': False,
                # 'energy_func': None,
                # 'energy_lambda': 0.0,
            },
            return_tensor=False,
        ):
        '''
            Sample from the diffusion model with optional text conditioning.
            When text_k, text_v, mask_text, text_lengths are provided, the generation is conditioned on text embeddings.
        '''

        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)

        block_ids = length_to_batch_id(block_lengths)

        # ensure there is no data leakage
        S[generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[generate_mask[block_ids]] = 0
        ctx_atom_mask = ~generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # encoding context
        self.autoencoder.eval()
        Zh, Zx, _, signed_Zx_log_var, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths, deterministic=self.latent_deterministic
        ) # [Nblock, d_latent], [Nblock, 3]

        # if 'energy_func' in sample_opt:
        #     if sample_opt['energy_func'] is None:
        #         pass
        #     elif sample_opt['energy_func'] == 'default':
        #         sample_opt['energy_func'] = self.latent_geometry_guidance
        #     # otherwise this should be a function
        

        # normalize
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # topo embedding for structure prediction
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)
        
        # position embedding
        position_embedding = self.position_encoding(position_ids)

        # is aa embedding
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        
        # condition embedding
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))
        
        # TEXT INJECTION MODE: Add text embedding to cond_embedding during sampling
        if self.text_injection_mode and text_v is not None and mask_text is not None:
            if text_v.dim() == 4:
                B, L_text, n_heads, head_dim = text_v.shape
                text_v_flat = text_v.view(B, L_text, n_heads * head_dim)
            else:
                B, L_text, text_hidden_dim = text_v.shape
                text_v_flat = text_v
            
            batch_size = lengths.shape[0]
            proj_dtype = next(self.text_proj.parameters()).dtype
            text_v_flat = text_v_flat.to(dtype=proj_dtype)
            text_cond = self.text_proj(text_v_flat)  # [B, L_text, hidden_size]
            
            offset = 0
            for sample_idx in range(batch_size):
                sample_len = int(lengths[sample_idx].item())
                sample_mask = generate_mask[offset:offset + sample_len]
                n_cdr = int(sample_mask.sum().item())
                
                if text_lengths is not None:
                    n_tokens = int(text_lengths[sample_idx].item())
                elif mask_text is not None:
                    n_tokens = int(mask_text[sample_idx].sum().item())
                else:
                    n_tokens = L_text
                
                n_to_add = min(n_cdr, n_tokens)
                
                if n_to_add > 0:
                    cdr_positions = sample_mask.nonzero(as_tuple=True)[0][:n_to_add]
                    text_embed = text_cond[sample_idx, :n_to_add]
                    global_positions = offset + cdr_positions
                    cond_embedding[global_positions] = cond_embedding[global_positions] + text_embed
                
                offset += sample_len
            
            # In text injection mode, we don't use attention
            text_k, text_v, mask_text, text_lengths = None, None, None, None
        
        traj = self.diffusion.sample(
            H=Zh,
            X=Zx,
            cond_embedding=cond_embedding,
            chain_ids=chain_ids,
            generate_mask=generate_mask,
            lengths=lengths,
            text_k=text_k,
            text_v=text_v,
            mask_text=mask_text,
            text_lengths=text_lengths,
            **sample_opt
        )
        X_0, H_0 = traj[0]
        X_0 = torch.where(generate_mask[:, None].expand_as(X_0), X_0, Zx)
        H_0 = torch.where(generate_mask[:, None].expand_as(H_0), H_0, Zh)

        # unnormalize
        X_0 = self._unnormalize_position(X_0, centers, batch_ids)

        # autodecoder decode
        return self.autoencoder.generate(
            X=X, S=S, A=A, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_0, X_0, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=generate_mask
        )