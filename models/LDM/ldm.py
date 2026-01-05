#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
LDM (Latent Diffusion Model) for Protein Structure Generation.

This module implements the LDMMolDesign class which combines:
- VAE encoder for structure latent representation
- Diffusion model for structure generation
- Sequence conditioners for text/AA/ESM-based conditioning

Conditioning modes (mutually exclusive):
1. text_injection_mode: Qwen per-residue embeddings
2. use_learned_aa_embed: Learned AA embeddings with position
3. use_esm_embed: ESM-2 protein language model embeddings
4. Default: Qwen QKV attention conditioning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Import conditioners
from .conditioners import QwenTextConditioner, LearnedAAConditioner, ESMConditioner, AnswerSeqConditioner


@R.register('LDMMolDesign')
class LDMMolDesign(nn.Module):
    """
    Latent Diffusion Model for Molecular Design.
    
    Combines VAE encoding with diffusion-based structure generation,
    optionally conditioned on sequence information via pluggable conditioners.
    """

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
            # Conditioning options (mutually exclusive)
            text_injection_mode=False,
            text_embed_dim=2560,
            use_learned_aa_embed=False,
            use_esm_embed=False,
            esm_model_name="esm2_t33_650M_UR50D",
            use_answer_seq_und=False,  # Use answer_sequence hidden states from Qwen
            no_context_attention=False,  # ABLATION: answer_seq tokens don't attend to context
            # Auxiliary loss weight (0 to disable)
            aux_loss_weight=1.0,
            # Debug options
            debug_conditioning=True,
        ):
        super().__init__()
        self.latent_deterministic = latent_deterministic
        self.text_injection_mode = text_injection_mode
        self.use_learned_aa_embed = use_learned_aa_embed
        self.use_esm_embed = use_esm_embed
        self.use_answer_seq_und = use_answer_seq_und
        self.no_context_attention = no_context_attention

        # Load frozen VAE
        self.autoencoder: CondIterAutoEncoder = torch.load(
            autoencoder_ckpt, map_location='cpu', weights_only=False
        )
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.eval()

        latent_size = self.autoencoder.latent_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        
        # ========== CONDITIONER SETUP ==========
        # Create appropriate conditioner based on config
        self.conditioner = None
        self._setup_conditioner(
            text_injection_mode=text_injection_mode,
            text_embed_dim=text_embed_dim,
            use_learned_aa_embed=use_learned_aa_embed,
            use_esm_embed=use_esm_embed,
            esm_model_name=esm_model_name,
            use_answer_seq_und=use_answer_seq_und,
            hidden_size=hidden_size,
            latent_size=latent_size,
            aux_loss_weight=aux_loss_weight,
            debug=debug_conditioning,
        )

        # ========== TOPOLOGY EMBEDDINGS ==========
        self.bond_embed = nn.Embedding(5, hidden_size)  # [None, single, double, triple, aromatic]
        self.atom_embed = nn.Embedding(VOCAB.get_num_atom_type(), hidden_size)
        self.topo_gnn = GINEConv(hidden_size, hidden_size, hidden_size, hidden_size)

        self.position_encoding = SinusoidalPositionEmbedding(hidden_size)
        self.is_aa_embed = nn.Embedding(2, hidden_size)  # is or is not standard amino acid

        # Condition embedding MLP
        self.cond_mlp = MLP(
            input_size=3 * hidden_size,  # [position, topo, is_aa]
            hidden_size=hidden_size,
            output_size=hidden_size,
            n_layers=3,
            dropout=0.1
        )

        # ========== DIFFUSION MODEL ==========
        self.diffusion = FullDPM(
            latent_size=latent_size,
            hidden_size=hidden_size,
            num_steps=num_steps,
            **diffusion_opt
        )
        
        if h_loss_weight is None:
            self.h_loss_weight = 3 / latent_size
        else:
            self.h_loss_weight = h_loss_weight
        self.register_buffer('std', torch.tensor(std, dtype=torch.float))
        self.is_aa_corrupt_ratio = is_aa_corrupt_ratio

    def _setup_conditioner(
        self,
        text_injection_mode: bool,
        text_embed_dim: int,
        use_learned_aa_embed: bool,
        use_esm_embed: bool,
        esm_model_name: str,
        use_answer_seq_und: bool,
        hidden_size: int,
        latent_size: int,
        aux_loss_weight: float,
        debug: bool,
    ):
        """Setup the appropriate conditioner based on config."""
        if use_esm_embed:
            self.conditioner = ESMConditioner(
                hidden_size=hidden_size,
                latent_size=latent_size,
                esm_model_name=esm_model_name,
                aux_loss_weight=aux_loss_weight,
                debug=debug,
            )
            # Store reference for ESM extraction convenience
            self._esm_conditioner = self.conditioner
        elif use_answer_seq_und:
            # Use hidden states from answer_sequence tokens in Qwen output
            self.conditioner = AnswerSeqConditioner(
                hidden_size=hidden_size,
                latent_size=latent_size,
                text_embed_dim=text_embed_dim,
                aux_loss_weight=aux_loss_weight,
                debug=debug,
            )
            print("📌 AnswerSeqConditioner: Uses hidden states from answer_sequence tokens")
            print("   ⚠️ Text K/V attention in diffusion will be DISABLED")
        elif use_learned_aa_embed:
            self.conditioner = LearnedAAConditioner(
                hidden_size=hidden_size,
                latent_size=latent_size,
                aux_loss_weight=aux_loss_weight,
                debug=debug,
            )
        elif text_injection_mode:
            self.conditioner = QwenTextConditioner(
                hidden_size=hidden_size,
                latent_size=latent_size,
                text_embed_dim=text_embed_dim,
                aux_loss_weight=aux_loss_weight,
                debug=debug,
            )
        else:
            # No conditioner - use default attention-based conditioning
            self.conditioner = None
            print("📌 No sequence conditioner - using default attention-based conditioning")

    @property
    def aux_loss_weight(self) -> float:
        """Get auxiliary loss weight from conditioner."""
        if self.conditioner is not None:
            return self.conditioner.aux_loss_weight
        return 0.0

    def reset_text_scale(self, value: float = 1.0):
        """Reset conditioner scale to a specific value."""
        if self.conditioner is not None and hasattr(self.conditioner, 'scale'):
            with torch.no_grad():
                self.conditioner.scale.fill_(value)
            print(f"📌 Reset conditioner scale to {value}")

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with backward compatibility for old checkpoints.
        
        Old checkpoints have weights like:
            esm_cond_proj.0.weight, esm_h0_proj.0.weight, esm_scale
            aa_embed.weight, aa_pos_embed.weight, aa_scale
            text_proj.0.weight, text_h0_proj.weight, text_scale
            
        New code expects:
            conditioner.esm_cond_proj.0.weight, conditioner.scale, etc.
        """
        # Map old weight names to new names
        remapped_state_dict = {}
        conditioner_prefixes = [
            # ESM conditioner
            ('esm_cond_proj.', 'conditioner.esm_cond_proj.'),
            ('esm_h0_proj.', 'conditioner.esm_h0_proj.'),
            ('esm_scale', 'conditioner.scale'),
            # Learned AA conditioner  
            ('aa_embed.', 'conditioner.aa_embed.'),
            ('aa_pos_embed.', 'conditioner.aa_pos_embed.'),
            ('aa_cond_proj.', 'conditioner.aa_cond_proj.'),
            ('aa_scale', 'conditioner.scale'),
            # Qwen text conditioner
            ('text_proj.', 'conditioner.text_proj.'),
            ('text_h0_proj.', 'conditioner.text_h0_proj.'),
            ('text_scale', 'conditioner.scale'),
        ]
        
        remapped_count = 0
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in conditioner_prefixes:
                if key == old_prefix or key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix, 1)
                    if new_key != key:
                        remapped_count += 1
                    break
            remapped_state_dict[new_key] = value
        
        if remapped_count > 0:
            print(f"📌 Loaded old checkpoint: remapped {remapped_count} conditioner weights to new format")
        
        return super().load_state_dict(remapped_state_dict, strict=strict)

    # ========== FORWARD PASS ==========
    @oom_decorator
    def forward(
            self,
            X,              # [Natom, 3], atom coordinates
            S,              # [Nblock], block types
            A,              # [Natom], atom types
            bonds,          # [Nbonds, 3], chemical bonds
            position_ids,   # [Nblock], block position ids
            chain_ids,      # [Nblock], split different chains
            generate_mask,  # [Nblock], 1 for generation, 0 for context
            center_mask,    # [Nblock], 1 for used to calculate complex center of mass
            block_lengths,  # [Nblock], number of atoms in each block
            lengths,        # [batch_size]
            is_aa,          # [Nblock], 1 for amino acid
            # Conditioning inputs (optional, depends on mode)
            text_k=None,
            text_v=None,
            mask_text=None,
            text_lengths=None,
            aa_indices=None,
            esm_embeddings=None,
            esm_valid_mask=None,  # [B, L] bool: True for non-X positions (for aux_loss)
            answer_seq_embeddings=None,  # [B, L_seq, hidden] for use_answer_seq_und mode
            t=None,
        ):
        """
        Forward pass with optional sequence conditioning.
        
        Conditioning is handled by the configured conditioner:
        - ESMConditioner: uses esm_embeddings
        - LearnedAAConditioner: uses aa_indices
        - QwenTextConditioner: uses text_v
        - AnswerSeqConditioner: uses answer_seq_embeddings (disables text K/V attention)
        - None: uses text_k, text_v for attention
        
        esm_valid_mask: Optional mask for non-X positions in sequences.
            When provided, aux_loss only includes non-X (True) positions.
        """
        # Encode structure to latent space
        with torch.no_grad():
            self.autoencoder.eval()
            Zh, Zx, _, _, _, _, _, _ = self.autoencoder.encode(
                X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths,
                deterministic=self.latent_deterministic
            )

        position_embedding = self.position_encoding(position_ids)

        # Normalize positions
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)

        # Is AA embedding (corrupt during training)
        corrupt_mask = generate_mask & (torch.rand_like(is_aa, dtype=torch.float) < self.is_aa_corrupt_ratio)
        is_aa_embedding = self.is_aa_embed(
            torch.where(corrupt_mask, torch.zeros_like(is_aa), is_aa).long()
        )

        # Base conditioning
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))

        # Apply sequence conditioning via conditioner
        aux_loss = None
        if self.conditioner is not None:
            embeddings = self._get_conditioning_embeddings(
                esm_embeddings=esm_embeddings,
                aa_indices=aa_indices,
                text_v=text_v,
                answer_seq_embeddings=answer_seq_embeddings,
            )
            if embeddings is not None:
                cond_embedding, aux_loss = self.conditioner.forward(
                    embeddings=embeddings,
                    cond_embedding=cond_embedding,
                    generate_mask=generate_mask,
                    lengths=lengths,
                    Zh=Zh,
                    mask_text=mask_text,
                    text_lengths=text_lengths,
                    valid_mask=esm_valid_mask,  # Mask for non-X positions
                )
                # Disable attention conditioning when using conditioner
                text_k, text_v, mask_text, text_lengths = None, None, None, None

        # Run diffusion
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

        # Compute total loss
        loss_dict['total'] = loss_dict['H'] * self.h_loss_weight + loss_dict['X']

        # Add auxiliary loss if computed
        if aux_loss is not None:
            loss_dict['aux_loss'] = aux_loss
            loss_dict['total'] = loss_dict['total'] + self.aux_loss_weight * aux_loss

        # Log conditioner scale (both generic 'text_scale' and specific name for wandb)
        if self.conditioner is not None and hasattr(self.conditioner, 'scale'):
            scale_val = self.conditioner.scale.detach()
            loss_dict['text_scale'] = scale_val  # For backward compatibility with training script
            loss_dict[f'{self.conditioner.name}_scale'] = scale_val  # Specific name

        return loss_dict

    def _get_conditioning_embeddings(
        self,
        esm_embeddings=None,
        aa_indices=None,
        text_v=None,
        answer_seq_embeddings=None,
    ):
        """Get the appropriate embeddings for the current conditioner."""
        if self.use_esm_embed and esm_embeddings is not None:
            return esm_embeddings
        elif self.use_answer_seq_und and answer_seq_embeddings is not None:
            return answer_seq_embeddings
        elif self.use_learned_aa_embed and aa_indices is not None:
            return aa_indices
        elif self.text_injection_mode and text_v is not None:
            return text_v
        return None

    # ========== TOPOLOGY EMBEDDING ==========
    def topo_embedding(self, A, bonds, block_ids, generate_mask):
        ctx_mask = ~generate_mask[block_ids]

        # Only retain bonds in the context
        bond_select_mask = ctx_mask[bonds[:, 0]] & ctx_mask[bonds[:, 1]]
        bonds = bonds[bond_select_mask]

        # Embed bond type
        edge_attr = self.bond_embed(bonds[:, 2])
        
        # Embed atom type
        H = self.atom_embed(A)

        # Get topo embedding
        topo_embedding = self.topo_gnn(H, bonds[:, :2].T, edge_attr)

        # Aggregate to each block
        topo_embedding = std_conserve_scatter_mean(topo_embedding, block_ids, dim=0)

        # Set generation part to zero
        topo_embedding = torch.where(
            generate_mask[:, None].expand_as(topo_embedding),
            torch.zeros_like(topo_embedding),
            topo_embedding
        )

        return topo_embedding

    # ========== POSITION NORMALIZATION ==========
    def _normalize_position(self, X, batch_ids, center_mask):
        centers = scatter_mean(X[center_mask], batch_ids[center_mask], dim=0, dim_size=batch_ids.max() + 1)
        centers = centers[batch_ids]
        X = (X - centers) / self.std
        return X, centers

    def _unnormalize_position(self, X_norm, centers, batch_ids):
        X = X_norm * self.std + centers
        return X

    # ========== SAMPLING ==========
    @torch.no_grad()
    def sample(
            self,
            X,
            S,
            A,
            bonds,
            position_ids,
            chain_ids,
            generate_mask,
            center_mask,
            block_lengths,
            lengths,
            is_aa,
            # Conditioning inputs
            text_k=None,
            text_v=None,
            mask_text=None,
            text_lengths=None,
            aa_indices=None,
            esm_embeddings=None,
            answer_seq_embeddings=None,  # [B, L_seq, hidden] for use_answer_seq_und mode
            sample_opt={},
            return_tensor=False,
        ):
        """
        Sample from the diffusion model with optional conditioning.
        """
        vae_decode_n_iter = sample_opt.pop('vae_decode_n_iter', 10)

        block_ids = length_to_batch_id(block_lengths)

        # Ensure no data leakage
        S[generate_mask] = 0
        X[generate_mask[block_ids]] = 0
        A[generate_mask[block_ids]] = 0
        ctx_atom_mask = ~generate_mask[block_ids]
        bonds = bonds[ctx_atom_mask[bonds[:, 0]] & ctx_atom_mask[bonds[:, 1]]]

        # Encode context
        self.autoencoder.eval()
        Zh, Zx, _, _, _, _, _, _ = self.autoencoder.encode(
            X, S, A, bonds, chain_ids, generate_mask, block_lengths, lengths,
            deterministic=self.latent_deterministic
        )

        # Normalize positions
        batch_ids = length_to_batch_id(lengths)
        Zx, centers = self._normalize_position(Zx, batch_ids, center_mask)

        # Build conditioning
        topo_embedding = self.topo_embedding(A, bonds, length_to_batch_id(block_lengths), generate_mask)
        position_embedding = self.position_encoding(position_ids)
        is_aa_embedding = self.is_aa_embed(is_aa.long())
        cond_embedding = self.cond_mlp(torch.cat([position_embedding, topo_embedding, is_aa_embedding], dim=-1))

        # Apply sequence conditioning
        if self.conditioner is not None:
            embeddings = self._get_conditioning_embeddings(
                esm_embeddings=esm_embeddings,
                aa_indices=aa_indices,
                text_v=text_v,
                answer_seq_embeddings=answer_seq_embeddings,
            )
            if embeddings is not None:
                cond_embedding = self.conditioner.condition_sample(
                    embeddings=embeddings,
                    cond_embedding=cond_embedding,
                    generate_mask=generate_mask,
                    lengths=lengths,
                    mask_text=mask_text,
                    text_lengths=text_lengths,
                )
                # Disable attention conditioning
                text_k, text_v, mask_text, text_lengths = None, None, None, None

        # Run diffusion sampling
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

        # Unnormalize
        X_0 = self._unnormalize_position(X_0, centers, batch_ids)

        # VAE decode
        return self.autoencoder.generate(
            X=X, S=S, A=A, bonds=bonds, position_ids=position_ids,
            chain_ids=chain_ids, generate_mask=generate_mask, block_lengths=block_lengths,
            lengths=lengths, is_aa=is_aa, given_latent=(H_0, X_0, None),
            n_iter=vae_decode_n_iter, topo_generate_mask=generate_mask
        )

    # ========== CONVENIENCE PROPERTIES FOR ESM ==========
    @property
    def esm_model(self):
        """Get ESM model from conditioner (for extraction)."""
        if hasattr(self, '_esm_conditioner') and self._esm_conditioner is not None:
            self._esm_conditioner._load_esm_model()
            return self._esm_conditioner.esm_model
        return None

    @property
    def esm_alphabet(self):
        """Get ESM alphabet from conditioner."""
        if hasattr(self, '_esm_conditioner') and self._esm_conditioner is not None:
            self._esm_conditioner._load_esm_model()
            return self._esm_conditioner.esm_alphabet
        return None

    @property
    def esm_batch_converter(self):
        """Get ESM batch converter from conditioner."""
        if hasattr(self, '_esm_conditioner') and self._esm_conditioner is not None:
            self._esm_conditioner._load_esm_model()
            return self._esm_conditioner.esm_batch_converter
        return None

    # ========== BACKWARD COMPATIBILITY ==========
    # These properties maintain backward compatibility with code that
    # accesses conditioner attributes directly on LDM
    
    @property
    def text_scale(self):
        """Get text scale from Qwen conditioner."""
        if self.conditioner is not None and hasattr(self.conditioner, 'scale'):
            return self.conditioner.scale
        return None

    @property
    def aa_scale(self):
        """Get AA scale from learned AA conditioner."""
        if self.conditioner is not None and hasattr(self.conditioner, 'scale'):
            return self.conditioner.scale
        return None

    @property
    def esm_scale(self):
        """Get ESM scale from ESM conditioner."""
        if self.conditioner is not None and hasattr(self.conditioner, 'scale'):
            return self.conditioner.scale
        return None

    @property
    def aa_vocab(self):
        """Get AA vocab from learned AA conditioner."""
        if isinstance(self.conditioner, LearnedAAConditioner):
            return self.conditioner.AA_VOCAB
        return None

    @property
    def aa_to_idx(self):
        """Get AA to index mapping from learned AA conditioner."""
        if isinstance(self.conditioner, LearnedAAConditioner):
            return self.conditioner.aa_to_idx
        return None
