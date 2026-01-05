"""
Learned Amino Acid Embedding Conditioner.

Direct learned embeddings for each amino acid, bypassing Qwen.
This is the simplest possible sequence → structure shortcut.

Input: Amino acid indices [B, L]
Output: Conditioning added to cond_embedding at CDR positions
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .base import BaseConditioner


class LearnedAAConditioner(BaseConditioner):
    """
    Conditioner using learned amino acid embeddings.
    
    Architecture:
    - aa_embed: Learnable AA embeddings directly in H_0 (latent) space
    - aa_pos_embed: Learnable position embeddings for CDR positions
    - aa_cond_proj: Projects combined embedding to cond_embedding space
    - aa_scale: Learnable scaling factor
    
    This provides a direct, fully learnable shortcut: AA → H_0.
    The model can learn the exact H_0 representation for each AA.
    """
    
    # Standard amino acid vocabulary
    AA_VOCAB = "ACDEFGHIKLMNPQRSTVWYX"  # 21 amino acids (including X for unknown)
    
    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        max_positions: int = 50,
        h0_init_std: float = 0.35,  # Match H_0 scale
        pos_init_std: float = 0.1,
        aux_loss_weight: float = 1.0,
        debug: bool = True,
    ):
        super().__init__(hidden_size, latent_size, aux_loss_weight, debug)
        
        self.max_positions = max_positions
        
        # Build vocab mapping
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.AA_VOCAB)}
        
        # Direct embedding to VAE latent space
        # Initialize with std matching H_0 scale (~0.35)
        self.aa_embed = nn.Embedding(len(self.AA_VOCAB), latent_size)
        nn.init.normal_(self.aa_embed.weight, mean=0.0, std=h0_init_std)
        
        # Position embedding for CDR positions
        # Smaller std to not overwhelm AA identity signal
        self.aa_pos_embed = nn.Embedding(max_positions, latent_size)
        nn.init.normal_(self.aa_pos_embed.weight, mean=0.0, std=pos_init_std)
        
        # Project to cond_embedding space
        self.aa_cond_proj = nn.Linear(latent_size, hidden_size)
        
        print(f"📌 {self.name}: AA → H_0 ({latent_size}), AA → cond ({hidden_size})")
        print(f"   Vocab: {self.AA_VOCAB}")
    
    @property
    def name(self) -> str:
        return "LearnedAAConditioner"
    
    def sequences_to_indices(self, sequences: list, device: torch.device) -> torch.Tensor:
        """
        Convert amino acid sequences to indices.
        
        Args:
            sequences: List of AA sequences (strings)
            device: Target device
            
        Returns:
            aa_indices: [B, max_len] tensor of AA indices
        """
        unknown_idx = self.aa_to_idx['X']
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        
        aa_indices = torch.full((batch_size, max_len), unknown_idx, dtype=torch.long, device=device)
        
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                aa_indices[i, j] = self.aa_to_idx.get(aa.upper(), unknown_idx)
        
        return aa_indices
    
    def forward(
        self,
        embeddings: torch.Tensor,
        cond_embedding: torch.Tensor,
        generate_mask: torch.Tensor,
        lengths: torch.Tensor,
        Zh: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply learned AA conditioning during training.
        
        Args:
            embeddings: AA indices [B, L] (torch.long)
            cond_embedding: Existing conditioning [N, hidden_size]
            generate_mask: CDR mask [N]
            lengths: Sample lengths [B]
            Zh: Target H_0 [N, latent_size]
            valid_mask: [B, L] bool - True for non-X positions (optional)
                        If provided, only non-X positions contribute to aux_loss.
        """
        aa_indices = embeddings  # [B, L_aa]
        B, L_aa = aa_indices.shape
        
        # Get AA embeddings
        aa_embed_raw = self.aa_embed(aa_indices)  # [B, L, latent_size]
        
        # Add position embeddings
        pos_indices = torch.arange(L_aa, device=aa_indices.device).unsqueeze(0).expand(B, -1)
        pos_indices = pos_indices.clamp(max=self.max_positions - 1)
        pos_embed = self.aa_pos_embed(pos_indices)  # [B, L, latent_size]
        
        # Combine: AA identity + position
        aa_h0_pred = aa_embed_raw + pos_embed  # [B, L, latent_size]
        
        # Project to cond_embedding space
        aa_cond = self.aa_cond_proj(aa_h0_pred)  # [B, L, hidden_size]
        
        self._debug_print(f"\n🔤 {self.name} DEBUG:")
        self._debug_print(f"  aa_indices shape: {aa_indices.shape}")
        self._debug_print(f"  aa_embed_raw stats: mean={aa_embed_raw.mean():.4f}, std={aa_embed_raw.std():.4f}")
        self._debug_print(f"  pos_embed stats: mean={pos_embed.mean():.4f}, std={pos_embed.std():.4f}")
        self._debug_print(f"  aa_h0_pred stats: mean={aa_h0_pred.mean():.4f}, std={aa_h0_pred.std():.4f}")
        self._debug_print(f"  aa_cond stats: mean={aa_cond.mean():.4f}, std={aa_cond.std():.4f}")
        
        # Add to cond_embedding
        cond_embedding = self._add_to_cond_embedding(
            aa_cond, cond_embedding, generate_mask, lengths
        )
        
        # Compute auxiliary loss (direct AA → H_0)
        # If valid_mask provided, only non-X positions contribute to loss
        aux_loss = self._compute_aux_loss(aa_h0_pred, Zh, generate_mask, lengths, L_aa, valid_mask)
        
        self._debug_print(f"  🎯 AA AUX LOSS: {aux_loss.item():.4f}")
        self._debug_print(f"  scale: {self.scale.item():.4f}")
        
        return cond_embedding, aux_loss
    
    def condition_sample(
        self,
        embeddings: torch.Tensor,
        cond_embedding: torch.Tensor,
        generate_mask: torch.Tensor,
        lengths: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Apply learned AA conditioning during sampling."""
        aa_indices = embeddings  # [B, L_aa]
        B, L_aa = aa_indices.shape
        
        # Get AA + position embeddings
        aa_embed_raw = self.aa_embed(aa_indices)
        pos_indices = torch.arange(L_aa, device=aa_indices.device).unsqueeze(0).expand(B, -1)
        pos_indices = pos_indices.clamp(max=self.max_positions - 1)
        pos_embed = self.aa_pos_embed(pos_indices)
        
        aa_h0_pred = aa_embed_raw + pos_embed
        aa_cond = self.aa_cond_proj(aa_h0_pred)
        
        # Add to cond_embedding
        return self._add_to_cond_embedding(
            aa_cond, cond_embedding, generate_mask, lengths
        )

