"""
Qwen Text Injection Conditioner.

Uses per-residue hidden states from Qwen3 for conditioning.
Each token corresponds to one amino acid (1:1 mapping).

Input: Qwen hidden states [B, L, 2560]
Output: Conditioning added to cond_embedding at CDR positions
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .base import BaseConditioner


class QwenTextConditioner(BaseConditioner):
    """
    Conditioner using Qwen3 per-residue embeddings.
    
    Architecture:
    - text_proj: Projects Qwen hidden states to cond_embedding space
    - text_h0_proj: Projects to H_0 for auxiliary supervision
    - text_scale: Learnable scaling factor
    
    The auxiliary loss provides direct supervision for the projection,
    helping the model learn to map text representations to structure.
    """
    
    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        text_embed_dim: int = 2560,  # Qwen3-4B hidden_size
        aux_loss_weight: float = 1.0,
        debug: bool = True,
    ):
        super().__init__(hidden_size, latent_size, aux_loss_weight, debug)
        
        self.text_embed_dim = text_embed_dim
        
        # Project text embeddings to cond_embedding space
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        
        # Project to H_0 for auxiliary supervision
        self.text_h0_proj = nn.Linear(hidden_size, latent_size)
        
        # Override scale with better name
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        print(f"📌 {self.name}: text ({text_embed_dim}) → cond ({hidden_size}) + H_0 ({latent_size})")
    
    @property
    def name(self) -> str:
        return "QwenTextConditioner"
    
    def forward(
        self,
        embeddings: torch.Tensor,
        cond_embedding: torch.Tensor,
        generate_mask: torch.Tensor,
        lengths: torch.Tensor,
        Zh: torch.Tensor,
        mask_text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply Qwen text conditioning during training.
        
        Args:
            embeddings: Qwen hidden states [B, L, text_embed_dim]
            cond_embedding: Existing conditioning [N, hidden_size]
            generate_mask: CDR mask [N]
            lengths: Sample lengths [B]
            Zh: Target H_0 [N, latent_size]
            mask_text: Text attention mask [B, L]
            text_lengths: Actual text lengths [B]
        """
        # Handle different input formats
        if embeddings.dim() == 4:
            # Old format: [B, L, n_heads, head_dim] -> flatten
            B, L, n_heads, head_dim = embeddings.shape
            embeddings = embeddings.view(B, L, n_heads * head_dim)
        
        B, L_text, _ = embeddings.shape
        
        # Project to cond_embedding space
        proj_dtype = next(self.text_proj.parameters()).dtype
        embeddings = embeddings.to(dtype=proj_dtype)
        text_cond = self.text_proj(embeddings)  # [B, L, hidden_size]
        
        self._debug_print(f"\n🔍 {self.name} DEBUG:")
        self._debug_print(f"  embeddings shape: {embeddings.shape}, text_cond shape: {text_cond.shape}")
        self._debug_print(f"  text_cond stats: mean={text_cond.mean():.4f}, std={text_cond.std():.4f}")
        self._debug_print(f"  cond_embedding (before) stats: mean={cond_embedding.mean():.4f}, std={cond_embedding.std():.4f}")
        
        # Add to cond_embedding at CDR positions
        cond_embedding = self._add_to_cond_embedding(
            text_cond, cond_embedding, generate_mask, lengths
        )
        
        self._debug_print(f"  scale: {self.scale.item():.4f}")
        self._debug_print(f"  cond_embedding (after) stats: mean={cond_embedding.mean():.4f}, std={cond_embedding.std():.4f}")
        
        # Compute auxiliary loss
        text_h0_pred = self.text_h0_proj(text_cond)  # [B, L, latent_size]
        aux_loss = self._compute_aux_loss(text_h0_pred, Zh, generate_mask, lengths, L_text)
        
        self._debug_print(f"  🎯 AUX LOSS: {aux_loss.item():.4f}")
        
        return cond_embedding, aux_loss
    
    def condition_sample(
        self,
        embeddings: torch.Tensor,
        cond_embedding: torch.Tensor,
        generate_mask: torch.Tensor,
        lengths: torch.Tensor,
        mask_text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply Qwen text conditioning during sampling."""
        # Handle different input formats
        if embeddings.dim() == 4:
            B, L, n_heads, head_dim = embeddings.shape
            embeddings = embeddings.view(B, L, n_heads * head_dim)
        
        # Project to cond_embedding space
        proj_dtype = next(self.text_proj.parameters()).dtype
        embeddings = embeddings.to(dtype=proj_dtype)
        text_cond = self.text_proj(embeddings)
        
        # Add to cond_embedding
        return self._add_to_cond_embedding(
            text_cond, cond_embedding, generate_mask, lengths
        )

