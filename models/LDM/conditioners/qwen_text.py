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
    
    Architecture (matches learned_aa_embed pattern):
    - text_h0_proj: Qwen → H_0 (latent_size) - SUPERVISED by aux_loss
    - text_h0_to_cond: H_0 → conditioning (hidden_size)
    - text_scale: Learnable scaling factor
    
    Key insight: Conditioning must come FROM the supervised representation.
    This matches learned_aa_embed which achieves AAR ≈ 1.0:
    
    learned_aa:  aa_indices → aa_embed (SUPERVISED) → aa_cond_proj → cond
    Qwen:        qwen_embed → text_h0_proj (SUPERVISED) → text_h0_to_cond → cond
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
        
        # Step 1: Project Qwen to H_0 space (SUPERVISED by aux_loss)
        self.text_h0_proj = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, latent_size),
        )
        # Initialize output layer with small weights to match H_0 scale
        nn.init.normal_(self.text_h0_proj[-1].weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.text_h0_proj[-1].bias)
        
        # Step 2: Project text_h0_pred (supervised) → conditioning space
        self.text_h0_to_cond = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        
        # Override scale with better name
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        print(f"📌 {self.name} (conditioning FROM supervised rep):")
        print(f"   Qwen ({text_embed_dim}) → text_h0_proj → H_0 ({latent_size}) [SUPERVISED]")
        print(f"   H_0 ({latent_size}) → text_h0_to_cond → cond ({hidden_size})")
    
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
        
        # Step 1: Qwen → text_h0_pred (SUPERVISED by aux_loss)
        proj_dtype = next(self.text_h0_proj.parameters()).dtype
        embeddings = embeddings.to(dtype=proj_dtype)
        text_h0_pred = self.text_h0_proj(embeddings)  # [B, L, latent_size] - SUPERVISED
        
        # Step 2: text_h0_pred → conditioning (derived FROM supervised rep)
        text_cond = self.text_h0_to_cond(text_h0_pred)  # [B, L, hidden_size]
        
        self._debug_print(f"\n🔍 {self.name} DEBUG:")
        self._debug_print(f"  embeddings shape: {embeddings.shape}")
        self._debug_print(f"  text_h0_pred (supervised) stats: mean={text_h0_pred.mean():.4f}, std={text_h0_pred.std():.4f}")
        self._debug_print(f"  text_cond (from h0_pred) stats: mean={text_cond.mean():.4f}, std={text_cond.std():.4f}")
        self._debug_print(f"  H_0 (target) stats: mean={Zh.mean():.4f}, std={Zh.std():.4f}")
        
        # Add text_cond (derived from supervised rep) to cond_embedding
        cond_embedding = self._add_to_cond_embedding(
            text_cond, cond_embedding, generate_mask, lengths
        )
        
        self._debug_print(f"  scale: {self.scale.item():.4f}")
        
        # Compute auxiliary loss (supervises text_h0_pred directly)
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
        
        # Step 1: Qwen → text_h0_pred (supervised representation)
        proj_dtype = next(self.text_h0_proj.parameters()).dtype
        embeddings = embeddings.to(dtype=proj_dtype)
        text_h0_pred = self.text_h0_proj(embeddings)  # [B, L, latent_size]
        
        # Step 2: text_h0_pred → conditioning (from supervised rep)
        text_cond = self.text_h0_to_cond(text_h0_pred)  # [B, L, hidden_size]
        
        # Add to cond_embedding
        return self._add_to_cond_embedding(
            text_cond, cond_embedding, generate_mask, lengths
        )

