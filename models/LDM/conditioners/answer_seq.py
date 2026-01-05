"""
Answer Sequence Conditioner (use_answer_seq_und mode).

Uses hidden states from the answer_sequence tokens in the Qwen output
for conditioning the diffusion model.

Unlike QKV-based conditioning which uses K/V from thinking tokens,
this conditioner:
1. Extracts last-layer hidden states from answer_sequence tokens
2. Projects them to conditioning space
3. Does NOT use text K/V attention in diffusion

Input: Qwen last-layer hidden states from answer_sequence tokens [B, L_seq, 2560]
Output: Conditioning added to cond_embedding at CDR positions

Advantage: The answer_sequence tokens directly encode the CDR sequence,
providing strong per-residue conditioning signal.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from .base import BaseConditioner


class AnswerSeqConditioner(BaseConditioner):
    """
    Conditioner using hidden states from answer_sequence tokens.
    
    Architecture (same as QwenTextConditioner):
    - answer_h0_proj: Hidden states → H_0 (latent_size) - SUPERVISED by aux_loss
    - answer_h0_to_cond: H_0 → conditioning (hidden_size)
    - answer_scale: Learnable scaling factor
    
    Key insight: answer_sequence tokens directly encode the CDR amino acids,
    so their hidden states contain rich per-residue information.
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
        
        # Step 1: Project hidden states to H_0 space (SUPERVISED by aux_loss)
        self.answer_h0_proj = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, latent_size),
        )
        # Initialize output layer with small weights to match H_0 scale
        nn.init.normal_(self.answer_h0_proj[-1].weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.answer_h0_proj[-1].bias)
        
        # Step 2: Project answer_h0_pred (supervised) → conditioning space
        self.answer_h0_to_cond = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        
        # Override scale with better name
        self.scale = nn.Parameter(torch.tensor(1.0))
        
        print(f"📌 {self.name} (conditioning FROM supervised rep):")
        print(f"   Hidden states ({text_embed_dim}) → answer_h0_proj → H_0 ({latent_size}) [SUPERVISED]")
        print(f"   H_0 ({latent_size}) → answer_h0_to_cond → cond ({hidden_size})")
    
    @property
    def name(self) -> str:
        return "AnswerSeqConditioner"
    
    def forward(
        self,
        embeddings: torch.Tensor,
        cond_embedding: torch.Tensor,
        generate_mask: torch.Tensor,
        lengths: torch.Tensor,
        Zh: torch.Tensor,
        mask_text: Optional[torch.Tensor] = None,
        text_lengths: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply answer_sequence conditioning during training.
        
        Args:
            embeddings: Hidden states from answer_sequence tokens [B, L_seq, text_embed_dim]
            cond_embedding: Existing conditioning [N, hidden_size]
            generate_mask: CDR mask [N]
            lengths: Sample lengths [B]
            Zh: Target H_0 [N, latent_size]
            mask_text: Not used (included for API compatibility)
            text_lengths: Actual sequence lengths [B]
            valid_mask: [B, L] bool - True for non-X positions (optional)
        """
        if embeddings is None:
            return cond_embedding, None
        
        B, L_seq, _ = embeddings.shape
        
        # Step 1: Hidden states → answer_h0_pred (SUPERVISED by aux_loss)
        proj_dtype = next(self.answer_h0_proj.parameters()).dtype
        embeddings = embeddings.to(dtype=proj_dtype)
        answer_h0_pred = self.answer_h0_proj(embeddings)  # [B, L_seq, latent_size] - SUPERVISED
        
        # Step 2: answer_h0_pred → conditioning (derived FROM supervised rep)
        answer_cond = self.answer_h0_to_cond(answer_h0_pred)  # [B, L_seq, hidden_size]
        
        self._debug_print(f"\n🔍 {self.name} DEBUG:")
        self._debug_print(f"  embeddings shape: {embeddings.shape}")
        self._debug_print(f"  answer_h0_pred (supervised) stats: mean={answer_h0_pred.mean():.4f}, std={answer_h0_pred.std():.4f}")
        self._debug_print(f"  answer_cond (from h0_pred) stats: mean={answer_cond.mean():.4f}, std={answer_cond.std():.4f}")
        self._debug_print(f"  H_0 (target) stats: mean={Zh.mean():.4f}, std={Zh.std():.4f}")
        
        # Add answer_cond (derived from supervised rep) to cond_embedding
        cond_embedding = self._add_to_cond_embedding(
            answer_cond, cond_embedding, generate_mask, lengths
        )
        
        self._debug_print(f"  scale: {self.scale.item():.4f}")
        
        # Compute auxiliary loss (supervises answer_h0_pred directly)
        aux_loss = self._compute_aux_loss(answer_h0_pred, Zh, generate_mask, lengths, L_seq, valid_mask)
        
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
        """Apply answer_sequence conditioning during sampling."""
        if embeddings is None:
            return cond_embedding
        
        # Step 1: Hidden states → answer_h0_pred
        proj_dtype = next(self.answer_h0_proj.parameters()).dtype
        embeddings = embeddings.to(dtype=proj_dtype)
        answer_h0_pred = self.answer_h0_proj(embeddings)  # [B, L_seq, latent_size]
        
        # Step 2: answer_h0_pred → conditioning (from supervised rep)
        answer_cond = self.answer_h0_to_cond(answer_h0_pred)  # [B, L_seq, hidden_size]
        
        # Add to cond_embedding
        return self._add_to_cond_embedding(
            answer_cond, cond_embedding, generate_mask, lengths
        )

