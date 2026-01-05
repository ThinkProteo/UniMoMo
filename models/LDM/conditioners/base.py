"""
Base class for sequence conditioners.

All conditioners follow a common pattern:
1. Project input embeddings to cond_embedding space (hidden_size)
2. Optionally project to H_0 space (latent_size) for auxiliary supervision
3. Apply scaling factor to balance conditioning strength
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn


class BaseConditioner(ABC, nn.Module):
    """
    Abstract base class for sequence conditioners.
    
    Conditioners add sequence information to the diffusion process:
    - Forward: Add conditioning to cond_embedding during training
    - Sample: Add conditioning to cond_embedding during inference
    - Aux loss: Optional direct supervision for the projection layers
    
    Attributes:
        hidden_size: Dimension of cond_embedding
        latent_size: Dimension of H_0 (VAE latent)
        scale: Learnable scaling factor for conditioning strength
        aux_loss_weight: Weight for auxiliary loss (0 to disable)
    """
    
    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        aux_loss_weight: float = 1.0,
        debug: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.aux_loss_weight = aux_loss_weight
        self.debug = debug
        
        # Learnable scaling factor (subclasses should initialize)
        self.scale = nn.Parameter(torch.tensor(1.0))
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this conditioner for logging."""
        pass
    
    @abstractmethod
    def forward(
        self,
        embeddings: torch.Tensor,
        cond_embedding: torch.Tensor,
        generate_mask: torch.Tensor,
        lengths: torch.Tensor,
        Zh: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply conditioning during training forward pass.
        
        Args:
            embeddings: Input embeddings [B, L, embed_dim]
            cond_embedding: Existing conditioning [N, hidden_size]
            generate_mask: CDR mask [N]
            lengths: Sample lengths [B]
            Zh: Target H_0 for auxiliary loss [N, latent_size]
            **kwargs: Additional conditioner-specific args
            
        Returns:
            cond_embedding: Modified conditioning tensor [N, hidden_size]
            aux_loss: Auxiliary loss (or None if not computed)
        """
        pass
    
    @abstractmethod
    def condition_sample(
        self,
        embeddings: torch.Tensor,
        cond_embedding: torch.Tensor,
        generate_mask: torch.Tensor,
        lengths: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply conditioning during sampling (inference).
        
        Args:
            embeddings: Input embeddings [B, L, embed_dim]
            cond_embedding: Existing conditioning [N, hidden_size]
            generate_mask: CDR mask [N]
            lengths: Sample lengths [B]
            **kwargs: Additional conditioner-specific args
            
        Returns:
            cond_embedding: Modified conditioning tensor [N, hidden_size]
        """
        pass
    
    def _add_to_cond_embedding(
        self,
        projected_emb: torch.Tensor,
        cond_embedding: torch.Tensor,
        generate_mask: torch.Tensor,
        lengths: torch.Tensor,
        return_positions: bool = False,
    ) -> torch.Tensor:
        """
        Helper to add projected embeddings to cond_embedding at CDR positions.
        
        This is the common operation across all conditioners:
        1. For each sample, find CDR positions
        2. Map embeddings 1:1 to CDR positions
        3. Add scaled embeddings to cond_embedding
        
        Args:
            projected_emb: [B, L, hidden_size] - projected embeddings
            cond_embedding: [N, hidden_size] - existing conditioning
            generate_mask: [N] - CDR mask
            lengths: [B] - sample lengths
            return_positions: If True, also return list of (global_positions, n_to_add) per sample
            
        Returns:
            cond_embedding: Modified conditioning tensor
            positions: (optional) List of (global_positions, n_to_add) per sample
        """
        batch_size = lengths.shape[0]
        B, L_emb, _ = projected_emb.shape
        
        positions_list = [] if return_positions else None
        
        offset = 0
        for sample_idx in range(batch_size):
            sample_len = int(lengths[sample_idx].item())
            sample_mask = generate_mask[offset:offset + sample_len]
            n_cdr = int(sample_mask.sum().item())
            n_to_add = min(n_cdr, L_emb)
            
            if n_to_add > 0:
                cdr_positions = sample_mask.nonzero(as_tuple=True)[0][:n_to_add]
                global_positions = offset + cdr_positions
                
                emb = projected_emb[sample_idx, :n_to_add]
                cond_embedding[global_positions] = cond_embedding[global_positions] + self.scale * emb
                
                if return_positions:
                    positions_list.append((global_positions, n_to_add))
            else:
                if return_positions:
                    positions_list.append((None, 0))
            
            offset += sample_len
        
        if return_positions:
            return cond_embedding, positions_list
        return cond_embedding
    
    def _compute_aux_loss(
        self,
        h0_pred: torch.Tensor,
        h0_target: torch.Tensor,
        generate_mask: torch.Tensor,
        lengths: torch.Tensor,
        L_emb: int,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for direct H_0 prediction.
        
        Args:
            h0_pred: [B, L, latent_size] - predicted H_0
            h0_target: [N, latent_size] - target H_0 (Zh)
            generate_mask: [N] - CDR mask
            lengths: [B] - sample lengths
            L_emb: Length of embedding sequence
            valid_mask: [B, L] bool - True for non-X positions (optional)
                        If provided, only non-X positions contribute to loss.
            
        Returns:
            aux_loss: Scalar loss tensor
        """
        import torch.nn.functional as F
        
        batch_size = lengths.shape[0]
        
        aux_preds = []
        aux_targets = []
        n_valid_total = 0
        n_masked_total = 0
        
        offset = 0
        for sample_idx in range(batch_size):
            sample_len = int(lengths[sample_idx].item())
            sample_mask = generate_mask[offset:offset + sample_len]
            n_cdr = int(sample_mask.sum().item())
            n_to_match = min(n_cdr, L_emb)
            
            if n_to_match > 0:
                cdr_positions = sample_mask.nonzero(as_tuple=True)[0][:n_to_match]
                global_positions = offset + cdr_positions
                
                # Get predictions and targets for this sample
                sample_preds = h0_pred[sample_idx, :n_to_match]  # [n_to_match, latent_size]
                sample_targets = h0_target[global_positions]     # [n_to_match, latent_size]
                
                # Apply valid_mask if provided (exclude X positions from loss)
                if valid_mask is not None:
                    sample_valid = valid_mask[sample_idx, :n_to_match]  # [n_to_match]
                    n_valid = sample_valid.sum().item()
                    n_masked = n_to_match - n_valid
                    n_valid_total += n_valid
                    n_masked_total += n_masked
                    
                    if n_valid > 0:
                        # Only include non-X positions
                        aux_preds.append(sample_preds[sample_valid])
                        aux_targets.append(sample_targets[sample_valid])
                else:
                    aux_preds.append(sample_preds)
                    aux_targets.append(sample_targets)
            
            offset += sample_len
        
        if aux_preds:
            aux_preds_cat = torch.cat(aux_preds, dim=0)
            aux_targets_cat = torch.cat(aux_targets, dim=0)
            
            if valid_mask is not None and self.debug:
                print(f"  📍 aux_loss: {n_valid_total} valid positions, {n_masked_total} X positions masked")
            
            return F.mse_loss(aux_preds_cat, aux_targets_cat)
        
        return torch.tensor(0.0, device=h0_target.device)
    
    def _debug_print(self, msg: str):
        """Print debug message if debug mode is enabled."""
        if self.debug:
            print(msg)

