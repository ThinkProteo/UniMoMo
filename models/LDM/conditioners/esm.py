"""
ESM-2 Embedding Conditioner.

Uses pretrained ESM-2 protein language model for conditioning.
ESM embeddings encode evolutionary and structural context, not just AA identity.

Input: ESM embeddings [B, L, 1280] (pre-extracted) or sequences
Output: Conditioning added to cond_embedding at CDR positions
"""

from typing import Optional, Tuple, List
import torch
import torch.nn as nn

from .base import BaseConditioner


class ESMConditioner(BaseConditioner):
    """
    Conditioner using ESM-2 protein language model embeddings.
    
    Architecture (matches learned_aa_embed pattern):
    - esm_model: Frozen ESM-2 model (loaded on demand)
    - esm_h0_proj: ESM → H_0 (latent_size) - SUPERVISED by aux_loss
    - esm_h0_to_cond: H_0 → conditioning (hidden_size)
    - esm_scale: Learnable scaling factor
    
    Key insight: Conditioning must come FROM the supervised representation.
    This matches learned_aa_embed which achieves AAR ≈ 1.0:
    
    learned_aa:  aa_indices → aa_embed (SUPERVISED) → aa_cond_proj → cond
    ESM:         esm_embed → esm_h0_proj (SUPERVISED) → esm_h0_to_cond → cond
    
    Both derive conditioning FROM the aux_loss supervised representation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        latent_size: int,
        esm_model_name: str = "esm2_t33_650M_UR50D",
        esm_embed_dim: int = 1280,  # ESM-2 650M hidden size
        aux_loss_weight: float = 1.0,
        debug: bool = True,
    ):
        super().__init__(hidden_size, latent_size, aux_loss_weight, debug)
        
        self.esm_model_name = esm_model_name
        self.esm_embed_dim = esm_embed_dim
        
        # ESM model (loaded on demand to avoid import errors)
        self.esm_model = None
        self.esm_alphabet = None
        self.esm_batch_converter = None
        
        # Architecture matching learned_aa_embed (which works):
        # ESM → esm_h0_proj → esm_h0_pred (SUPERVISED) → esm_h0_to_cond → conditioning
        # The key: conditioning must come FROM the supervised representation!
        
        # Step 1: Project ESM to H_0 space (this is SUPERVISED by aux_loss)
        self.esm_h0_proj = nn.Sequential(
            nn.Linear(esm_embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, latent_size),
        )
        # Initialize output layer with small weights to match H_0 scale
        nn.init.normal_(self.esm_h0_proj[-1].weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.esm_h0_proj[-1].bias)
        
        # Step 2: Project esm_h0_pred (supervised) → conditioning space
        # This is analogous to aa_cond_proj in learned_aa_embed
        self.esm_h0_to_cond = nn.Sequential(
            nn.Linear(latent_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        
        print(f"📌 {self.name} (conditioning FROM supervised rep):")
        print(f"   ESM-2 ({esm_embed_dim}) → esm_h0_proj → H_0 ({latent_size}) [SUPERVISED]")
        print(f"   H_0 ({latent_size}) → esm_h0_to_cond → cond ({hidden_size})")
        print(f"   ✓ Conditioning derived FROM supervised representation (like learned_aa)")
    
    @property
    def name(self) -> str:
        return "ESMConditioner"
    
    def _load_esm_model(self):
        """Load ESM model on first use."""
        if self.esm_model is not None:
            return
        
        try:
            import esm
            self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
            self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
            self.esm_model.eval()
            for param in self.esm_model.parameters():
                param.requires_grad = False
            print(f"✓ {self.name}: Loaded ESM-2 650M")
        except ImportError:
            raise ImportError("ESM not installed! Run: pip install fair-esm")
    
    def extract_embeddings(self, sequences: List[str], device: torch.device) -> torch.Tensor:
        """
        Extract ESM embeddings from sequences.
        
        Args:
            sequences: List of AA sequences
            device: Target device
            
        Returns:
            embeddings: [B, max_len, esm_embed_dim]
        """
        self._load_esm_model()
        
        batch_size = len(sequences)
        max_len = max(len(seq) for seq in sequences)
        
        # Prepare data for ESM
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.esm_batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        # Move ESM model to device if needed
        self.esm_model = self.esm_model.to(device)
        
        # Extract representations
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        
        # Get per-residue embeddings from last layer
        # Shape: [B, seq_len + 2, esm_embed_dim] (includes BOS/EOS)
        token_representations = results["representations"][33]
        
        # Remove BOS/EOS tokens
        embeddings = token_representations[:, 1:-1, :]
        
        # Pad/truncate to max_len
        if embeddings.shape[1] != max_len:
            import torch.nn.functional as F
            if embeddings.shape[1] > max_len:
                embeddings = embeddings[:, :max_len, :]
            else:
                pad_len = max_len - embeddings.shape[1]
                embeddings = F.pad(embeddings, (0, 0, 0, pad_len))
        
        return embeddings
    
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
        Apply ESM conditioning during training.
        
        Args:
            embeddings: ESM embeddings [B, L, esm_embed_dim]
            cond_embedding: Existing conditioning [N, hidden_size]
            generate_mask: CDR mask [N]
            lengths: Sample lengths [B]
            Zh: Target H_0 [N, latent_size]
        """
        B, L_esm, esm_dim = embeddings.shape
        
        # Step 1: ESM → esm_h0_pred (SUPERVISED by aux_loss)
        proj_dtype = next(self.esm_h0_proj.parameters()).dtype
        embeddings = embeddings.to(dtype=proj_dtype)
        esm_h0_pred = self.esm_h0_proj(embeddings)  # [B, L, latent_size] - SUPERVISED
        
        # Step 2: esm_h0_pred → conditioning (derived FROM supervised rep)
        esm_cond = self.esm_h0_to_cond(esm_h0_pred)  # [B, L, hidden_size]
        
        self._debug_print(f"\n🧬 {self.name} DEBUG:")
        self._debug_print(f"  ESM embeddings shape: {embeddings.shape}")
        self._debug_print(f"  esm_h0_pred (supervised) stats: mean={esm_h0_pred.mean():.4f}, std={esm_h0_pred.std():.4f}")
        self._debug_print(f"  esm_cond (from h0_pred) stats: mean={esm_cond.mean():.4f}, std={esm_cond.std():.4f}")
        self._debug_print(f"  H_0 (target) stats: mean={Zh.mean():.4f}, std={Zh.std():.4f}")
        
        # Add esm_cond (derived from supervised rep) to cond_embedding
        cond_embedding = self._add_to_cond_embedding(
            esm_cond, cond_embedding, generate_mask, lengths
        )
        
        # Compute auxiliary loss (supervises esm_h0_pred directly)
        # Gradients flow: aux_loss → esm_h0_proj → ESM
        # And: aux_loss → esm_h0_pred → esm_h0_to_cond → cond_embedding
        aux_loss = self._compute_aux_loss(esm_h0_pred, Zh, generate_mask, lengths, L_esm)
        
        self._debug_print(f"  🎯 ESM AUX LOSS: {aux_loss.item():.4f}")
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
        """Apply ESM conditioning during sampling."""
        # Step 1: ESM → esm_h0_pred (supervised representation)
        proj_dtype = next(self.esm_h0_proj.parameters()).dtype
        embeddings = embeddings.to(dtype=proj_dtype)
        esm_h0_pred = self.esm_h0_proj(embeddings)  # [B, L, latent_size]
        
        # Step 2: esm_h0_pred → conditioning (from supervised rep)
        esm_cond = self.esm_h0_to_cond(esm_h0_pred)  # [B, L, hidden_size]
        
        # Add to cond_embedding
        return self._add_to_cond_embedding(
            esm_cond, cond_embedding, generate_mask, lengths
        )

