"""
Sequence conditioners for the LDM diffusion model.

Each conditioner handles:
1. Initialization (create projection layers)
2. Forward conditioning (add to cond_embedding during training)
3. Sample conditioning (add to cond_embedding during inference)
4. Auxiliary loss computation (optional direct supervision)

Available conditioners:
- QwenTextConditioner: Per-residue Qwen embeddings for conditioning
- LearnedAAConditioner: Learned amino acid embeddings with position
- ESMConditioner: ESM-2 protein language model embeddings
"""

from .base import BaseConditioner
from .qwen_text import QwenTextConditioner
from .learned_aa import LearnedAAConditioner
from .esm import ESMConditioner

__all__ = [
    'BaseConditioner',
    'QwenTextConditioner', 
    'LearnedAAConditioner',
    'ESMConditioner',
]

