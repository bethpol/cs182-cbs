import torch
import torch.distributed as dist
from typing import Tuple

class TokenCounter:
    def __init__(
        self,
        batch_size: int,
        block_size: int,
        world_size: int = 1,
        checkpoint_interval_tokens: int = 500_000_000,
        accumulation_steps: int = 1,  # NEW
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.world_size = world_size
        self.checkpoint_interval_tokens = checkpoint_interval_tokens
        self.accumulation_steps = accumulation_steps
        
        # Tokens processed per forward/backward pass (not per update)
        self.tokens_per_forward = batch_size * (block_size - 1) * world_size
        
        # Tokens processed per weight update
        self.tokens_per_update = self.tokens_per_forward * accumulation_steps
        
        self.tokens_seen = 0
        self.tokens_at_last_checkpoint = 0
    
    def step(self, is_update_step: bool = True) -> None:
        """
        Count tokens.
        
        Args:
            is_update_step: True if this is a weight update step
                           (after accumulation_steps forward passes)
        """
        if is_update_step:
            # Count all accumulated tokens in this update
            self.tokens_seen += self.tokens_per_update
        else:
            # Count individual forward pass (for intermediate logging)
            self.tokens_seen += self.tokens_per_forward