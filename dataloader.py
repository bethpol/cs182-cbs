import os
import numpy as np
import torch
from typing import Dict, Tuple


class CBSDataLoader:
    """
    Efficient sequential dataloader with direct token position tracking.
    
    Key optimization: store token_pos directly, no line/token conversions in get_batch
    
    On init:
      - Load train.bin as memmap
      - Set starting token position (checkpoint or branch offset)
      - Track current token position
    
    On get_batch:
      - Read sequences starting from current token position
      - Advance token counter
      - Support checkpointing/resume
    
    IMPORTANT: checkpoint_token_pos is the ABSOLUTE position in the file where we start.
    branch_seed applies an OFFSET on top of that for branching experiments.
    """
    
    def __init__(
        self,
        data_file: str,
        block_size: int = 1024,
        total_tokens: int = None,
        checkpoint_token_pos: int = 0,
        branch_seed: int = -1,
        branch_window_size_tokens: int = None,
        device: str = 'cuda',
        num_gpus: int = 1,
        gpu_rank: int = 0,
    ):
        """
        Initialize dataloader with direct token position tracking.
        
        Args:
            data_file: Path to train.bin
            block_size: Tokens per sequence
            total_tokens: Total tokens in file (auto-detected if None)
            checkpoint_token_pos: ABSOLUTE token position in file to start reading from (0-indexed)
                                 This is where you resume from after a checkpoint
            branch_seed: Offset seed for branching
                        -1 = no offset (use checkpoint_token_pos as-is for resuming)
                        0+ = offset by (seed * window_size_tokens) for branching experiments
            branch_window_size_tokens: Size of branch window in tokens (REQUIRED - no auto-calculation)
            device: 'cuda' or 'cpu'
            num_gpus: Number of GPUs
            gpu_rank: This GPU's rank
        """
        self.data_file = data_file
        self.block_size = block_size
        self.device = device
        self.checkpoint_token_pos = checkpoint_token_pos
        self.branch_seed = branch_seed
        self.num_gpus = num_gpus
        self.gpu_rank = gpu_rank
        
        if gpu_rank < 0 or gpu_rank >= num_gpus:
            raise ValueError(f"gpu_rank ({gpu_rank}) must be in range [0, {num_gpus})")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        self.data = np.memmap(data_file, dtype=np.uint16, mode='r')
        
        if total_tokens is None:
            total_tokens = len(self.data)
        elif total_tokens != len(self.data):
            raise ValueError(
                f"total_tokens ({total_tokens}) != actual data size ({len(self.data)})"
            )
        
        self.total_tokens = total_tokens
        
        # Window size must be explicitly specified for branching
        if branch_window_size_tokens is None:
            raise ValueError(
                "branch_window_size_tokens must be explicitly specified. "
                "No auto-calculation - you control the window size for branching."
            )
        self.branch_window_size_tokens = branch_window_size_tokens
        
        # Calculate actual start position
        # checkpoint_token_pos is the ABSOLUTE token position where we START reading
        # branch_seed applies an OFFSET on top of checkpoint_token_pos for branching
        if branch_seed < 0:
            # No offset - use checkpoint position as-is (this is used for resuming)
            self.start_token_pos = checkpoint_token_pos
            mode_desc = "sequential (no offset)"
        else:
            # Apply offset for branching experiments
            offset = branch_seed * self.branch_window_size_tokens
            self.start_token_pos = checkpoint_token_pos + offset
            mode_desc = f"offset by {offset:,} tokens (seed={branch_seed} × window={self.branch_window_size_tokens:,})"
        
        if self.start_token_pos >= total_tokens:
            raise ValueError(
                f"Start position {self.start_token_pos:,} >= total tokens {total_tokens:,}"
            )
        
        # Calculate available sequences
        self.available_tokens = total_tokens - self.start_token_pos
        self.available_sequences = self.available_tokens // block_size
        self.sequences_per_gpu = (self.available_sequences + num_gpus - 1) // num_gpus
        self.stride = num_gpus
        
        gpu_info = f"Multi-GPU: {num_gpus} GPUs, rank={gpu_rank}" if num_gpus > 1 else "Single GPU"
        
        print(f"\n[CBSDataLoader]")
        print(f"  Data file: {data_file}")
        print(f"  Block size: {block_size} tokens")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Checkpoint position: {checkpoint_token_pos:,}")
        print(f"  Mode: {mode_desc}")
        print(f"  Start position: {self.start_token_pos:,}")
        print(f"  Available tokens: {self.available_tokens:,}")
        print(f"  {gpu_info}")
        print(f"  Sequences per GPU: {self.sequences_per_gpu:,}")
        print()
        
        self.current_sequence_idx = 0
    
    def get_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next batch of sequences."""
        if self.current_sequence_idx + batch_size > self.sequences_per_gpu:
            raise StopIteration(
                f"GPU {self.gpu_rank}: Reached end at sequence {self.current_sequence_idx}. "
                f"Total: {self.sequences_per_gpu}"
            )
        
        x_list = []
        y_list = []
        
        for i in range(batch_size):
            # Direct token position: no conversion overhead
            global_seq_idx = self.gpu_rank + (self.current_sequence_idx + i) * self.stride
            token_pos = self.start_token_pos + global_seq_idx * self.block_size
            
            # Read and process
            sequence = self.data[token_pos:token_pos + self.block_size].astype(np.int64)
            x_list.append(torch.from_numpy(sequence[:-1]).long())
            y_list.append(torch.from_numpy(sequence[1:]).long())
        
        x = torch.stack(x_list)
        y = torch.stack(y_list)
        
        if self.device != 'cpu':
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x = x.to(self.device)
            y = y.to(self.device)
        
        self.current_sequence_idx += batch_size
        return x, y
    
    def get_state(self) -> Dict:
        """
        Checkpoint: return current token position.
        This is the ABSOLUTE position in the file where we currently are.
        """
        # current_sequence_idx is relative to start_token_pos
        # To get absolute position, we add start_token_pos
        absolute_token_pos = self.start_token_pos + self.current_sequence_idx * self.block_size
        return {'token_pos': absolute_token_pos}
    
    def load_state(self, state: Dict):
        """
        Resume from checkpoint.
        The checkpoint contains the ABSOLUTE token position in the file.
        """
        absolute_token_pos = state['token_pos']
        
        # Verify alignment
        if absolute_token_pos % self.block_size != 0:
            raise ValueError(f"Checkpoint token_pos {absolute_token_pos} not aligned to block_size {self.block_size}")
        
        # Convert absolute position to relative position from start_token_pos
        relative_pos = absolute_token_pos - self.start_token_pos
        
        # Verify we're in the right range
        if relative_pos < 0:
            raise ValueError(
                f"Checkpoint position {absolute_token_pos} is before start position {self.start_token_pos}"
            )
        if relative_pos > self.available_tokens:
            raise ValueError(
                f"Checkpoint position {absolute_token_pos} is beyond available tokens"
            )
        
        self.current_sequence_idx = relative_pos // self.block_size
        print(f"[CBSDataLoader] GPU {self.gpu_rank}: Loaded state, absolute_token_pos={absolute_token_pos}, relative_sequence_idx={self.current_sequence_idx}")
    
    def get_progress(self) -> Tuple[int, int]:
        """Returns (sequences_seen, total_sequences_available)."""
        return self.current_sequence_idx, self.sequences_per_gpu
    
    def get_tokens_seen(self) -> int:
        """Total tokens processed so far (relative to start position)."""
        return self.current_sequence_idx * self.block_size


def create_dataloader(
    data_file: str,
    block_size: int = 1024,
    total_tokens: int = None,
    checkpoint_token_pos: int = 0,
    branch_seed: int = -1,
    branch_window_size_tokens: int = None,
    device: str = 'cuda',
    num_gpus: int = 1,
    gpu_rank: int = 0,
) -> CBSDataLoader:
    """Create a CBS sequential dataloader."""
    return CBSDataLoader(
        data_file=data_file,
        block_size=block_size,
        total_tokens=total_tokens,
        checkpoint_token_pos=checkpoint_token_pos,
        branch_seed=branch_seed,
        branch_window_size_tokens=branch_window_size_tokens,
        device=device,
        num_gpus=num_gpus,
        gpu_rank=gpu_rank,
    )