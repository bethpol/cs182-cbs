======================================================================
C4 Dataset Preparation Summary
======================================================================

Created: 2025-11-03 15:02:00
Processing time: 6.1 minutes

SOURCE
----------------------------------------------------------------------
Dataset: allenai/c4 (English subset, ~365M documents)
Streaming mode: Yes
Sampling method: 100,000 random documents per 700,000-doc window
Tokenization: Immediate (no batch accumulation)
Shuffle seed: 42
Expected sample size: ~52,100,000 documents

TOKENIZATION
----------------------------------------------------------------------
Tokenizer: GPT-2 BPE (tiktoken)
Vocabulary size: 50,257
EOT token: 50256

TRAINING DATA
----------------------------------------------------------------------
File: 100M/train.bin
Tokens: 100,000,000 (0.100B)
Documents processed: 300,000
Avg tokens/doc: 478.0
File size: 0.20 GB
Data type: uint16

VALIDATION DATA
----------------------------------------------------------------------
File: 100M/val.bin
Tokens: 4,779,946 (4.78M)
Documents: 10,000
File size: 9.56 MB

TRAINING CONFIGURATION
----------------------------------------------------------------------
Block size: 1024
Max batch size: 16,384
Tokens per step (max): 16,777,216
Batch size: 100,000

STEPS FOR DIFFERENT BATCH SIZES
----------------------------------------------------------------------
Batch size    64:   1,525 steps (    65,536 tokens/step)
Batch size   128:     762 steps (   131,072 tokens/step)
Batch size   256:     381 steps (   262,144 tokens/step)
Batch size   512:     190 steps (   524,288 tokens/step)
Batch size  1024:      95 steps ( 1,048,576 tokens/step)
Batch size  2048:      47 steps ( 2,097,152 tokens/step)
Batch size  4096:      23 steps ( 4,194,304 tokens/step)
Batch size  8192:      11 steps ( 8,388,608 tokens/step)
Batch size 16384:       5 steps (16,777,216 tokens/step)

HOW TO LOAD
----------------------------------------------------------------------
Python:
  import numpy as np
  train_data = np.memmap('100M/train.bin', dtype=np.uint16, mode='r')
  val_data = np.memmap('100M/val.bin', dtype=np.uint16, mode='r')

NOTES
----------------------------------------------------------------------
- Sampling is deterministic: same seed always produces same samples
- Truly random within each window: 100,000 docs randomly selected from each 700,000-doc window
- Tokenization is immediate: no batch accumulation (better CPU utilization)
- Memory efficient: never holds more than 700,000 documents in memory
- Resumable: if interrupted, simply run again to resume from checkpoint
- Files are binary (uint16), 2 bytes per token
- Last document may be truncated to hit exact token count
