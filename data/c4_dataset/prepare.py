"""
Prepares the C4 dataset for training using HuggingFace datasets.

STREAMING VERSION with:
- DETERMINISTIC RANDOM SAMPLING (1 random doc per 7-doc window)
- RESUMABLE CHECKPOINTS with minimal state
- NO reservoir or complex randomness
- Memory efficient and truly random
"""

import os
num_cpus = "32"
os.environ["OMP_NUM_THREADS"] = num_cpus
os.environ["OPENBLAS_NUM_THREADS"] = num_cpus
os.environ["MKL_NUM_THREADS"] = num_cpus

import json
import random
from datetime import datetime
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset

# number of workers in .map() call
num_proc = 32

# Configuration for token cutoff
BLOCK_SIZE = 1024
MAX_BATCH_SIZE = 16384
TOKENS_PER_STEP_MAX = MAX_BATCH_SIZE * BLOCK_SIZE
TARGET_TRAIN_TOKENS = 100_000_000
SHUFFLE_SEED = 42

# Sampling configuration
WINDOW_SIZE = 700_000  # Process 700k docs at a time (2.1GB text, excellent randomness)
SAMPLE_SIZE = 100_000  # Randomly select 100k from each window (tokenize immediately)
BATCH_SIZE = SAMPLE_SIZE  # No batch accumulation - tokenize as soon as window completes
WRITE_FREQUENCY = 2

print(f"Configuration:")
print(f"  Block size: {BLOCK_SIZE:,}")
print(f"  Target training tokens: {TARGET_TRAIN_TOKENS:,} ({TARGET_TRAIN_TOKENS/1e9:.3f}B)")
print(f"  Window size: {WINDOW_SIZE:,} documents")
print(f"  Sample size: {SAMPLE_SIZE:,} documents (tokenized immediately)")
print(f"  Expected documents: ~{(365_000_000 // WINDOW_SIZE) * SAMPLE_SIZE:,}")
print(f"  Shuffle seed: {SHUFFLE_SEED}")

enc = tiktoken.get_encoding("gpt2")


if __name__ == '__main__':
    start_time = datetime.now()

    # File paths
    train_filename = '100M/train.bin'
    checkpoint_file = '100M/train_checkpoint.json'
    
    # Load C4 dataset
    print("\nLoading C4 dataset (streaming mode)...")
    dataset = load_dataset("allenai/c4", "en", streaming=True)
    print("✓ Dataset loaded in streaming mode")
    
    # Tokenization function
    def process(example):
        try:
            ids = enc.encode_ordinary(example['text'])
            ids.append(enc.eot_token)
            return {'ids': ids, 'len': len(ids)}
        except Exception as e:
            return {'ids': [enc.eot_token], 'len': 1}
    
    # Process validation split
    print("\n" + "="*70)
    print("Processing validation set...")
    print("="*70)
    
    print("Loading 10,000 validation documents...")
    val_list = []
    for i, example in tqdm(enumerate(dataset['validation']), total=10_000, desc="Loading validation docs"):
        if i >= 10_000:
            break
        val_list.append(example)
    
    print(f"✓ Loaded {len(val_list):,} documents")
    val_dataset = Dataset.from_list(val_list)
    
    tokenized_val = val_dataset.map(
        process,
        remove_columns=['text', 'timestamp', 'url'],
        desc="tokenizing validation split",
        num_proc=num_proc,
    )
    
    val_arr_len = np.sum(tokenized_val['len'], dtype=np.uint64)
    val_filename = '100M/val.bin'
    dtype = np.uint16
    val_arr = np.memmap(val_filename, dtype=dtype, mode='w+', shape=(val_arr_len,))
    
    idx = 0
    for doc in tqdm(tokenized_val, desc=f'writing {val_filename}'):
        tokens = np.array(doc['ids'], dtype=dtype)
        val_arr[idx : idx + len(tokens)] = tokens
        idx += len(tokens)
    
    val_arr.flush()
    del val_arr
    
    print(f"✓ Validation set: {val_arr_len:,} tokens ({val_arr_len/1e6:.2f}M)")
    avg_tokens_per_doc = val_arr_len / len(val_list)
    print(f"Average tokens per doc: {avg_tokens_per_doc:.1f}")
    
    # Process training split
    print("\n" + "="*70)
    print(f"Processing training split (target: {TARGET_TRAIN_TOKENS:,} tokens)...")
    print("="*70)
    
    # Check for existing checkpoint
    tokens_written = 0
    total_docs_processed = 0
    docs_seen = 0  # Track how many docs we've streamed through
    checkpoint = {}
    rng_state_data = None
    
    if os.path.exists(checkpoint_file):
        print("Found checkpoint, resuming...")
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            tokens_written = checkpoint['tokens_written']
            total_docs_processed = checkpoint['total_docs_processed']
            docs_seen = checkpoint['docs_seen']
            rng_state_data = checkpoint.get('rng_state')
        print(f"Resuming from: {tokens_written:,} tokens")
        print(f"  Documents processed: {total_docs_processed:,}")
        print(f"  Documents streamed: {docs_seen:,}")
        train_arr = np.memmap(train_filename, dtype=dtype, mode='r+')
    else:
        print("Starting fresh...")
        print(f"Sampling: 1 random document per {WINDOW_SIZE}-doc window from C4 stream...")
        train_arr = np.memmap(train_filename, dtype=dtype, mode='w+', shape=(TARGET_TRAIN_TOKENS,))
        # Initialize RNG for fresh start
        random.seed(SHUFFLE_SEED)
    
    print(f"Estimated documents needed: ~{int(TARGET_TRAIN_TOKENS / avg_tokens_per_doc):,}")
    print(f"Starting sampling...\n")
    
    # On resume, restore RNG to exact state (no regeneration needed)
    if rng_state_data is not None:
        print(f"Restoring RNG state...")
        # RNG state from random.getstate() is (version, tuple, index)
        # JSON converts the tuple to list, so convert back
        rng_version, rng_tuple_list, rng_index = rng_state_data
        random.setstate((rng_version, tuple(rng_tuple_list), rng_index))
    
    batches_written = 0
    
    print(f"Writing tokens (exactly {TARGET_TRAIN_TOKENS:,})...")
    
    with tqdm(total=TARGET_TRAIN_TOKENS, desc="Writing tokens", unit="tokens", initial=tokens_written) as pbar_tokens:
        window_buffer = []
        window_count = 0
        
        for example in dataset['train']:
            docs_seen += 1
            
            # Skip already-processed documents on resume
            if docs_seen <= checkpoint.get('docs_seen', 0):
                continue
            
            # Add document to current window
            window_buffer.append(example)
            
            # When window is full, sample and tokenize immediately
            if len(window_buffer) == WINDOW_SIZE:
                window_count += 1
                
                # Randomly select SAMPLE_SIZE documents from window
                selected_indices = random.sample(range(WINDOW_SIZE), min(SAMPLE_SIZE, WINDOW_SIZE))
                sampled_docs = [window_buffer[idx] for idx in selected_indices]
                total_docs_processed += len(sampled_docs)
                
                window_buffer = []
                
                # Tokenize immediately (no batch accumulation)
                with tqdm(total=SAMPLE_SIZE, desc=f"Tokenizing window {window_count}", unit="docs", position=1, leave=False) as pbar_tokenize:
                    batch_dataset = Dataset.from_list(sampled_docs)
                    tokenized_batch = batch_dataset.map(
                        process,
                        remove_columns=['text', 'timestamp', 'url'],
                        desc=f"tokenizing",
                        num_proc=num_proc,
                    )
                    
                    # Write tokens immediately as they're tokenized
                    for doc in tokenized_batch:
                        doc_tokens = np.array(doc['ids'], dtype=dtype)
                        remaining = TARGET_TRAIN_TOKENS - tokens_written
                        
                        if remaining <= 0:
                            break
                        
                        tokens_to_write = min(len(doc_tokens), remaining)
                        train_arr[tokens_written : tokens_written + tokens_to_write] = doc_tokens[:tokens_to_write]
                        tokens_written += tokens_to_write
                        pbar_tokens.update(tokens_to_write)
                        pbar_tokenize.update(1)
                    
                    batches_written += 1
                    
                    # Save checkpoint (including RNG state for perfect resumption)
                    rng_state = random.getstate()
                    # Convert RNG state to JSON-serializable format
                    rng_state_serializable = (rng_state[0], list(rng_state[1]), rng_state[2])
                    
                    with open(checkpoint_file, 'w') as f:
                        json.dump({
                            'tokens_written': tokens_written,
                            'total_docs_processed': total_docs_processed,
                            'docs_seen': docs_seen,
                            'batches_written': batches_written,
                            'rng_state': rng_state_serializable,
                            'timestamp': datetime.now().isoformat()
                        }, f)
                    
                    # Periodic flush
                    if batches_written % WRITE_FREQUENCY == 0:
                        train_arr.flush()
                
                if tokens_written >= TARGET_TRAIN_TOKENS:
                    break
            
            if tokens_written >= TARGET_TRAIN_TOKENS:
                break
    
    train_arr.flush()
    del train_arr
    
    # Verify target
    assert tokens_written == TARGET_TRAIN_TOKENS, \
        f"Expected {TARGET_TRAIN_TOKENS:,} tokens, wrote {tokens_written:,}"
    
    actual_size = os.path.getsize(train_filename)
    expected_size = tokens_written * dtype().itemsize
    assert actual_size == expected_size, \
        f"File size mismatch: expected {expected_size:,}, got {actual_size:,}"
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Calculate batch statistics
    batch_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    batch_stats = {}
    for bs in batch_sizes:
        tokens_per_step = bs * BLOCK_SIZE
        step_count = tokens_written // tokens_per_step
        if step_count > 0:
            batch_stats[bs] = {
                'tokens_per_step': tokens_per_step,
                'num_steps': step_count
            }
    
    # Save metadata
    metadata = {
        'dataset_info': {
            'source': 'allenai/c4',
            'subset': 'en',
            'streaming': True,
            'shuffle_seed': SHUFFLE_SEED,
            'sampling_method': 'random_sampling_within_windows',
            'window_size': WINDOW_SIZE,
            'sample_size': SAMPLE_SIZE,
            'expected_sample_size': f"~{(365_000_000 // WINDOW_SIZE) * SAMPLE_SIZE:,}",
            'created_at': start_time.isoformat(),
            'processing_duration_seconds': duration,
        },
        'tokenization': {
            'tokenizer': 'gpt2',
            'vocab_size': enc.n_vocab,
            'eot_token': enc.eot_token,
        },
        'training_data': {
            'filename': train_filename,
            'num_tokens': int(tokens_written),
            'num_documents_processed': total_docs_processed,
            'avg_tokens_per_doc': float(avg_tokens_per_doc),
            'file_size_bytes': int(actual_size),
            'file_size_gb': round(actual_size / 1e9, 2),
        },
        'validation_data': {
            'filename': val_filename,
            'num_tokens': int(val_arr_len),
            'num_documents': len(val_list),
            'file_size_bytes': int(os.path.getsize(val_filename)),
            'file_size_mb': round(os.path.getsize(val_filename) / 1e6, 2),
        },
        'configuration': {
            'block_size': BLOCK_SIZE,
            'max_batch_size': MAX_BATCH_SIZE,
            'tokens_per_step_max': TOKENS_PER_STEP_MAX,
            'target_train_tokens': TARGET_TRAIN_TOKENS,
            'dtype': str(dtype),
            'num_proc': num_proc,
            'batch_size': BATCH_SIZE,
            'window_size': WINDOW_SIZE,
            'sample_size': SAMPLE_SIZE,
        },
        'batch_size_info': batch_stats,
    }
    
    metadata_filename = '100M/dataset_info.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Write human-readable info
    info_filename = '100M/README.txt'
    with open(info_filename, 'w') as f:
        f.write("="*70 + "\n")
        f.write("C4 Dataset Preparation Summary\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Created: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Processing time: {duration/60:.1f} minutes\n\n")
        
        f.write("SOURCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Dataset: allenai/c4 (English subset, ~365M documents)\n")
        f.write(f"Streaming mode: Yes\n")
        f.write(f"Sampling method: {SAMPLE_SIZE:,} random documents per {WINDOW_SIZE:,}-doc window\n")
        f.write(f"Tokenization: Immediate (no batch accumulation)\n")
        f.write(f"Shuffle seed: {SHUFFLE_SEED}\n")
        f.write(f"Expected sample size: ~{(365_000_000 // WINDOW_SIZE) * SAMPLE_SIZE:,} documents\n\n")
        
        f.write("TOKENIZATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Tokenizer: GPT-2 BPE (tiktoken)\n")
        f.write(f"Vocabulary size: {enc.n_vocab:,}\n")
        f.write(f"EOT token: {enc.eot_token}\n\n")
        
        f.write("TRAINING DATA\n")
        f.write("-" * 70 + "\n")
        f.write(f"File: {train_filename}\n")
        f.write(f"Tokens: {tokens_written:,} ({tokens_written/1e9:.3f}B)\n")
        f.write(f"Documents processed: {total_docs_processed:,}\n")
        f.write(f"Avg tokens/doc: {avg_tokens_per_doc:.1f}\n")
        f.write(f"File size: {actual_size/1e9:.2f} GB\n")
        f.write(f"Data type: uint16\n\n")
        
        f.write("VALIDATION DATA\n")
        f.write("-" * 70 + "\n")
        f.write(f"File: {val_filename}\n")
        f.write(f"Tokens: {val_arr_len:,} ({val_arr_len/1e6:.2f}M)\n")
        f.write(f"Documents: {len(val_list):,}\n")
        f.write(f"File size: {os.path.getsize(val_filename)/1e6:.2f} MB\n\n")
        
        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Block size: {BLOCK_SIZE}\n")
        f.write(f"Max batch size: {MAX_BATCH_SIZE:,}\n")
        f.write(f"Tokens per step (max): {TOKENS_PER_STEP_MAX:,}\n")
        f.write(f"Batch size: {BATCH_SIZE:,}\n\n")
        
        f.write("STEPS FOR DIFFERENT BATCH SIZES\n")
        f.write("-" * 70 + "\n")
        for bs, stats in batch_stats.items():
            f.write(f"Batch size {bs:>5}: {stats['num_steps']:>7,} steps ")
            f.write(f"({stats['tokens_per_step']:>10,} tokens/step)\n")
        f.write("\n")
        
        f.write("HOW TO LOAD\n")
        f.write("-" * 70 + "\n")
        f.write("Python:\n")
        f.write("  import numpy as np\n")
        f.write(f"  train_data = np.memmap('{train_filename}', dtype=np.uint16, mode='r')\n")
        f.write(f"  val_data = np.memmap('{val_filename}', dtype=np.uint16, mode='r')\n\n")
        
        f.write("NOTES\n")
        f.write("-" * 70 + "\n")
        f.write(f"- Sampling is deterministic: same seed always produces same samples\n")
        f.write(f"- Truly random within each window: {SAMPLE_SIZE:,} docs randomly selected from each {WINDOW_SIZE:,}-doc window\n")
        f.write(f"- Tokenization is immediate: no batch accumulation (better CPU utilization)\n")
        f.write(f"- Memory efficient: never holds more than {WINDOW_SIZE:,} documents in memory\n")
        f.write(f"- Resumable: if interrupted, simply run again to resume from checkpoint\n")
        f.write("- Files are binary (uint16), 2 bytes per token\n")
        f.write("- Last document may be truncated to hit exact token count\n")
    
    # Clean up checkpoint when done
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"\n✓ Checkpoint cleaned up")
    
    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"{'='*60}")
    print(f"Training tokens: {tokens_written:,} ({tokens_written/1e9:.3f}B)")
    print(f"Documents processed: {total_docs_processed:,}")
    print(f"Processing time: {duration/60:.1f} minutes")
    print(f"{'='*60}")
    
    if tokens_written >= TOKENS_PER_STEP_MAX:
        steps = tokens_written // TOKENS_PER_STEP_MAX
        print(f"\nAt max batch size {MAX_BATCH_SIZE:,} (block size {BLOCK_SIZE}):")
        print(f"  • Tokens per step: {TOKENS_PER_STEP_MAX:,}")
        print(f"  • Total steps: {steps:,}")
        
        print(f"\nSteps for other batch sizes:")
        for bs in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]:
            if bs <= MAX_BATCH_SIZE:
                tokens_per_step = bs * BLOCK_SIZE
                step_count = tokens_written // tokens_per_step
                if step_count > 0:
                    print(f"  • Batch {bs:>5}: {step_count:>7,} steps")
    print(f"{'='*60}")
    
    print(f"\nFiles created:")
    print(f"  {train_filename}: {os.path.getsize(train_filename) / 1e9:.2f} GB")
    print(f"  {val_filename}: {os.path.getsize(val_filename) / 1e6:.2f} MB")
    print(f"  {metadata_filename}")
    print(f"  {info_filename}")
    
    print(f"\nDataset metadata saved to:")
    print(f"  {metadata_filename} (JSON)")
    print(f"  {info_filename} (human-readable)")