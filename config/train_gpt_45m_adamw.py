# I/O
out_dir = 'out_adamw_blockSize1024_vocabSize50304' # directory with all checkpoints
eval_interval = 10 # for checkpointing and wandb logging
init_from = 'scratch' # 'scratch' or 'resume'. if you resume remember to set the max_iters and lr_decay_iters to the level you want and also provide the checkpoint filename
# resume_ckpt_fname = "best_step(0000030)_tokens(4.92e5)_tloss9.012_vloss9.056_ckpt.pt"
input_seed = 1337

# wandb logging


# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 2 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024 # should set to 1024 based on sammie's default params. The maximum number of tokens the model can "see" at once (per sequence). It's the model's context window.

# model -- sammie's params. DON'T CHANGE FOR 45M MODEL !!!!!
vocab_size = 50304
n_layer = 8
n_head = 12
n_embd = 480
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+

# adamw optimizer
ignore_fused = True
learning_rate = 3e-4 # max learning rate
max_iters = 30 # total number of training iterations

# learning rate decay settings
lr_decay_iters = 30 # should be ~= max_iters per Chinchilla

# DDP settings


# system
