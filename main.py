import datasets
import numpy as np
import os
import torch
import transformers
import wandb
from dataclasses import dataclass, field
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback

import warnings
warnings.filterwarnings("ignore")

WANDB_API_KEY = "REPLACE_THIS_WITH_YOUR_WANDB_API_KEY"
wandb.login(key=WANDB_API_KEY)

os.environ["WANDB_PROJECT"] = "pretraining"   # Name of the Weights & Biases project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # Log all model checkpoints

# STEP 1 - Download a pretraining dataset (training data) from Hugging Face
# =========================================================================

pretraining_dataset = datasets.load_dataset(
    "upstage/Pretraining_Dataset",
    split="train"
)

# Only work with the text column
pretraining_dataset = pretraining_dataset.select_columns(
    ["text"]
)

# Save the dataset to disk
file_path = "./data/pretraining_dataset.parquet"
pretraining_dataset.to_parquet(file_path)


# STEP 2 - Packaging data for pretraining
# =======================================

# Split the dataset into 10 pieces (shards)
pretraining_dataset = pretraining_dataset.shard(num_shards=10, index=0)

# Load the tokenizer
model_path_or_name = "upstage/SOLAR-10.7B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name, 
    use_fast=False
)

# Helper function
def tokenization(example):
    # Tokenize
    tokens = tokenizer.tokenize(example["text"])
    # Convert tokens to ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    # Add <bos>, <eos> tokens to the front and back of tokens_ids 
    # bos means "begin of sequence", eos means "end of sequence"
    token_ids = [
        tokenizer.bos_token_id] \
        + token_ids \
        + [tokenizer.eos_token_id
    ]
    example["input_ids"] = token_ids
    return example

# Tokenize all examples in the pretraining dataset
pretraining_dataset = pretraining_dataset.map(tokenization, load_from_cache_file=False)

# Concatenate input_ids for all examples into a single list
input_ids = np.concatenate(pretraining_dataset["input_ids"])

max_seq_length = 32
total_length = len(input_ids) - len(input_ids) % max_seq_length
# Discard extra tokens from the end of the list, so that the number of tokens is exactly divisible by max_seq_length
input_ids = input_ids[:total_length]
input_ids_reshaped = input_ids.reshape(-1, max_seq_length).astype(np.int32)
input_ids_reshaped.shape

# Convert to Hugging Face dataset
input_ids_list = input_ids_reshaped.tolist()
packaged_pretrain_dataset = datasets.Dataset.from_dict(
    {"input_ids": input_ids_list}
)

# Save the packaged dataset to disk
packaged_pretrain_dataset.to_parquet("./data/packaged_pretrain_dataset.parquet")


# STEP 3 - Load the model and continue the (pre)training of an exisitng model
# ===========================================================================

# Load the model (change from "cpu" to "auto", if the system has a GPU)
pretrained_model = AutoModelForCausalLM.from_pretrained(
    "upstage/TinySolar-248m-4k",
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

# Load the dataset
class CustomDataset(Dataset):
    def __init__(self, args, split="train"):
        """Initializes the custom dataset object."""
        self.args = args
        self.dataset = datasets.load_dataset(
            "parquet",
            data_files=args.dataset_name,
            split=split
        )

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample from the dataset 
        at the specified index
        """
        # Convert the lists to a LongTensor for PyTorch
        input_ids = torch.LongTensor(self.dataset[idx]["input_ids"])
        labels = torch.LongTensor(self.dataset[idx]["input_ids"])

        # Return the sample as a dictionary
        return {"input_ids": input_ids, "labels": labels}
    
# Configure training arguments
@dataclass
class CustomArguments(transformers.TrainingArguments):
    dataset_name: str = field(                           # Dataset configuration
        default="./data/packaged_pretrain_dataset.parquet")
    num_proc: int = field(default=1)                     # Number of subprocesses for data preprocessing
    max_seq_length: int = field(default=32)              # Maximum sequence length

    # Core training configurations
    seed: int = field(default=0)                         # Random seed for initialization, ensuring reproducibility
    optim: str = field(default="adamw_torch")            # Use the AdamW optimizer from PyTorch
    max_steps: int = field(default=30)                   # Number of maximum training steps
    per_device_train_batch_size: int = field(default=2)  # Batch size per device during training

    # Other training configurations
    learning_rate: float = field(default=5e-5)           # Initial learning rate for the optimizer
    weight_decay: float = field(default=0)               # Weight decay
    warmup_steps: int = field(default=10)                # Number of steps for the learning rate warmup phase
    lr_scheduler_type: str = field(default="linear")     # Type of learning rate scheduler
    gradient_checkpointing: bool = field(default=True)   # Enable gradient checkpointing to save memory
    # The following setting for subprocesses is machine specific. It might speed things up, or it might crash
    # dataloader_num_workers: int = field(default=2)     # Number of subprocesses for data loading
    bf16: bool = field(default=True)                     # Use bfloat16 precision for training on supported hardware
    gradient_accumulation_steps: int = field(default=1)  # Number of steps to accumulate gradients before updating model weights
    
    # Logging configuration
    logging_steps: int = field(default=1)                # Frequency of logging training information
    report_to: str = field(default="wandb")              # Destination for logging (e.g., WandB, TensorBoard)

    # Saving configuration
    # save_strategy: str = field(default="steps")        # Can be replaced with "epoch"
    # save_steps: int = field(default=3)                 # Frequency of saving training checkpoint
    # save_total_limit: int = field(default=2)           # The total number of checkpoints to be saved

# Parse the custom arguments and set the output directory
parser = transformers.HfArgumentParser(CustomArguments)
args, = parser.parse_args_into_dataclasses(
    args=["--output_dir", "output"]
)

# Set up the training dataset
train_dataset = CustomDataset(args=args)

# Define a custom callback to log the loss values
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.logs.append(logs)

    def __init__(self):
        self.logs = []

# Initialize the callback
loss_logging_callback = LossLoggingCallback()

# Create an instance of the Hugging Face Trainer object from the transformers library
trainer = Trainer(
    model=pretrained_model, 
    args=args, 
    train_dataset=train_dataset, 
    eval_dataset=None,
    callbacks=[loss_logging_callback] 
)

# Initialize the training run
trainer.train()

# Save the (pre)trained model to disk
trainer.save_model("./output/Pretrained_TinySolar-248m-4k")
