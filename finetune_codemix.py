import torch
import json
import os
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import gc
from huggingface_hub import login
login(token="")

# Make sure to install peft first:
# !pip install peft

from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# Set RAM disk for caching
os.environ['TRANSFORMERS_CACHE'] = '/mnt/ramdisk/huggingface_cache'
os.environ['HF_HOME'] = '/mnt/ramdisk/huggingface_home'

# Force garbage collection
gc.collect()
torch.cuda.empty_cache()

# Load your data
def load_json_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Flatten the dialogue into a single prompt-response per sample
    examples = []
    for item in data:
        messages = item["messages"]
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        prompt = f"<|user|>\n{user_msg}\n<|assistant|>\n{assistant_msg}"
        examples.append({"text": prompt})
    return Dataset.from_list(examples)

# Load model with 4-bit quantization
try:
    import bitsandbytes as bnb
except ImportError:
    raise ImportError("Please install bitsandbytes: pip install bitsandbytes")

# Load the model with 4-bit quantization
model_name = "mistralai/Ministral-8B-Instruct-2410"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,
    device_map="auto",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    use_cache=False,
    cache_dir='/mnt/ramdisk/model_cache'  # Use RAM disk for model cache
)

# Prepare the model for training
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply LoRA adapter
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # This shows the parameter reduction

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir='/mnt/ramdisk/model_cache'  # Use RAM disk for tokenizer cache
)
tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess datasets
train_dataset = load_json_dataset("train.json")
eval_dataset = load_json_dataset("test.json")

# Keep sequence length manageable
max_length = 512

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Process datasets with multiprocessing to reduce memory pressure
train_tokenized = train_dataset.map(
    lambda x: tokenizer(
        x["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    ),
    batched=True,
    remove_columns=["text"]
)

eval_tokenized = eval_dataset.map(
    lambda x: tokenizer(
        x["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length
    ),
    batched=True,
    remove_columns=["text"]
)

# Add labels for casual language modeling
train_tokenized = train_tokenized.map(
    lambda x: {"labels": x["input_ids"].copy()},
    batched=True
)

eval_tokenized = eval_tokenized.map(
    lambda x: {"labels": x["input_ids"].copy()},
    batched=True
)

# Define persistent storage path for final model
PERSISTENT_OUTPUT_DIR = "./disk_results_lora"  # This will be on regular disk storage
TEMP_OUTPUT_DIR = "/mnt/ramdisk/temp_results"  # Temporary checkpoints in RAM
TEMP_FINAL_DIR = "/mnt/ramdisk/final_model"

# Define training arguments
training_args = TrainingArguments(
    output_dir=TEMP_OUTPUT_DIR,  # Save intermediate checkpoints to RAM
    num_train_epochs=15,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    logging_dir='/mnt/ramdisk/logs',  # Logs in RAM
    logging_steps=100,
    save_steps=500,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    load_best_model_at_end=True,
    save_total_limit=1,
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    learning_rate=2e-4
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    tokenizer=tokenizer,
)

# Free memory before training
gc.collect()
torch.cuda.empty_cache()

# Start training
trainer.train()

# Make sure the persistent directory exists
os.makedirs(PERSISTENT_OUTPUT_DIR, exist_ok=True)

# Save the adapter weights to persistent storage (disk)
model.save_pretrained(PERSISTENT_OUTPUT_DIR)
model.save_pretrained(TEMP_FINAL_DIR)
tokenizer.save_pretrained(PERSISTENT_OUTPUT_DIR)
tokenizer.save_pretrained(TEMP_FINAL_DIR)

print(f"Model successfully saved to persistent storage at: {PERSISTENT_OUTPUT_DIR} and {TEMP_FINAL_DIR}")

# Optional: Clean up RAM disk after training
import shutil
try:
    shutil.rmtree("/mnt/ramdisk/temp_results")
    print("Cleaned up temporary training files from RAM disk")
except:
    print("Note: Could not clean up temporary files from RAM disk")
