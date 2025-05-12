from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

# Path to adapter
adapter_path = "/home/isfcr/codemix/final_model"

# Load LoRA adapter config
config = PeftConfig.from_pretrained(adapter_path)
base_model_name = config.base_model_name_or_path

# Load tokenizer from base model (not adapter!)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # Just in case

# Load base model and apply LoRA weights
model = AutoModelForCausalLM.from_pretrained(
    base_model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)

# Set to eval mode
model.eval()

# Kanglish question
question = "India da current prime minister yaaru?"

# Prompt format exactly as in training
prompt = f"<|system|>\nYou are a helpful assistant that replies in code-mixed Kanglish.\n<|user|>\n{question}\n<|assistant|>\n"

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode and clean up generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
answer = generated_text.split("<|assistant|>\n")[-1].strip()

print("Answer:", answer)
