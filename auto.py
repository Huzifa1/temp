import torch
import time
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# === Settings ===
model_name = "../models/llama3-3b"  # or another model
offload_dir = "./offload"
prompt = "The future of AI is"

# === Load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === Load model with disk offloading ===
config = AutoConfig.from_pretrained(model_name)

with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model.tie_weights()

model = load_checkpoint_and_dispatch(
    model,
    checkpoint=model_name,
    device_map="auto",
    offload_folder=offload_dir,
    offload_buffers=True,
    dtype=torch.float16  # Or torch.float32 if float16 is not supported
)

# === Tokenize input ===
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# === Generate text + measure time ===
max_new_tokens = 50
start_time = time.time()

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

end_time = time.time()
elapsed_time = end_time - start_time

# === Decode ===
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# === Calculate speed ===
num_generated_tokens = output.shape[1] - inputs["input_ids"].shape[1]
tokens_per_second = num_generated_tokens / elapsed_time

# === Print results ===
print("\nGenerated text:\n", generated_text)
print(f"\nTime taken: {elapsed_time:.2f} seconds")
print(f"Tokens generated: {num_generated_tokens}")
print(f"Throughput: {tokens_per_second:.2f} tokens/second")
