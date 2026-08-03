import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from environment import PythonEnvironment
from reward import calculate_reward
import re

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # adjust to your base model path/id
adapter_path = "./adapter/checkpoint-10"            # or "./adapter" depending on where adapter_config.json is saved

# 1. Define 4-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 2. Load Base Model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# 3. Load trained LoRA adapters onto the quantized base model
print("Loading fine-tuned GRPO adapters...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("Model successfully loaded and ready for inference!")


# ==========================================
# DEMO EVALUATION
# ==========================================
env = PythonEnvironment()

question = "Calculate 25*12 using Python. Return only executable Python code."

# Format prompt using Qwen Chat Template
messages = [
    {
        "role": "system", 
        "content": "You are a code generator. Output only executable Python code directly without any markdown formatting or backticks."
    },
    {"role": "user", "content": question}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    # Pass **inputs (unpacks input_ids & attention_mask dictionary)
    output = model.generate(
        **inputs,
        max_new_tokens=384,
        do_sample=True,  # Note: do_sample=False ignores temperature parameter
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )

# Slice output using inputs["input_ids"] shape
prompt_length = inputs["input_ids"].shape[-1]
generated_tokens = output[0][prompt_length:]
answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

print("\n====================")
print("MODEL OUTPUT AFTER GRPO")
print("====================")
print(answer)

execution_result = env.run(answer)

print("\n====================")
print("EXECUTION RESULT")
print("====================")
print(execution_result)

reward = calculate_reward(execution_result, "300")

print("\n====================")
print("FINAL REWARD")
print("====================")
print(reward)