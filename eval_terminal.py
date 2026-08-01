import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from utils import extract_numerical_answer, calculate_math_reward

# ==========================================
# 1. SETUP MODEL & DATASET
# ==========================================
base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = "./adapter/checkpoint-10"  # Update path if needed

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("Loading fine-tuned GRPO adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Sample dataset matching GSM8K format
dataset = [
    {"question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?", "answer": "72"},
    {"question": "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?", "answer": "10"},
    {"question": "Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?", "answer": "5"},
    {"question": "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?", "answer": "42"},
    {"question": "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?", "answer": "624"}
]

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def extract_numeric_answer(text: str) -> str:
    """Extracts the final answer after #### or \\boxed{} or last number."""
    hash_match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    if hash_match:
        return hash_match.group(1).strip()
    
    boxed_match = re.search(r'\\boxed\{([-+]?\d*\.?\d+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    return numbers[-1].strip() if numbers else "N/A"

def generate_response(question_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "Solve the math problem step by step. State the final answer at the end on a new line starting with '#### ' followed by the numeric value."
        },
        {"role": "user", "content": question_text}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_tokens = output[0][prompt_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# ==========================================
# 3. TERMINAL INTERACTIVE LOOP
# ==========================================
print("\n" + "="*50)
print("  MATH QUESTION INTERACTIVE TERMINAL")
print("="*50)
print("Options:")
print(" - Enter an index number (0 to 4) to pick from built-in examples.")
print(" - Type/Paste any custom question.")
print(" - Type 'exit' or 'quit' to stop.")
print("="*50 + "\n")

while True:
    user_input = input("\n[Enter Question / Index / 'exit']: ").strip()
    
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        break
    
    if not user_input:
        continue

    # Handle dataset index selection
    if user_input.isdigit() and int(user_input) < len(dataset):
        idx = int(user_input)
        question = dataset[idx]["question"]
        expected = dataset[idx]["answer"]
        print(f"\n---> Loaded Sample [{idx}]: {question}")
    else:
        question = user_input
        expected = None

    print("\nGenerating model output...")
    output = generate_response(question)
    pred_answer = extract_numeric_answer(output)

    print("\n-------------------- MODEL OUTPUT --------------------")
    print(output)
    print("------------------------------------------------------")
    print(f"Extracted Final Answer: {pred_answer}")
    if expected:
        match = "SUCCESS (Match)" if pred_answer == expected else "FAILED (Mismatch)"
        print(f"Expected Answer       : {expected}")
        print(f"Evaluation            : {match}")