import json
import re
from model import load_model

# 1. Load model and tokenizer
model, tokenizer = load_model()

# 2. Load evaluation dataset (requires import json)
with open("data/gsm8k.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def extract_answer_num(text: str) -> str:
    """Extracts the final numeric value after #### or \\boxed{}."""
    # Look for #### <number>
    hash_match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    if hash_match:
        return hash_match.group(1).strip()
    
    # Fallback: Find the last trailing number in the generated text
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    return numbers[-1].strip() if numbers else ""

correct = 0
total = 20

print(f"Running evaluation on first {total} samples...\n")

for i, item in enumerate(data[:total]):
    question = item["question"]
    # Parse ground truth answer from string (e.g. "... #### 72" -> "72")
    expected_answer = extract_answer_num(item["answer"])

    # Format prompt using chat template
    messages = [
        {
            "role": "system",
            "content": "Solve the math problem step by step. State the final answer at the end on a new line starting with '#### ' followed by the numeric value."
        },
        {"role": "user", "content": question}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate response
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    # Slice output to decode ONLY the generated tokens (ignoring prompt tokens)
    prompt_length = inputs["input_ids"].shape[-1]
    generated_text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

    # Extract model prediction
    predicted_answer = extract_answer_num(generated_text)

    # Check correctness
    is_correct = (predicted_answer == expected_answer)
    if is_correct:
        correct += 1

    print(f"Sample [{i+1}/{total}] | Pred: {predicted_answer} | Expected: {expected_answer} | {'✓' if is_correct else '✗'}")

accuracy = correct / total
print("\n" + "="*30)
print(f"Final Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
print("="*30)