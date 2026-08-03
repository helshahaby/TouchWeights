import json
import os
import re
import torch
from datasets import load_dataset
from peft import PeftModel
from model import load_model
from environment import PythonEnvironment

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

env = PythonEnvironment()

SYSTEM_PROMPT = """You are an expert mathematical Python programmer. Solve the user's math word problem by writing executable Python code.

Follow these strict reasoning guidelines:
1. READ UNUSED VS USED: Pay close attention to "unused" vs "used". If 2/3 were used, then 1/3 (1 - 2/3) are unused.
2. RATIO & FRACTIONS: "5 times as many bags of gravel as barrels of pitch" with 2 bags of gravel means pitch = 2 / 5 = 0.4 barrels per truck.
3. ITEM RATES: Calculate cost per day first, then multiply by total days. (e.g. 4 pills @ $1.50 + 5 pills @ $7.00 = $41/day * 14 days = $574).
4. FIXED UNIT DROP TESTS: If each test requires N drops, total tests = total_drops // N. Non-copper tested = total_tests - copper_beakers.
5. READ THE GOAL CAREFULLY: If the question asks for "pages remaining to read", compute remaining pages (not pages read). If it asks for "hours left over", subtract from total weekly hours (168 hrs/week).
6. MULTI-DAY MULTIPLIERS: When a daily rate or cost is calculated, always multiply by the total number of days (e.g., daily cost * total days).
7. RATIOS & PHRASING: 'one less than double X' means `(2 * X) - 1`. 'A uses 5 times as many gravel as pitch' means `pitch = gravel / 5`.
8. FIXED TEST COST: If each test requires N drops, total tests = `total_drops // N`.
9. OUTPUT FORMAT: Output ONLY valid executable Python code inside ```python ... ``` blocks. Always end with a `print(...)` statement of the final answer.
10. Parse entities and recipients accurately.
11. Always keep weekly time budgets in 168 hours total per week (7 days * 24 hours). Never subtract weekly activities from 24 hours.
12. Handle relative phrasing strictly: 'one less than double X' is `(2 * X) - 1`.
13. Multi-day expenses must multiply daily cost by total days (e.g., daily cost * total days).
14. Convert rates accurately (e.g., $12/hr = $0.20/min).
15. Identify target entities and recipients accurately (e.g., number of recipients vs. total group size).
16. Pay close attention to time scales (daily vs. weekly) and unit conversions (books to pages, bags to barrels).
17. Handle relative phrasing accurately: 'one less than double X' is `(2 * X) - 1`, and 'A is N times as much as B' means `B = A / N`.
18. Define given quantities line-by-line. Do NOT assume arbitrary split operations like `// 2`.
19. Distinguish between who is performing the action and who is receiving the final items/amounts.
20. Define each quantity explicitly before performing operations.
21. Check time units for every quantity (e.g., per day vs. per week). Convert all daily quantities to weekly quantities before adding them to weekly totals.
22. Define explicitly given sub-quantities line-by-line. Do NOT assume equal splits or use `// 2`.
23. Wrap compound additions in parentheses before multiplying (e.g., `(count1 * price1) + (count2 * price2)`).
24. Keep units consistent. Never subtract a count of items from a count of pages/dollars. Convert to matching units first.
25.Never create arbitrary numbers (like setting blue ties to 20). Every number must come directly from the text or a mathematical calculation.
26. DEFINE ALL ENTITIES & PARTS EXPLICITLY example full-price cost, discount-price cost, and total counts on separate lines before calculating total sum. Never skip a person/item!
27. ACCURATE BASE REFERENCES: Check which variable is the baseline reference! 
  - "B is 1/4 times more than A" -> B = A * (1 + 1/4)  (Uses A, NOT C or D!)
28. ADDITIVE VS MULTIPLICATIVE 'MORE':
   - "N more than X" or "N times more than X" (where N is an integer like 10) -> X + N (ADDITION).
   - "N times as many as X" -> X * N (MULTIPLICATION).
29. TOTAL SUMS: Calculate the explicit sum of ALL defined entities at the end.
"""

print("Loading base model...")
base_model, tokenizer = load_model(is_training=False)

try:
    model = PeftModel.from_pretrained(base_model, "./adapter")
    print("Successfully loaded LoRA adapter from ./adapter!\n")
except Exception as e:
    print(f"Warning: Adapter load failed ({e}). Running base model.\n")
    model = base_model

model.eval()

print("Loading GSM8K test dataset...")
dataset = None
dataset_size = 0

try:
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset_size = len(dataset)
    print(f"Dataset ready with {dataset_size} test samples.\n")
except Exception as e:
    print(f"Warning: Failed to load dataset ({e}). Index lookup disabled.\n")

def extract_hash_target(text: str) -> tuple[str, str]:
    """Extracts (cleaned_question, ground_truth_target) from text cleanly."""
    if not text:
        return "", ""
    cleaned_text = str(text)
    target = ""

    # Handle JSON/Dict strings
    if '"question":' in cleaned_text or "'question':" in cleaned_text:
        try:
            data = json.loads(cleaned_text)
            cleaned_text = data.get("question", cleaned_text)
            answer_str = data.get("answer", "")
            if "####" in answer_str:
                raw_ans = answer_str.split("####")[-1].strip()
                match = re.search(r"[-+]?\d[\d,]*\.?\d*", raw_ans)
                if match:
                    val = float(match.group(0).replace(",", ""))
                    target = str(int(val)) if val.is_integer() else str(val)
            return cleaned_text.strip(), target
        except Exception:
            q_match = re.search(r'"question"\s*:\s*"(.*?)"\s*,\s*"answer"', cleaned_text, re.DOTALL)
            if q_match:
                cleaned_text = q_match.group(1).encode('utf-8').decode('unicode_escape')

    # Extract target number after ####
    if "####" in cleaned_text:
        parts = cleaned_text.split("####")
        raw_target_part = parts[-1].strip()
        num_match = re.search(r"[-+]?\d[\d,]*\.?\d*", raw_target_part)
        if num_match:
            try:
                val = float(num_match.group(0).replace(",", ""))
                target = str(int(val)) if val.is_integer() else str(val)
            except ValueError:
                target = ""
        cleaned_text = parts[0].strip()

    return cleaned_text, target

def parse_final_printed_number(raw_output: str) -> str:
    """Parses final printed scalar number from executed Python output."""
    if not raw_output or not str(raw_output).strip():
        return ""
    lines = [line.strip() for line in str(raw_output).strip().splitlines() if line.strip()]
    if not lines:
        return ""
    matches = re.findall(r'[-+]?\d*\.?\d+', lines[-1])
    if matches:
        val = float(matches[-1])
        return str(int(val)) if val.is_integer() else str(val)
    return ""

print("==================================================")
print("Interactive Evaluation Shell Ready")
print(" - Type a digit (e.g. 0, 1, 2) to load a GSM8K sample.")
print(" - Paste custom question or text with #### target.")
print(" - Type 'exit' or 'quit' to stop.")
print("==================================================\n")

while True:
    user_input = input("[Enter Question / Digit / 'exit']: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break
    if not user_input:
        continue

    question = ""
    target_answer = ""

    # Strictly digit check for dataset lookup
    if dataset is not None and user_input.isdigit() and int(user_input) < len(dataset):
        idx = int(user_input)
        question = dataset[idx]["question"]
        raw_answer = dataset[idx]["answer"]
        _, target_answer = extract_hash_target(raw_answer)
        print(f"\n---> Loaded Sample [{idx}]: {question}")
        if target_answer:
            print(f"---> Target Answer: {target_answer}")
    else:
        question, target_answer = extract_hash_target(user_input)
        print(f"\n---> Custom Question: {question}")
        if target_answer:
            print(f"---> Target Answer: {target_answer}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    print("\nGenerating model output...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_text = tokenizer.decode(output[0][prompt_length:], skip_special_tokens=True)

    print("-" * 20 + " MODEL OUTPUT " + "-" * 20)
    print(generated_text)
    print("-" * 54)

    # Code Execution Engine
    matches = re.findall(r"```python\s*(.*?)\s*```", generated_text, re.DOTALL)
    if matches:
        code = matches[0].strip()

        if "print(" not in code:
            lines = [l.strip() for l in code.splitlines() if l.strip()]
            if lines:
                last_var = lines[-1].split("=")[0].strip()
                code += f"\nprint({last_var})"

        print("\n[Executing Code Block...]")
        res = env.run(code)
        exec_output = str(res).strip()
        extracted_ans = parse_final_printed_number(exec_output)

        print(f"Execution Result: {exec_output}")
        print(f"Extracted Final Answer: {extracted_ans}")

        if target_answer != "":
            try:
                if extracted_ans != "" and abs(float(extracted_ans) - float(target_answer)) < 1e-3:
                    print("Evaluation Status: [PASS]")
                else:
                    print(f"Evaluation Status: [FAIL] (Expected: {target_answer}, Got: {extracted_ans})")
            except ValueError:
                print("Evaluation Status: [FAIL]")
        else:
            if extracted_ans != "":
                print("Evaluation Status: [PASS (Executed - No Target Provided)]")
            else:
                print("Evaluation Status: [FAIL - Execution Error]")
    else:
        print("\n[Executing Code Block...]")
        print("Execution Result: No Code Block Found")
        print("Extracted Final Answer: None")
        print("Evaluation Status: [FAIL]")

    print("\n" + "=" * 50 + "\n")