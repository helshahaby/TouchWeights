import os
import re
import ast
import torch
import multiprocessing
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

from model import load_model
from environment import PythonEnvironment

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

env = PythonEnvironment()

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model, tokenizer = load_model()

SYSTEM_PROMPT = """You are a mathematical reasoning expert. For every math word problem:
1. Parse entities and recipients accurately.
2. Ensure consistent time units (convert daily metrics to weekly when calculating weekly totals: 7 days = 168 hours).
3. Handle relative phrasing strictly: 'one less than double X' is `(2 * X) - 1`.
4. Pay attention to multi-day multipliers (e.g., daily expense * total days).
5. Output ONLY Python code inside ```python ... ``` blocks."""

def get_completion_text(comp) -> str:
    if isinstance(comp, str):
        return comp
    elif isinstance(comp, list) and len(comp) > 0:
        first = comp[0]
        if isinstance(first, dict):
            return first.get("content", "")
        return str(first)
    elif isinstance(comp, dict):
        return comp.get("content", "")
    return str(comp)

def extract_python_code(text: str) -> str:
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else ""

def parse_final_printed_number(raw_output: str) -> str:
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

def logic_safety_reward(completions, **kwargs):
    """Penalizes impossible daily/weekly balance setups & negative physical outputs."""
    rewards = []
    for comp in completions:
        comp_text = get_completion_text(comp)
        code = extract_python_code(comp_text)
        if not code:
            rewards.append(0.0)
            continue
        
        score = 1.0
        # Penalize subtracting weekly totals directly from 24 hours
        if "24 -" in code and ("hours_per_week" in code or "168" not in code):
            score -= 4.0

        # Penalize arbitrary halving heuristics
        if "// 2" in code or "/ 2" in code:
            score -= 2.0

        rewards.append(score)
    return rewards

def execution_reward_function(completions, **kwargs):
    rewards = []
    targets = kwargs.get("target", None)
    if targets is None:
        targets = [""] * len(completions)
    elif not isinstance(targets, list):
        targets = [targets] * len(completions)

    for comp, target_ans in zip(completions, targets):
        comp_text = get_completion_text(comp)
        code = extract_python_code(comp_text)
        if not code or not target_ans:
            rewards.append(-2.0)
            continue

        try:
            if "print(" not in code:
                lines = [l.strip() for l in code.splitlines() if l.strip()]
                if lines:
                    last_var = lines[-1].split("=")[0].strip()
                    code += f"\nprint({last_var})"

            exec_result = env.run(code)
            exec_str = str(exec_result).strip() if exec_result else ""

            if "Execution Error:" in exec_str or "SyntaxError" in exec_str:
                rewards.append(-2.0)
                continue

            extracted_ans = parse_final_printed_number(exec_str)
            
            # Reject negative output values for physical quantities
            if extracted_ans != "" and float(extracted_ans) < 0:
                rewards.append(-5.0)
                continue

            if extracted_ans and abs(float(extracted_ans) - float(target_ans)) < 1e-3:
                rewards.append(5.0)
            else:
                rewards.append(-2.0)
        except Exception:
            rewards.append(-2.0)
            
    return rewards

raw_ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=42).select(range(500))

def prepare_sample(example):
    answer_text = example["answer"]
    target_num = ""
    match = re.search(r"####\s*([-+]?\d*\.?\d+)", str(answer_text))
    if match:
        val = float(match.group(1).replace(",", ""))
        target_num = str(int(val)) if val.is_integer() else str(val)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["question"]}
    ]

    return {
        "prompt": messages,
        "target": target_num
    }

dataset = raw_ds.map(prepare_sample)

training_args = GRPOConfig(
    output_dir="./adapter",
    gradient_checkpointing=True,  
    learning_rate=1e-4,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=2,
    max_completion_length=220,
    max_steps=50,
    logging_steps=5,
    save_steps=25,
    temperature=0.7,
    use_cpu=False,
    fp16=True,
)

trainer = GRPOTrainer(
    model=model,
    peft_config=peft_config,
    reward_funcs=[logic_safety_reward, execution_reward_function],
    args=training_args,
    train_dataset=dataset
)

print("Starting Training...")
trainer.train(resume_from_checkpoint=False)

print("Saving adapter...")
trainer.save_model("./adapter")
tokenizer.save_pretrained("./adapter")
print("Saved adapter successfully to ./adapter!")