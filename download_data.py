from datasets import load_dataset
from train_grpo import format_prompt
import json
import os
import re


os.makedirs("data",exist_ok=True)


# Load 100 real GSM8K samples for training diversity
raw_ds = load_dataset("gsm8k", "main", split="train[:100]")

def prepare_sample(example):
    # Extract numeric target from GSM8K answer field
    match = re.search(r'####\s*([-+]?\d*\.?\d+)', example["answer"])
    target = match.group(1).strip() if match else ""
    return {
        "prompt": format_prompt(example["question"]),
        "target": target
    }

dataset = raw_ds.map(prepare_sample)

samples=[]

for item in dataset["train"].select(range(500)):

    samples.append({

        "question":item["question"],
        "answer":item["answer"]

    })


with open(
    "data/gsm8k.json",
    "w"
) as f:

    json.dump(
        samples,
        f,
        indent=2
    )


print("Dataset downloaded")
