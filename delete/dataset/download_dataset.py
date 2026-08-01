from datasets import load_dataset
import json

dataset = load_dataset("gsm8k", "main")

with open("gsm8k.json", "w") as f:
    json.dump(dataset["train"][:1000], f)

print("Dataset saved.")