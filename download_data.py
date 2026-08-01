from datasets import load_dataset
import json
import os


os.makedirs("data",exist_ok=True)


dataset = load_dataset(
    "openai/gsm8k",
    "main"
)


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
