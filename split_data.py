from datasets import load_dataset

# Load your local JSON file
full_dataset = load_dataset("json", data_files="gsm8k.json", split="train")

# Perform an 80% Train / 20% Test split with fixed random seed
split_dataset = full_dataset.train_test_split(test_size=0.2, seed=42)

# Save to distinct files
split_dataset["train"].to_json("train_data.json")
split_dataset["test"].to_json("test_data.json")

print(f"Splitting complete!")
print(f" - Training samples saved to train_data.json ({len(split_dataset['train'])})")
print(f" - Testing samples saved to test_data.json   ({len(split_dataset['test'])})")