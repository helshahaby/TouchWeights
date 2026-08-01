MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

DATASET_SIZE = 500

MAX_LENGTH = 512

DEVICE = "cuda"

LORA_CONFIG = {
    "r":16,
    "lora_alpha":32,
    "lora_dropout":0.05
}