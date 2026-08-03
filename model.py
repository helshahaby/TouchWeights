import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

def load_model(is_training: bool = False):
    print(f"Loading model and tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    has_cuda = torch.cuda.is_available()
    torch_dtype = torch.bfloat16 if has_cuda and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # FIX: Do NOT use device_map="auto" during GRPO training to avoid tensor split errors
    device_map = None if is_training else ("auto" if has_cuda else "cpu")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch_dtype if has_cuda else torch.float32, # Changed torch_dtype -> dtype
        device_map=device_map,
        trust_remote_code=True
    )

    return model, tokenizer