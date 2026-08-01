import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model(model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct"):
    # Configure 4-bit quantization using BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config, # <--- Pass config here instead of load_in_4bit=True
        device_map="auto"
    )

    return model, tokenizer