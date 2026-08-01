# utils.py
import re

def extract_numerical_answer(text: str) -> str:
    """Extracts the number inside ####, \\boxed{}, or falls back to the last trailing number."""
    hash_match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    if hash_match:
        return hash_match.group(1).strip()

    boxed_match = re.search(r'\\boxed\{([-+]?\d*\.?\d+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', text)
    return numbers[-1].strip() if numbers else ""

def calculate_math_reward(model_output: str, expected_answer: str) -> float:
    pred = extract_numerical_answer(model_output)
    return 1.0 if pred == str(expected_answer).strip() else -1.0