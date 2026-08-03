# utils.py
import re

def extract_numerical_answer(text: str) -> str:
    """Extracts numerical answer from markdown, execution logs, or raw numbers."""
    if not text or "Error" in text or "Traceback" in text:
        return ""

    # 1. Match GSM8K target format: #### 121
    hash_match = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    if hash_match:
        return hash_match.group(1).strip()

    # 2. Match LaTeX box format: \boxed{121}
    boxed_match = re.search(r'\\boxed\{([-+]?\d*\.?\d+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # 3. Fallback: Get the last printed number in the output
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    return numbers[-1].strip() if numbers else ""


def calculate_math_reward(model_output: str, expected_answer: str) -> float:
    """Computes reward (+1.0 for correct answer, -1.0 for incorrect or error)."""
    pred = extract_numerical_answer(model_output)
    truth = str(expected_answer).strip()

    if not pred:
        return -1.0

    # First check: Direct String Match
    if pred == truth:
        return 1.0

    # Second check: Numeric Equality (e.g. 121.0 == 121 or 300 == 300.0)
    try:
        if float(pred) == float(truth):
            return 1.0
    except ValueError:
        pass

    return -1.0