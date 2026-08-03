import re

def calculate_reward(exec_result: str, target: str) -> float:
    """Calculates execution accuracy reward against target ground truth."""
    if "Execution Error" in str(exec_result) or "No output printed" in str(exec_result):
        return -1.0

    target_clean = str(target).strip()
    
    # Extract numbers from code output
    numbers = re.findall(r'[-+]?\d*\.\d+|\d+', str(exec_result))
    if not numbers:
        return -0.5
    
    pred_val = numbers[-1].strip()

    # Direct match
    if pred_val == target_clean:
        return 1.0

    # Floating point comparison fallback
    try:
        if abs(float(pred_val) - float(target_clean)) < 1e-4:
            return 1.0
    except ValueError:
        pass

    return -0.5