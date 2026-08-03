from model import load_model
from environment import PythonEnvironment
from reward import calculate_reward
import re

def extract_python_code(text: str) -> str:
    pattern = r"```python\s*(.*?)\s*```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip()
    return text.strip()

model, tokenizer = load_model()
env = PythonEnvironment()

question = """
Calculate 25*12 using Python.
"""

prompt = f"""
Solve this problem.
Return only python code.

{question}
"""

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

output = model.generate(
    **inputs,
    max_new_tokens=512
)

answer = tokenizer.decode(
    output[0],
    skip_special_tokens=True
)

print("================")
print("MODEL OUTPUT")
print(answer)

# EXTRACT CLEAN CODE BEFORE EXECUTING
code_to_run = extract_python_code(answer)

print("================")
print("EXTRACTED CODE TO EXECUTE:")
print(code_to_run)

result = env.run(code_to_run)

print("================")
print("EXECUTION")
print(result)

reward = calculate_reward(
    result,
    "300"
)

print("================")
print("REWARD:", reward)