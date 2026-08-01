from model import load_model
from environment import PythonEnvironment
from reward import calculate_reward


model,tokenizer=load_model()


env=PythonEnvironment()


question="""

Calculate 25*12 using Python.

"""


prompt=f"""

Solve this problem.

Return only python code.

{question}

"""


inputs=tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)



output=model.generate(

    **inputs,

    max_new_tokens=200

)


answer=tokenizer.decode(
    output[0],
    skip_special_tokens=True
)


print("================")
print("MODEL OUTPUT")
print(answer)


result=env.run(answer)


print("================")
print("EXECUTION")
print(result)


reward=calculate_reward(
    result,
    "300"
)


print("================")
print("REWARD:",reward)