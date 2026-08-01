#Reasoning....

#### 42

import re

def extract_answer(text):

    match = re.findall(r"#### (.*)", text)

    if len(match):

        return match[-1].strip()

    return ""