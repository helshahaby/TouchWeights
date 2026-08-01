import io
import sys
import re

class PythonEnvironment:
    def run(self, code: str) -> str:
        # 1. Clean markdown code fences (```python ... ``` or ``` ... ```)
        cleaned_code = re.sub(r"^```(?:python)?\n?", "", code.strip(), flags=re.MULTILINE)
        cleaned_code = re.sub(r"\n?```$", "", cleaned_code, flags=re.MULTILINE).strip()

        # 2. Redirect stdout to capture print(...) output
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer

        try:
            exec_globals = {}
            exec(cleaned_code, exec_globals)
            sys.stdout = old_stdout
            output = buffer.getvalue().strip()

            # Fallback: If no print() was called, check if a variable named 'result' exists
            if not output and "result" in exec_globals:
                output = str(exec_globals["result"])

            return output
            
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error: {e}"