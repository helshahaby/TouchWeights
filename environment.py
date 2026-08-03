import sys
import io
import contextlib

class PythonEnvironment:
    """Executes extracted Python code in a safe local scope and captures stdout."""
    def __init__(self, timeout: int = 5):
        self.timeout = timeout

    def run(self, code_str: str) -> str:
        output_buffer = io.StringIO()
        
        # FIX: Allow standard module imports (math, sympy, re, etc.)
        global_scope = {"__name__": "__main__"}
        local_scope = {}

        try:
            with contextlib.redirect_stdout(output_buffer):
                exec(code_str, global_scope, local_scope)
            
            output = output_buffer.getvalue().strip()
            
            # If stdout is empty, return the last assigned local variable
            if not output and local_scope:
                last_val = list(local_scope.values())[-1]
                output = str(last_val).strip()

            return output if output else "No output printed"
            
        except Exception as e:
            return f"Execution Error: {type(e).__name__}: {str(e)}"