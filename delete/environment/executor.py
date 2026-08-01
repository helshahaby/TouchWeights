import io
import contextlib

class PythonExecutor:

    def execute(self, code):

        output = io.StringIO()

        try:
            with contextlib.redirect_stdout(output):
                exec(code)

            return True, output.getvalue()

        except Exception as e:
            return False, str(e)