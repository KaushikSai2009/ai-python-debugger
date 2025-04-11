import ast

class PythonDebugger:
    def __init__(self, code):
        self.code = code

    def find_syntax_errors(self):
        try:
            ast.parse(self.code)
            return None
        except SyntaxError as e:
            return str(e)

    def debug(self):
        syntax_error = self.find_syntax_errors()
        if syntax_error:
            return f"Syntax Error: {syntax_error}"
        return "No syntax errors found."

if __name__ == "__main__":
    code = """
def add(a, b):
return a + b
"""
    debugger = PythonDebugger(code)
    print(debugger.debug())