"""Calculator tool using RestrictedPython for safe code execution."""

import math

from RestrictedPython import compile_restricted, safe_globals

from .base import BaseTool, ToolResult


class CalculatorTool(BaseTool):
    """Calculator tool for mathematical computations using RestrictedPython."""

    def __init__(self):
        description = (
            "Perform mathematical calculations. Actions: 'calculate' (evaluate expression with math functions like sqrt, sin, cos, log, etc.)"
        )
        super().__init__("calculator", description)

        # Safe globals with math functions
        self.safe_env = {
            **safe_globals,
            "__builtins__": {
                # Basic types
                "int": int,
                "float": float,
                "str": str,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                # Math operations
                "abs": abs,
                "min": min,
                "max": max,
                "sum": sum,
                "round": round,
                "pow": pow,
                # Constants
                "True": True,
                "False": False,
                "None": None,
            },
            # Math module functions
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "log10": math.log10,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute calculation."""
        action = kwargs.get("action", "calculate")

        if action == "calculate":
            return self._calculate(kwargs.get("expression"))
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _calculate(self, expression: str | None) -> ToolResult:
        """Evaluate a mathematical expression safely."""
        if not expression:
            return ToolResult(success=False, data=None, error="expression is required")

        try:
            # Compile the expression
            compile_result = compile_restricted(expression, "<calculator>", "eval")

            # Check if it's a CompileResult object with errors attribute
            if hasattr(compile_result, "errors") and compile_result.errors:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Syntax errors: {', '.join(compile_result.errors)}",
                )

            # Get the code object
            code_obj = compile_result.code if hasattr(compile_result, "code") else compile_result

            # Execute with restricted globals
            result = eval(code_obj, self.safe_env, {})

            return ToolResult(
                success=True,
                data={"result": result, "expression": expression},
            )

        except ZeroDivisionError:
            return ToolResult(success=False, data=None, error="Division by zero")
        except OverflowError:
            return ToolResult(success=False, data=None, error="Numerical overflow")
        except Exception as e:
            return ToolResult(success=False, data=None, error=f"Calculation error: {str(e)}")
