"""Code execution tool using RestrictedPython for safe execution."""

import sys
from io import StringIO
from typing import Any

from RestrictedPython import compile_restricted, safe_globals, limited_builtins
from RestrictedPython.PrintCollector import PrintCollector
from RestrictedPython.Guards import guarded_iter_unpack_sequence

from .base import BaseTool, ToolResult


class CodeExecutionTool(BaseTool):
    """Python code execution with RestrictedPython sandbox."""

    def __init__(self, test_cases: list[dict[str, Any]] | None = None):
        description = "Execute Python code safely. Actions: 'execute' (run Python code string), 'test' (run code against test cases)"
        super().__init__("python_executor", description)
        self.test_cases = test_cases or []
        
        # Safe execution environment with all RestrictedPython guards
        self.safe_env = {
            **safe_globals,
            "__builtins__": limited_builtins,
            "_print_": PrintCollector,
            "_getattr_": getattr,
            "_getitem_": lambda obj, index: obj[index],
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            "__name__": "restricted_module",
            "__metaclass__": type,
        }

    def execute(self, **kwargs) -> ToolResult:
        """Execute code action."""
        action = kwargs.get("action", "execute")

        if action == "execute":
            return self._execute_code(kwargs.get("code"))
        elif action == "test":
            return self._test_code(kwargs.get("code"))
        else:
            return ToolResult(success=False, data=None, error=f"Unknown action: {action}")

    def _execute_code(self, code: str | None) -> ToolResult:
        """Execute Python code and return output."""
        if not code:
            return ToolResult(success=False, data=None, error="code is required")

        try:
            # Compile the code with RestrictedPython
            compile_result = compile_restricted(code, "<string>", "exec")

            if hasattr(compile_result, "errors") and compile_result.errors:
                # Format errors properly - they can be tuples or strings
                error_msgs = []
                for err in compile_result.errors:
                    if isinstance(err, tuple):
                        error_msgs.append(str(err[0]) if err else "Unknown error")
                    else:
                        error_msgs.append(str(err))
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Syntax errors: {'; '.join(error_msgs)}",
                )

            code_obj = compile_result.code if hasattr(compile_result, "code") else compile_result

            # Execute with RestrictedPython environment
            exec_globals = self.safe_env.copy()
            exec(code_obj, exec_globals)

            # Get output from PrintCollector
            output = exec_globals.get("printed", "")

            return ToolResult(
                success=True,
                data={
                    "output": output.strip() if output else None,
                    "status": "completed",
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Execution error: {str(e)}",
            )

    def _test_code(self, code: str | None) -> ToolResult:
        """Execute code and run test cases."""
        if not code:
            return ToolResult(success=False, data=None, error="code is required")

        if not self.test_cases:
            return ToolResult(
                success=False,
                data=None,
                error="No test cases configured for this task",
            )

        try:
            # Compile the code with RestrictedPython
            compile_result = compile_restricted(code, "<string>", "exec")

            if hasattr(compile_result, "errors") and compile_result.errors:
                # Format errors properly - they can be tuples or strings
                error_msgs = []
                for err in compile_result.errors:
                    if isinstance(err, tuple):
                        error_msgs.append(str(err[0]) if err else "Unknown error")
                    else:
                        error_msgs.append(str(err))
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Syntax errors: {'; '.join(error_msgs)}",
                )

            code_obj = compile_result.code if hasattr(compile_result, "code") else compile_result

            # Execute code to define functions
            exec_globals = self.safe_env.copy()
            exec(code_obj, exec_globals)

            # Find the main function (assume it's the first non-builtin function)
            func_name = None
            for name, obj in exec_globals.items():
                if callable(obj) and not name.startswith("_"):
                    func_name = name
                    break

            if not func_name:
                return ToolResult(
                    success=False,
                    data=None,
                    error="No function found in code",
                )

            func = exec_globals[func_name]

            # Run test cases
            test_results = []
            all_passed = True

            for i, test_case in enumerate(self.test_cases):
                test_input = test_case["input"]
                expected_output = test_case["expected_output"]

                try:
                    actual_output = func(test_input)
                    passed = actual_output == expected_output

                    test_results.append(
                        {
                            "test_case": i + 1,
                            "input": test_input,
                            "expected": expected_output,
                            "actual": actual_output,
                            "passed": passed,
                        }
                    )

                    if not passed:
                        all_passed = False

                except Exception as e:
                    test_results.append(
                        {
                            "test_case": i + 1,
                            "input": test_input,
                            "expected": expected_output,
                            "actual": None,
                            "passed": False,
                            "error": str(e),
                        }
                    )
                    all_passed = False

            return ToolResult(
                success=True,
                data={
                    "all_passed": all_passed,
                    "test_results": test_results,
                    "total_tests": len(test_results),
                    "passed_tests": sum(1 for r in test_results if r["passed"]),
                },
            )

        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=f"Test execution error: {str(e)}",
            )
