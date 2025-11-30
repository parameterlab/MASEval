"""Evaluators for Task 2: Code Generation (House Robber DP).

Task: User asks for a house_robber function implementing dynamic programming algorithm.
Success criteria: Generated code passes unit tests and uses optimal algorithm.
"""

import ast
import json
from typing import Any, Dict, Optional

from RestrictedPython import compile_restricted, safe_globals, limited_builtins
from RestrictedPython.Guards import guarded_iter_unpack_sequence

from maseval import Evaluator, Environment, Task, User
from .utils import call_llm_judge


class UnitTestEvaluator(Evaluator):
    """Evaluates generated code by running unit tests.

    Evaluation type: Code execution
    Measures: Does the generated code actually work and produce correct outputs?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.test_cases = task.evaluation_data["test_cases"]
        self.function_name = task.evaluation_data["function_name"]

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to final_answer tool only. The code is in the agent's final response."""
        tools = traces.get("tools", {})
        return tools.get("final_answer", {})

    def __call__(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate code by running unit tests."""

        # Get final answer from tool invocations
        invocations = traces.get("invocations", [])

        if not invocations:
            return {"test_cases_passed": 0, "test_pass_rate": 0.0, "all_tests_passed": False, "error": "No final answer found"}

        # Get the final answer content and extract code
        final_answer = str(invocations[-1].get("inputs", {}).get("answer", ""))
        code = self._extract_code_from_answer(final_answer)

        if not code:
            return {"test_cases_passed": 0, "test_pass_rate": 0.0, "all_tests_passed": False, "error": "No code found in trace"}

        test_results = []
        errors = []

        for i, test_case in enumerate(self.test_cases):
            test_input = test_case["input"]
            expected_output = test_case["expected_output"]

            try:
                result = self._execute_code(code, self.function_name, test_input)
                passed = result == expected_output
                test_results.append(passed)

                if not passed:
                    errors.append(f"Test {i}: expected {expected_output}, got {result}")

            except Exception as e:
                test_results.append(False)
                errors.append(f"Test {i}: {type(e).__name__}: {str(e)}")

        tests_passed = sum(test_results)
        pass_rate = tests_passed / len(self.test_cases)

        return {
            "test_cases_passed": tests_passed,
            "test_pass_rate": pass_rate,
            "all_tests_passed": all(test_results),
            "total_tests": len(self.test_cases),
            "errors": errors if errors else None,
        }

    def _execute_code(self, code: str, function_name: str, test_input: Any) -> Any:
        """Execute code safely using RestrictedPython and return result."""
        # Compile with RestrictedPython
        compile_result = compile_restricted(code, "<evaluator>", "exec")

        if hasattr(compile_result, "errors") and compile_result.errors:
            # Format errors properly - they can be tuples or strings
            error_msgs = []
            for err in compile_result.errors:
                if isinstance(err, tuple):
                    error_msgs.append(str(err[0]) if err else "Unknown error")
                else:
                    error_msgs.append(str(err))
            raise SyntaxError("; ".join(error_msgs))

        code_obj = compile_result.code if hasattr(compile_result, "code") else compile_result

        # Safe execution environment
        safe_env = {
            **safe_globals,
            "__builtins__": limited_builtins,
            "_getattr_": getattr,
            "_getitem_": lambda obj, index: obj[index],
            "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
        }

        exec(code_obj, safe_env)

        if function_name not in safe_env:
            raise ValueError(f"Function '{function_name}' not found in code")

        return safe_env[function_name](test_input)

    def _extract_code_from_answer(self, answer: str) -> Optional[str]:
        """Extract Python code from final answer string."""
        import re

        # Look for code blocks with markdown
        code_block_pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, answer, re.DOTALL)

        if matches:
            return matches[-1]  # Return last code block

        # Look for function definitions without markdown
        code_pattern = r"def\s+\w+\s*\([^)]*\):"
        if re.search(code_pattern, answer):
            lines = answer.split("\n")
            code_lines = []
            in_function = False

            for line in lines:
                if re.match(r"def\s+\w+", line):
                    in_function = True
                if in_function:
                    code_lines.append(line)

            if code_lines:
                return "\n".join(code_lines)

        return None


class AlgorithmicComplexityEvaluator(Evaluator):
    """Evaluates algorithmic complexity of generated code using AST analysis.

    Evaluation type: Static analysis
    Measures: Is the algorithm optimal (O(n) time, O(1) space)?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to final_answer tool only. The code is in the agent's final response."""
        tools = traces.get("tools", {})
        return tools.get("final_answer", {})

    def __call__(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate algorithmic complexity."""

        # Get final answer from tool invocations
        invocations = traces.get("invocations", [])

        if not invocations:
            return {"time_complexity": "unknown", "is_optimal": False, "uses_dynamic_programming": False, "algorithm_efficiency_score": 0.0}

        # Get the final answer content and extract code
        final_answer = str(invocations[-1].get("inputs", {}).get("answer", ""))
        code = self._extract_code_from_answer(final_answer)

        if not code:
            return {"time_complexity": "unknown", "is_optimal": False, "uses_dynamic_programming": False, "algorithm_efficiency_score": 0.0}

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {
                "time_complexity": "syntax_error",
                "is_optimal": False,
                "uses_dynamic_programming": False,
                "algorithm_efficiency_score": 0.0,
            }

        analysis = self._analyze_complexity(tree)
        is_optimal = analysis["time_complexity"] == "O(n)"

        return {
            "time_complexity": analysis["time_complexity"],
            "is_optimal": is_optimal,
            "uses_dynamic_programming": analysis["uses_dp"],
            "algorithm_efficiency_score": 1.0 if is_optimal else 0.5,
        }

    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code complexity from AST."""
        loops = 0
        nested_loops = 0
        recursion = False

        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                loops += 1

        # Check for recursion
        function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        for func in function_defs:
            func_name = func.name
            for node in ast.walk(func):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == func_name:
                        recursion = True

        # Check for nested loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if child != node and isinstance(child, (ast.For, ast.While)):
                        nested_loops += 1
                        break

        # Determine complexity
        if nested_loops > 0:
            time_complexity = "O(n²)"
        elif recursion:
            time_complexity = "O(2^n)"
        elif loops > 0:
            time_complexity = "O(n)"
        else:
            time_complexity = "O(1)"

        uses_dp = loops > 0 and not recursion

        return {
            "time_complexity": time_complexity,
            "uses_dp": uses_dp,
        }

    def _extract_code_from_answer(self, answer: str) -> Optional[str]:
        """Extract Python code from final answer string."""
        import re

        # Look for code blocks with markdown
        code_block_pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, answer, re.DOTALL)

        if matches:
            return matches[-1]  # Return last code block

        # Look for function definitions without markdown
        code_pattern = r"def\s+\w+\s*\([^)]*\):"
        if re.search(code_pattern, answer):
            lines = answer.split("\n")
            code_lines = []
            in_function = False

            for line in lines:
                if re.match(r"def\s+\w+", line):
                    in_function = True
                if in_function:
                    code_lines.append(line)

            if code_lines:
                return "\n".join(code_lines)

        return None


class CodeQualityEvaluator(Evaluator):
    """Evaluates code quality and style.

    Evaluation type: LLM-as-judge
    Measures: Is the code well-written, documented, and handles edge cases?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)

    def filter_traces(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to final_answer tool only. The code is in the agent's final response."""
        tools = traces.get("tools", {})
        return tools.get("final_answer", {})

    def __call__(self, traces: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate code quality."""

        # Get final answer from tool invocations
        invocations = traces.get("invocations", [])

        if not invocations:
            return {"has_docstring": False, "handles_edge_cases": False, "code_quality_score": 0.0}

        # Get the final answer content and extract code
        final_answer = str(invocations[-1].get("inputs", {}).get("answer", ""))
        code = self._extract_code_from_answer(final_answer)

        if not code:
            return {"has_docstring": False, "handles_edge_cases": False, "code_quality_score": 0.0}

        prompt = f"""
        You are a senior software engineer evaluating Python code quality.
        
        Code Snippet:
        ```python
        {code}
        ```
        
        Evaluate the following:
        1. Does it have a docstring or meaningful comments?
        2. Does it handle edge cases (e.g., empty input, invalid types)?
        3. Is the variable naming and structure clean and pythonic?
        
        Return a JSON object with:
        - has_docstring: boolean
        - handles_edge_cases: boolean
        - code_quality_score: float (0.0 to 1.0)
        """

        try:
            response = call_llm_judge(prompt)
            response = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(response)

            return {
                "has_docstring": result.get("has_docstring", False),
                "handles_edge_cases": result.get("handles_edge_cases", False),
                "code_quality_score": result.get("code_quality_score", 0.0),
            }
        except Exception as e:
            return {
                "has_docstring": False,
                "handles_edge_cases": False,
                "code_quality_score": 0.0,
                "error": f"LLM evaluation failed: {str(e)}",
            }

    def _extract_code_from_answer(self, answer: str) -> Optional[str]:
        """Extract Python code from final answer string."""
        import re

        # Look for code blocks with markdown
        code_block_pattern = r"```python\s*(.*?)\s*```"
        matches = re.findall(code_block_pattern, answer, re.DOTALL)

        if matches:
            return matches[-1]  # Return last code block

        # Look for function definitions without markdown
        code_pattern = r"def\s+\w+\s*\([^)]*\):"
        if re.search(code_pattern, answer):
            lines = answer.split("\n")
            code_lines = []
            in_function = False

            for line in lines:
                if re.match(r"def\s+\w+", line):
                    in_function = True
                if in_function:
                    code_lines.append(line)

            if code_lines:
                return "\n".join(code_lines)

        return None
