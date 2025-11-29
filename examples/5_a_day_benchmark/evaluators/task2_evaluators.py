"""Task 2 Evaluators: Code Generation (House Robber DP).

Task: User asks for a house_robber function implementing dynamic programming algorithm.
Success criteria: Generated code passes unit tests and uses optimal algorithm.
"""

import ast
import re
from typing import Any, Dict, Optional
from io import StringIO
import sys

from maseval import Evaluator, Environment, Task, User, MessageHistory
from .utils import extract_python_code


class UnitTestEvaluator(Evaluator):
    """Evaluates generated code by running unit tests.
    
    Measures: Does the generated code actually work and produce correct outputs?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)
        self.test_cases = task.evaluation_data["test_cases"]
        self.function_name = task.evaluation_data["function_name"]

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate code by running unit tests."""
        # Extract code from trace
        code = extract_python_code(trace)
        
        if not code:
            return {
                "test_cases_passed": 0,
                "test_pass_rate": 0.0,
                "all_tests_passed": False,
                "error": "No code found in trace"
            }
        
        # Run unit tests
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
            "errors": errors if errors else None
        }

    def _execute_code(self, code: str, function_name: str, test_input: Any) -> Any:
        """Execute code and return result."""
        # Create namespace
        namespace = {}
        
        # Execute code to define function
        exec(code, namespace)
        
        # Get function
        if function_name not in namespace:
            raise ValueError(f"Function '{function_name}' not found in code")
        
        func = namespace[function_name]
        
        # Execute function with test input
        return func(test_input)


class AlgorithmicComplexityEvaluator(Evaluator):
    """Evaluates algorithmic complexity of generated code using AST analysis.
    
    Measures: Is the algorithm optimal (O(n) time, O(1) space)?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate algorithmic complexity."""
        code = extract_python_code(trace)
        
        if not code:
            return {
                "time_complexity": "unknown",
                "space_complexity": "unknown",
                "is_optimal_time": False,
                "is_optimal_space": False,
                "uses_dynamic_programming": False,
                "algorithm_efficiency_score": 0.0
            }
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {
                "time_complexity": "syntax_error",
                "space_complexity": "syntax_error",
                "is_optimal_time": False,
                "is_optimal_space": False,
                "uses_dynamic_programming": False,
                "algorithm_efficiency_score": 0.0
            }
        
        # Analyze complexity
        analysis = self._analyze_complexity(tree)
        
        # Optimal for house robber: O(n) time, O(1) space
        is_optimal_time = analysis["time_complexity"] == "O(n)"
        is_optimal_space = analysis["space_complexity"] == "O(1)"
        
        # Calculate efficiency score
        score = (int(is_optimal_time) + int(is_optimal_space)) / 2.0
        
        return {
            "time_complexity": analysis["time_complexity"],
            "space_complexity": analysis["space_complexity"],
            "is_optimal_time": is_optimal_time,
            "is_optimal_space": is_optimal_space,
            "uses_dynamic_programming": analysis["uses_dp"],
            "algorithm_efficiency_score": score,
            "analysis_details": analysis["details"]
        }

    def _analyze_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyze code complexity from AST."""
        details = {
            "loops": 0,
            "nested_loops": 0,
            "recursion": False,
            "list_creations": 0,
            "dict_creations": 0
        }
        
        # Count loops and data structures
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                details["loops"] += 1
            elif isinstance(node, ast.ListComp):
                details["list_creations"] += 1
            elif isinstance(node, ast.DictComp):
                details["dict_creations"] += 1
        
        # Check for recursion
        function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        for func in function_defs:
            func_name = func.name
            for node in ast.walk(func):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == func_name:
                        details["recursion"] = True
        
        # Check for nested loops
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if child != node and isinstance(child, (ast.For, ast.While)):
                        details["nested_loops"] += 1
                        break
        
        # Determine complexity
        if details["nested_loops"] > 0:
            time_complexity = "O(n²)"
        elif details["recursion"]:
            time_complexity = "O(2^n)" if "memoization" not in str(tree).lower() else "O(n)"
        elif details["loops"] > 0:
            time_complexity = "O(n)"
        else:
            time_complexity = "O(1)"
        
        # Space complexity
        if details["list_creations"] > 0 or details["dict_creations"] > 0:
            space_complexity = "O(n)"
        else:
            space_complexity = "O(1)"
        
        # Check for DP pattern (loop with previous values)
        uses_dp = details["loops"] > 0 and not details["recursion"]
        
        return {
            "time_complexity": time_complexity,
            "space_complexity": space_complexity,
            "uses_dp": uses_dp,
            "details": details
        }


class CodeQualityEvaluator(Evaluator):
    """Evaluates code quality and style.
    
    Measures: Is the code well-written, documented, and handles edge cases?
    """

    def __init__(self, task: Task, environment: Environment, user: Optional[User] = None):
        super().__init__(task, environment, user)

    def __call__(self, trace: MessageHistory) -> Dict[str, Any]:
        """Evaluate code quality."""
        code = extract_python_code(trace)
        
        if not code:
            return {
                "has_docstring": False,
                "variable_names_meaningful": False,
                "handles_edge_cases": False,
                "code_quality_score": 0.0
            }
        
        # Check for docstring
        has_docstring = '"""' in code or "'''" in code or '# ' in code
        
        # Check for meaningful variable names (not just a, b, c)
        # Look for descriptive names like 'prev', 'curr', 'max_rob', etc.
        single_char_vars = re.findall(r'\b[a-z]\b', code.lower())
        meaningful_vars = re.findall(r'\b[a-z_]{3,}\b', code.lower())
        variable_names_meaningful = len(meaningful_vars) > len(single_char_vars)
        
        # Check for edge case handling
        handles_edge_cases = (
            'if not' in code or 
            'if len(' in code or 
            '== 0' in code or
            '== []' in code or
            'len(nums) == 0' in code.replace(' ', '') or
            'len(nums)==0' in code.replace(' ', '')
        )
        
        # Calculate score
        checks = [has_docstring, variable_names_meaningful, handles_edge_cases]
        score = sum(checks) / len(checks)
        
        return {
            "has_docstring": has_docstring,
            "variable_names_meaningful": variable_names_meaningful,
            "handles_edge_cases": handles_edge_cases,
            "code_quality_score": score
        }
