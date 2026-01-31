"""
Safe execution environment for generated comparison functions.

Provides a restricted execution context with access to audio analysis
functions and numpy, while limiting access to potentially dangerous operations.
"""

import importlib
import sys
from collections import Counter, defaultdict
from typing import Callable

import numpy as np

from src.audio_analyzer import (
    extract_hpcp_features,
    extract_key_features,
    extract_rhythm_features,
)

# Safe builtins for generated code
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}


def get_sandbox_globals() -> dict:
    """
    Get the globals dict for executing generated functions.

    Returns:
        Dict containing safe builtins and allowed modules/functions
    """
    # Import essentia lazily to avoid import errors if not installed
    try:
        import essentia.standard as es
    except ImportError:
        es = None

    return {
        "__builtins__": SAFE_BUILTINS,
        # Numpy
        "np": np,
        "numpy": np,
        # Audio analysis functions
        "extract_hpcp_features": extract_hpcp_features,
        "extract_key_features": extract_key_features,
        "extract_rhythm_features": extract_rhythm_features,
        # Collections
        "Counter": Counter,
        "defaultdict": defaultdict,
        # Essentia (if available)
        "essentia": {"standard": es} if es else None,
    }


def load_function_from_file(function_name: str) -> Callable:
    """
    Load a function by name from generated_functions.py.

    Args:
        function_name: Name of the function to load

    Returns:
        The loaded function

    Raises:
        ValueError: If function not found
    """
    # Force reimport to get latest version
    if "src.generated_functions" in sys.modules:
        del sys.modules["src.generated_functions"]

    try:
        module = importlib.import_module("src.generated_functions")
    except ImportError as e:
        raise ValueError(f"Could not import generated_functions module: {e}") from e

    if not hasattr(module, function_name):
        raise ValueError(
            f"Function '{function_name}' not found in generated_functions.py. "
            "Use 'lyrebird functions list' to see available functions."
        )

    return getattr(module, function_name)


def compile_and_execute(function_code: str, function_name: str) -> Callable:
    """
    Compile and return a function from source code.

    Args:
        function_code: Python source code defining the function
        function_name: Name of the function to extract

    Returns:
        The compiled function

    Raises:
        RuntimeError: If compilation or execution fails
    """
    sandbox_globals = get_sandbox_globals()

    try:
        # Compile the code
        compiled = compile(function_code, "<generated>", "exec")

        # Execute in sandbox
        exec(compiled, sandbox_globals)

        # Extract the function
        if function_name not in sandbox_globals:
            raise RuntimeError(
                f"Function '{function_name}' not found after executing code"
            )

        return sandbox_globals[function_name]

    except SyntaxError as e:
        raise RuntimeError(f"Syntax error in generated code: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error executing generated code: {e}") from e


def run_comparison(
    comparison_func: Callable,
    ref_audio: np.ndarray,
    candidate_audio: np.ndarray,
) -> float:
    """
    Run a comparison function safely and return the similarity score.

    Args:
        comparison_func: The comparison function to run
        ref_audio: Reference audio as numpy array
        candidate_audio: Candidate audio as numpy array

    Returns:
        Similarity score between 0.0 and 1.0

    Raises:
        RuntimeError: If comparison fails
    """
    try:
        result = comparison_func(ref_audio, candidate_audio)

        # Ensure result is a valid float in [0, 1]
        if not isinstance(result, (int, float)):
            raise RuntimeError(
                f"Comparison function returned {type(result).__name__}, expected float"
            )

        return float(np.clip(result, 0.0, 1.0))

    except Exception as e:
        raise RuntimeError(f"Comparison function failed: {e}") from e
