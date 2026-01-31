"""
LLM-powered code generation for custom comparison functions.

Uses Claude API to generate Python functions that compare audio tracks
based on natural language descriptions.
"""

import os
import re
from datetime import datetime
from pathlib import Path

import anthropic

# Path to store generated functions
GENERATED_FUNCTIONS_PATH = Path(__file__).parent / "generated_functions.py"

PROMPT_TEMPLATE = """Your goal is to find songs similar in this manner: {user_prompt}

You have access to these functions that work on numpy audio arrays (mono, 44100 Hz sample rate):
- extract_hpcp_features(audio) -> 24-dim numpy array (HPCP mean + std, harmonic pitch class profile)
- extract_key_features(audio) -> 2-dim numpy array (key strength, mode: 0=minor, 1=major)
- extract_rhythm_features(audio) -> 16-dim numpy array (BPM, confidence, beat regularity, onset rate, onset stats, beat histogram)
- numpy (as np)
- essentia.standard - full Essentia DSP library

For detailed Essentia algorithms (pitch detection, onset detection, spectral analysis, etc.),
see: https://essentia.upf.edu/algorithms_reference.html

Write a Python function that takes (ref_audio, candidate_audio) where both are numpy arrays
of mono audio at 44100 Hz sample rate. The function should return a float from 0.0 to 1.0
representing the percentage similarity (1.0 = identical, 0.0 = completely different).

Give the function an appropriate descriptive name based on the comparison type (use snake_case).
Return ONLY the function code, no explanations or markdown formatting.

Example output format:
def rhythm_pattern_comparison(ref_audio, candidate_audio):
    # Extract rhythm features
    ref_rhythm = extract_rhythm_features(ref_audio)
    cand_rhythm = extract_rhythm_features(candidate_audio)
    # Compute similarity
    similarity = 1.0 - np.mean(np.abs(ref_rhythm - cand_rhythm))
    return float(np.clip(similarity, 0.0, 1.0))
"""


def generate_comparison_function(user_prompt: str) -> tuple[str, str]:
    """
    Generate a comparison function using Claude API.

    Args:
        user_prompt: Natural language description of how to compare songs

    Returns:
        Tuple of (function_name, function_code)

    Raises:
        ValueError: If ANTHROPIC_API_KEY is not set
        RuntimeError: If code generation fails or function cannot be parsed
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable is required. "
            "Set it with: export ANTHROPIC_API_KEY=sk-..."
        )

    client = anthropic.Anthropic(api_key=api_key)

    prompt = PROMPT_TEMPLATE.format(user_prompt=user_prompt)

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract the generated code
    code = message.content[0].text.strip()

    # Remove markdown code blocks if present
    code = re.sub(r"^```python\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    code = code.strip()

    # Extract function name
    match = re.search(r"def\s+(\w+)\s*\(", code)
    if not match:
        raise RuntimeError(
            f"Could not parse function name from generated code:\n{code}"
        )

    function_name = match.group(1)

    return function_name, code


def save_function(function_name: str, function_code: str, user_prompt: str) -> None:
    """
    Save a generated function to generated_functions.py.

    Args:
        function_name: Name of the function
        function_code: Full function source code
        user_prompt: Original natural language prompt
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create header comment
    header = f'''
# Generated: {timestamp}
# Prompt: "{user_prompt}"
'''

    # Read existing content or create file
    if GENERATED_FUNCTIONS_PATH.exists():
        existing = GENERATED_FUNCTIONS_PATH.read_text()
    else:
        existing = '"""Auto-generated comparison functions for lyrebird."""\n\nimport numpy as np\n\nfrom src.audio_analyzer import (\n    extract_hpcp_features,\n    extract_key_features,\n    extract_rhythm_features,\n)\n'

    # Check if function already exists
    if re.search(rf"def\s+{function_name}\s*\(", existing):
        # Remove the old version (including its header comment)
        pattern = rf"\n# Generated:.*?\n# Prompt:.*?\ndef {function_name}\s*\([^)]*\):.*?(?=\n# Generated:|\n\ndef |\Z)"
        existing = re.sub(pattern, "", existing, flags=re.DOTALL)

    # Append new function
    new_content = existing.rstrip() + "\n" + header + function_code + "\n"

    GENERATED_FUNCTIONS_PATH.write_text(new_content)


def list_saved_functions() -> list[dict]:
    """
    List all saved functions in generated_functions.py.

    Returns:
        List of dicts with 'name', 'prompt', and 'timestamp' keys
    """
    if not GENERATED_FUNCTIONS_PATH.exists():
        return []

    content = GENERATED_FUNCTIONS_PATH.read_text()

    # Find all function definitions with their headers
    pattern = r"# Generated: (.+?)\n# Prompt: \"(.+?)\"\ndef (\w+)\s*\("
    matches = re.findall(pattern, content)

    return [
        {"name": name, "prompt": prompt, "timestamp": timestamp}
        for timestamp, prompt, name in matches
    ]


def delete_function(function_name: str) -> bool:
    """
    Delete a function from generated_functions.py.

    Args:
        function_name: Name of the function to delete

    Returns:
        True if function was deleted, False if not found
    """
    if not GENERATED_FUNCTIONS_PATH.exists():
        return False

    content = GENERATED_FUNCTIONS_PATH.read_text()

    # Check if function exists
    if not re.search(rf"def\s+{function_name}\s*\(", content):
        return False

    # Remove the function and its header comment
    pattern = rf"\n# Generated:.*?\n# Prompt:.*?\ndef {function_name}\s*\([^)]*\):.*?(?=\n# Generated:|\n\ndef |\Z)"
    new_content = re.sub(pattern, "", content, flags=re.DOTALL)

    GENERATED_FUNCTIONS_PATH.write_text(new_content)
    return True
