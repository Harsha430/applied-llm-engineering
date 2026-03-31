import re
from langchain_core.tools import tool

def extract_numbers(query: str):
    nums = re.findall(r'-?\d+\.?\d*', query)
    if len(nums) < 2:
        raise ValueError(f"Could not extract two numbers from input: {query}")
    return float(nums[0]), float(nums[1])

@tool
def add(query: str):
    """Add two numbers together. Input MUST be a comma separated string of two numbers, e.g. '2, 3'."""
    a, b = extract_numbers(query)
    return float(a + b)

@tool
def sub(query: str):
    """Subtract the second number from the first. Input MUST be a comma separated string of two numbers, e.g. '5, 2'."""
    a, b = extract_numbers(query)
    return float(a - b)

@tool
def mul(query: str):
    """Multiply two numbers together. Input MUST be a comma separated string of two numbers, e.g. '4, 5'."""
    a, b = extract_numbers(query)
    return float(a * b)

@tool
def div(query: str):
    """Divide the first number by the second. Input MUST be a comma separated string of two numbers, e.g. '10, 2'."""
    a, b = extract_numbers(query)
    return float(a / b) if b != 0 else "Error: Division by zero."