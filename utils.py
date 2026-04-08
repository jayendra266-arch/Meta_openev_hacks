"""
utils.py
========
Utility functions shared across the entire DEBUG_ENV project.

Contains:
  - Formatting helpers
  - Color codes for terminal output (optional, cross-platform safe)
  - Separator line generators
"""


def separator(char: str = "=", width: int = 60) -> str:
    """
    Generate a separator line for display.

    Args:
        char (str): Character to fill the line with.
        width (int): Total width of the line.

    Returns:
        str: A separator string.
    """
    return char * width


def format_reward(reward: float) -> str:
    """
    Format a reward value with a leading + or - sign.

    Args:
        reward (float): The reward value.

    Returns:
        str: Formatted reward string (e.g., '+0.30' or '-0.20').
    """
    if reward >= 0:
        return f"+{reward:.2f}"
    return f"{reward:.2f}"


def format_score(score: float) -> str:
    """
    Format a score to 2 decimal places.

    Args:
        score (float): The score value.

    Returns:
        str: Formatted score string (e.g., '1.00').
    """
    return f"{score:.2f}"


def print_header(title: str, width: int = 60):
    """
    Print a centered header with separator lines above and below.

    Args:
        title (str): The header title text.
        width (int): Width for centering.
    """
    print()
    print(separator("=", width))
    print(title.center(width))
    print(separator("=", width))


def truncate(text: str, max_len: int = 80) -> str:
    """
    Truncate long strings for display.

    Args:
        text (str): The string to truncate.
        max_len (int): Maximum display length.

    Returns:
        str: Truncated string with '...' appended if shortened.
    """
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."
