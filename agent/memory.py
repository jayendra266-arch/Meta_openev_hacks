"""
agent/memory.py
===============
⭐ NEW FEATURE: Learning from Mistakes

The Memory class stores wrong decisions the agent made,
so it can avoid repeating them in the future.

How it works:
  1. When the agent takes a wrong action, Memory.store_mistake() is called.
  2. The mistake is saved as (state_key, action) → "avoid this!"
  3. Before choosing an action, the agent checks Memory.was_mistake() 
     to see if it already tried this and it didn't work.
  4. If the same mistake would be repeated, the agent picks a different action.

This simple mechanism teaches the agent to improve over episodes.
"""


class Memory:
    """
    Stores wrong (action, state) pairs to avoid repeating mistakes.

    Uses a dictionary where:
      key   = a string representation of the state + action
      value = the number of times this mistake was made
    """

    def __init__(self):
        """Initialize an empty mistake memory store."""
        # Dictionary: { state_key: { action: mistake_count } }
        self.mistakes = {}

        # Total number of mistakes ever stored
        self.total_mistakes = 0

    def _make_key(self, state: dict) -> str:
        """
        Turn a state dict into a short string key for lookup.

        We use issue_type + step as the key because the same issue at the
        same step should be recognizable across episodes.

        Args:
            state (dict): Current RL state (error_log, query, schema, step).

        Returns:
            str: A short, consistent string key.
        """
        # Extract key identifying features from the state
        error_snippet = state.get("error_log", "")[:30]   # First 30 chars of error
        step = state.get("step", 0)
        return f"step={step}|err={error_snippet}"

    def store_mistake(self, state: dict, action: str):
        """
        Record that this (state, action) combination led to a wrong result.

        Args:
            state (dict): The state at the time of the mistake.
            action (str): The action that caused the negative reward.
        """
        key = self._make_key(state)

        # Initialize if this state hasn't been seen before
        if key not in self.mistakes:
            self.mistakes[key] = {}

        # Increment the mistake count for this (state, action) pair
        if action not in self.mistakes[key]:
            self.mistakes[key][action] = 0
        self.mistakes[key][action] += 1

        self.total_mistakes += 1

    def was_mistake(self, state: dict, action: str) -> bool:
        """
        Check if this action was previously made as a mistake in this state.

        Args:
            state (dict): The current state.
            action (str): The action being considered.

        Returns:
            bool: True if this action caused a mistake before, False otherwise.
        """
        key = self._make_key(state)
        if key in self.mistakes and action in self.mistakes[key]:
            return True
        return False

    def get_mistake_count(self, state: dict, action: str) -> int:
        """
        How many times has the agent made this mistake?

        Args:
            state (dict): The current state.
            action (str): The action being considered.

        Returns:
            int: Number of times this mistake was made (0 if never).
        """
        key = self._make_key(state)
        return self.mistakes.get(key, {}).get(action, 0)

    def summary(self) -> str:
        """
        Return a human-readable summary of all stored mistakes.

        Returns:
            str: Summary text of the mistake memory.
        """
        if not self.mistakes:
            return "Memory is empty — no mistakes recorded yet."

        lines = [f"📚 Memory: {self.total_mistakes} mistake(s) stored"]
        for state_key, actions in self.mistakes.items():
            for action, count in actions.items():
                lines.append(f"  [{state_key}] action='{action}' → {count}x wrong")

        return "\n".join(lines)
