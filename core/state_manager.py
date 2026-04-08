"""
core/state_manager.py
=====================
Manages the current state of the RL environment for each episode.

The STATE is what the agent "sees" when deciding what action to take.
It includes:
  - error_log   : the error message from the failing pipeline
  - query        : the current SQL query (may be buggy or fixed)
  - schema       : the database schema (column names)
  - step         : which action step the agent is on (0 to 5)
"""


class StateManager:
    """
    Tracks and manages the agent's current state during an episode.

    Think of this like a notebook where the agent writes down
    what it currently knows about the problem.
    """

    def __init__(self, task: dict):
        """
        Initialize the state from a task definition.

        Args:
            task (dict): A task dict from tasks/easy.py, medium.py, etc.
        """
        # Start from the beginning of the task
        self.error_log = task["error_log"]
        self.query = task["query"]
        self.schema = task["schema"]
        self.step = 0   # Step index (0 = not started, 1 = first action done, etc.)

        # Store extra task metadata for the environment to use
        self.task = task

    def get_state(self) -> dict:
        """
        Return the current state as a dictionary.
        This is what the agent receives as input before each action.

        Returns:
            dict: State snapshot with all relevant info.
        """
        return {
            "error_log": self.error_log,
            "query":     self.query,
            "schema":    self.schema,
            "step":      self.step
        }

    def update_query(self, new_query: str):
        """
        Update the query (e.g., after self-healing or optimization).

        Args:
            new_query (str): The improved/fixed SQL query.
        """
        self.query = new_query

    def advance_step(self):
        """Move to the next step in the action sequence."""
        self.step += 1

    def reset(self, task: dict):
        """
        Reset state for a new episode (new task).

        Args:
            task (dict): The new task to initialize state from.
        """
        self.error_log = task["error_log"]
        self.query = task["query"]
        self.schema = task["schema"]
        self.step = 0
        self.task = task
