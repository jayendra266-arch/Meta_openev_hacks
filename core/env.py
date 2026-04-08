"""
core/env.py
===========
The main RL Environment — the "game world" the agent lives in.

This class follows the standard OpenAI Gym-style interface:
  - reset()  → start a new episode, return initial state
  - step()   → agent takes an action, environment returns (state, reward, done, info)

The environment wraps:
  - StateManager   : tracks current state
  - RewardEngine   : scores each action
  - Features       : query optimizer, validator, schema handler, explanation engine, self-healer
"""

from core.state_manager import StateManager
from core.reward_engine import RewardEngine
from features.query_optimizer import QueryOptimizer
from features.data_validator import DataValidator
from features.schema_handler import SchemaHandler
from features.explanation_engine import ExplanationEngine
from features.self_healing import SelfHealer

# The fixed order in which the agent MUST execute actions
ACTION_SEQUENCE = [
    "identify_issue",
    "suggest_fix",
    "optimize_query",
    "validate_data",
    "explain_fix",
    "final_answer",
]


class DebugEnvironment:
    """
    Simulates a data pipeline debugging environment.

    The agent steps through this environment one action at a time,
    receiving a reward for each correct debugging step.
    """

    def __init__(self, task: dict):
        """
        Set up the environment for a given debugging task.

        Args:
            task (dict): Task definition (from tasks/easy.py etc.)
        """
        self.task = task

        # Core modules
        self.state_manager = StateManager(task)
        self.reward_engine = RewardEngine()

        # Feature modules
        self.optimizer   = QueryOptimizer()
        self.validator   = DataValidator()
        self.schema_hdlr = SchemaHandler()
        self.explainer   = ExplanationEngine()
        self.healer      = SelfHealer()

        # Track episode state
        self.done = False
        self.current_step_index = 0     # Index into ACTION_SEQUENCE

        # Will store results from each step for printing
        self.results = {}

    def reset(self) -> dict:
        """
        Reset the environment for a new episode.

        Returns:
            dict: The initial state.
        """
        self.state_manager.reset(self.task)
        self.reward_engine.reset()
        self.done = False
        self.current_step_index = 0
        self.results = {}
        return self.state_manager.get_state()

    def step(self, action: str) -> tuple:
        """
        The agent performs an action. The environment evaluates it.

        Args:
            action (str): One of the 6 actions in ACTION_SEQUENCE.

        Returns:
            tuple: (next_state, reward, done, info)
              - next_state (dict): Updated state after action
              - reward (float): Score for this action
              - done (bool): True if episode is finished
              - info (dict): Extra diagnostic information
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # Enforce action ordering — agent must follow the fixed sequence
        expected_action = ACTION_SEQUENCE[self.current_step_index]
        if action != expected_action:
            # Wrong order = action rejected with penalty
            reward = self.reward_engine.compute_reward(expected_action, is_correct=False)
            info = {
                "error": f"Expected '{expected_action}', got '{action}'",
                "correct": False
            }
            return self.state_manager.get_state(), reward, False, info

        state = self.state_manager.get_state()
        is_correct, info = self._evaluate_action(action, state)

        # Score this action
        reward = self.reward_engine.compute_reward(action, is_correct)
        info["reward"] = reward
        info["correct"] = is_correct

        # Advance to next step
        self.current_step_index += 1
        self.state_manager.advance_step()

        # Episode ends after the 6th action
        if self.current_step_index >= len(ACTION_SEQUENCE):
            self.done = True

        return self.state_manager.get_state(), reward, self.done, info

    def _evaluate_action(self, action: str, state: dict) -> tuple:
        """
        Internal: evaluate whether the action is correct for the current state.

        Args:
            action (str): The action taken.
            state (dict): The current state.

        Returns:
            tuple: (is_correct: bool, info: dict)
        """
        task = self.task
        info = {}

        if action == "identify_issue":
            # Check if agent can correctly identify the issue type
            issue_type = task["issue_type"]
            issue_desc = task["expected_issue"]
            info["issue_found"] = issue_desc
            return True, info

        elif action == "suggest_fix":
            # The fix suggestion matches expected
            fix_desc = task["expected_fix"]
            info["fix_suggested"] = fix_desc
            return True, info

        elif action == "optimize_query":
            # Run the query through the optimizer
            optimized = self.optimizer.optimize(state["query"], task)
            info["original_query"] = state["query"]
            info["optimized_query"] = optimized

            # If optimization changed something, update the state's query
            if optimized != state["query"]:
                self.state_manager.update_query(optimized)

            return True, info

        elif action == "validate_data":
            # Check if the current (possibly healed) query passes validation
            current_query = self.state_manager.query
            healed = self.healer.heal(current_query, task)
            info["healed_query"] = healed

            # Update to healed query
            if healed != current_query:
                self.state_manager.update_query(healed)

            is_valid = self.validator.validate(healed, task)
            info["validation_result"] = "PASS" if is_valid else "FAIL"
            return is_valid, info

        elif action == "explain_fix":
            # Generate a human-readable explanation of what was wrong and how it was fixed
            explanation = self.explainer.explain(task, self.state_manager.query)
            info["explanation"] = explanation
            return True, info

        elif action == "final_answer":
            # Final answer: check if the current query matches the correct query
            final_query = self.state_manager.query
            correct_query = task["correct_query"]
            is_correct = (final_query.strip().lower() == correct_query.strip().lower())
            info["final_query"] = final_query
            info["expected_query"] = correct_query
            info["match"] = is_correct
            return is_correct, info

        return False, {"error": "Unknown action"}

    def get_reward_summary(self) -> dict:
        """Return the reward summary for this episode."""
        return self.reward_engine.get_summary()
