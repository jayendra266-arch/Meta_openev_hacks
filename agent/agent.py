"""
agent/agent.py
==============
The RL Agent — the "brain" that interacts with the environment.

The agent:
  1. Receives state from the environment
  2. Uses its reasoner to understand the problem
  3. Checks memory to avoid past mistakes
  4. Selects the next action (always in fixed order)
  5. Receives reward and stores mistakes if reward was negative
  6. Learns over time by remembering what NOT to do

The agent ALWAYS executes actions in this exact order:
  identify_issue → suggest_fix → optimize_query →
  validate_data  → explain_fix → final_answer
"""

from agent.memory import Memory
from agent.reasoning import ReasoningModule

# The mandatory action sequence — never skip, never reorder
ACTION_SEQUENCE = [
    "identify_issue",
    "suggest_fix",
    "optimize_query",
    "validate_data",
    "explain_fix",
    "final_answer",
]


class DebugAgent:
    """
    Rule-based RL agent that debugs data pipelines step by step.

    Uses:
      - ReasoningModule : to understand the problem
      - Memory          : to avoid past mistakes
    """

    def __init__(self):
        """Initialize the agent with fresh memory and reasoning."""
        self.memory   = Memory()      # Remembers past mistakes
        self.reasoner = ReasoningModule()  # Analyzes state to decide actions

        # How many episodes the agent has run
        self.episode_count = 0

        # Track total rewards across all episodes
        self.cumulative_reward = 0.0

    def select_action(self, state: dict, step_index: int) -> str:
        """
        Select the next action based on the fixed sequence.

        The agent always follows the 6-step sequence in order.
        Memory is checked to flag if a past mistake would be repeated —
        but since the sequence is mandatory, the action is still taken
        (the environment enforces correctness through rewards).

        Args:
            state (dict): Current RL state.
            step_index (int): Which step in the sequence we're on (0–5).

        Returns:
            str: The chosen action name.
        """
        if step_index >= len(ACTION_SEQUENCE):
            return "final_answer"  # Safety fallback

        action = ACTION_SEQUENCE[step_index]

        # Check memory: has this action failed here before?
        if self.memory.was_mistake(state, action):
            count = self.memory.get_mistake_count(state, action)
            print(
                f"    🧠 [Memory] This action '{action}' was wrong {count}x before "
                f"at step {step_index} — but sequence is mandatory. Proceeding carefully."
            )

        return action

    def learn(self, state: dict, action: str, reward: float):
        """
        Update the agent's memory if a mistake was made (negative reward).

        This is the core of the "learning from mistakes" feature.

        Args:
            state (dict): The state at the time of the action.
            action (str): The action that was taken.
            reward (float): The reward received (negative = mistake).
        """
        if reward < 0:
            # Store this (state, action) as a mistake to avoid in the future
            self.memory.store_mistake(state, action)

    def run_episode(self, env) -> dict:
        """
        Run a complete 6-step debugging episode in the given environment.

        Args:
            env (DebugEnvironment): The RL environment to interact with.

        Returns:
            dict: Episode results including total score, step details, and success status.
        """
        self.episode_count += 1

        # Reset the environment and get the starting state
        state = env.reset()

        episode_log = []      # Track each step's result
        total_reward = 0.0
        step_index = 0

        done = False

        while not done:
            # Choose the next action (always follows fixed order)
            action = self.select_action(state, step_index)

            # State BEFORE the action (used for memory storage)
            state_before = dict(state)

            # Take the action in the environment
            next_state, reward, done, info = env.step(action)

            # Learn from the result
            self.learn(state_before, action, reward)

            # Build step log entry
            step_log = {
                "step":    step_index + 1,
                "action":  action,
                "reward":  reward,
                "correct": info.get("correct", True),
                "info":    info
            }
            episode_log.append(step_log)

            total_reward += reward
            state = next_state
            step_index += 1

        self.cumulative_reward += total_reward

        return {
            "task_name":    env.task["name"],
            "episode":      self.episode_count,
            "steps":        episode_log,
            "total_reward": round(total_reward, 2),
            "success":      total_reward >= 0.95,  # Success if nearly full score
        }
