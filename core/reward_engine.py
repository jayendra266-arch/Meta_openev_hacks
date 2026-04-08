"""
core/reward_engine.py
=====================
Defines the reward system used to score agent actions.

The reward tells the agent if it did the right thing.
Positive reward = good action (move toward correct fix).
Negative reward = wrong action (repeat mistake or bad reasoning).

REWARD TABLE:
  identify_issue  → +0.30 / -0.20
  suggest_fix     → +0.30 / -0.20
  optimize_query  → +0.20 / -0.10
  validate_data   → +0.10 / -0.10
  explain_fix     → +0.05
  final_answer    → +0.05 / -0.30

Max score per episode = 1.00
"""

# -----------------------------------------------------------------
# Reward values for each action (positive = correct, negative = wrong)
# -----------------------------------------------------------------
REWARD_TABLE = {
    "identify_issue": {"correct": +0.30, "wrong": -0.20},
    "suggest_fix":    {"correct": +0.30, "wrong": -0.20},
    "optimize_query": {"correct": +0.20, "wrong": -0.10},
    "validate_data":  {"correct": +0.10, "wrong": -0.10},
    "explain_fix":    {"correct": +0.05, "wrong":  0.00},  # Always awarded (effort)
    "final_answer":   {"correct": +0.05, "wrong": -0.30},
}


class RewardEngine:
    """
    Computes and tracks rewards for every step in an episode.
    """

    def __init__(self):
        """Initialize the reward engine with zero totals."""
        self.total_reward = 0.0       # Accumulated score for this episode
        self.step_rewards = []        # List of (action, reward, correct) tuples

    def compute_reward(self, action: str, is_correct: bool, explicit_reward: float = None) -> float:
        """
        Look up the reward for this action or use an explicit value.

        Args:
            action (str): The action the agent took.
            is_correct (bool): Whether the action was correct.
            explicit_reward (float): Optional overrides the table reward.

        Returns:
            float: The reward value.
        """
        if action not in REWARD_TABLE and explicit_reward is None:
            return -0.05

        if explicit_reward is not None:
            reward = explicit_reward
        else:
            reward = REWARD_TABLE[action]["correct"] if is_correct else REWARD_TABLE[action]["wrong"]

        self.total_reward += reward
        self.step_rewards.append({
            "action":  action,
            "reward":  reward,
            "correct": is_correct
        })

        return reward

    def reset(self):
        """Reset the reward tracker for a new episode."""
        self.total_reward = 0.0
        self.step_rewards = []

    def get_summary(self) -> dict:
        """
        Return a summary of this episode's reward history.

        Returns:
            dict: Total reward and per-step breakdown.
        """
        return {
            "total_reward": round(self.total_reward, 2),
            "steps":        self.step_rewards
        }
