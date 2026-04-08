from __future__ import annotations
from typing import Any, Dict, List, Tuple

from env.models import Action, Observation, ResetResult, StepResult
from features.query_optimizer import QueryOptimizer
from features.data_validator import DataValidator
from features.schema_handler import SchemaHandler
from features.explanation_engine import ExplanationEngine
from features.self_healing import SelfHealer

from tasks.easy import EASY_TASK
from tasks.medium import MEDIUM_TASK
from tasks.hard import HARD_TASK
from tasks.messy_schema import MESSY_TASK

TASK_REGISTRY = {"easy": EASY_TASK, "medium": MEDIUM_TASK, "hard": HARD_TASK, "messy_schema": MESSY_TASK}
CORRECT_SEQUENCE = ["identify_issue", "suggest_fix", "optimize_query", "validate_data", "explain_fix", "final_answer"]

REWARD_TABLE = {
    "identify_issue": (0.30, -0.20),
    "suggest_fix": (0.30, -0.20),
    "optimize_query": (0.20, -0.10),
    "validate_data": (0.10, -0.10),
    "explain_fix": (0.05, 0.00),
    "final_answer": (0.05, -0.30),
}

MAX_STEPS = 10
PENALTY_REPEAT = -0.10
PENALTY_ORDER = -0.10

class DataPipelineEnv:
    def __init__(self):
        self.optimizer, self.validator, self.schema_hdlr = QueryOptimizer(), DataValidator(), SchemaHandler()
        self.explainer, self.healer = ExplanationEngine(), SelfHealer()
        self.reset()

    def reset(self, task_name: str = "easy") -> ResetResult:
        if task_name not in TASK_REGISTRY: raise ValueError(f"Task {task_name} unknown")
        self.task, self.task_name = TASK_REGISTRY[task_name], task_name
        self.current_query = self.task["query"]
        self.completed, self.total_reward, self.step_count, self.done = [], 0.0, 0, False
        return ResetResult(observation=self._obs(), done=False, info={"task": task_name, "difficulty": self.task["difficulty"]})

    def step(self, action: Action) -> StepResult:
        if self.done: return StepResult(observation=self._obs(), reward=0.0, done=True, info={"error": "Episode Done"})

        name = action.action_type.value
        payload = (action.payload or "").lower()

        # Termination Safety
        if self.step_count >= MAX_STEPS:
            self.done = True
            return StepResult(observation=self._obs(), reward=0.0, done=True, info={"error": "Max steps exceeded"})

        # Guard: Repetition
        if name in self.completed:
            reward = PENALTY_REPEAT
            self.total_reward += reward
            self.step_count += 1
            return StepResult(observation=self._obs(), reward=reward, done=False, info={"error": "Repeated action"})

        # Order Check
        expected = CORRECT_SEQUENCE[len(self.completed)]
        penalty = PENALTY_ORDER if name != expected else 0.0

        # Evaluate (Honest Reward Shaping)
        correct, base_reward, info = self._evaluate(name, payload)
        reward = round(base_reward + penalty, 2)

        self.completed.append(name)
        self.step_count += 1
        self.total_reward += reward

        if len(self.completed) == len(CORRECT_SEQUENCE): self.done = True

        # Enriched Info Schema
        step_info = {
            "correct": correct,
            "confidence": round(abs(reward), 2),
            "progress": round(len(self.completed) / 6, 2),
            "hint": None if correct else "Check query logic or schema mismatch",
            **info
        }

        return StepResult(observation=self._obs(), reward=reward, done=self.done, info=step_info)

    def _evaluate(self, name: str, payload: str) -> Tuple[bool, float, Dict]:
        task = self.task
        
        # 1. Honest Reasoning Fallback (0.15 instead of 0.30)
        if name in ["identify_issue", "suggest_fix"]:
            if not payload: return True, 0.15, {"info": "Partial credit (Missing payload)"}
            patterns = task.get("expected_issue_type" if name=="identify_issue" else "expected_fix_pattern", [])
            matches = [t for t in patterns if t.lower() in payload]
            ratio = len(matches) / len(patterns) if patterns else 1.0
            return (ratio >= 0.5), (0.30 if ratio >= 0.5 else -0.20), {"matches": matches}

        if name == "optimize_query":
            q = self.optimizer.optimize(self.current_query, task)
            self.current_query = q
            pts = task.get("expected_query_pattern", [])
            matches = [p for p in pts if p.lower() in q.lower()]
            reward = 0.20 if len(matches) == len(pts) else (0.10 if len(matches) >= 1 else -0.10)
            return (reward > 0), reward, {}

        if name == "validate_data":
            self.current_query = self.healer.heal(self.current_query, task)
            valid = self.validator.validate(self.current_query, task)
            return valid, (0.10 if valid else -0.10), {}

        if name == "explain_fix": return True, 0.05, {}

        # 2. Hard Multi-Condition Validation
        if name == "final_answer":
            if self.task_name == "hard":
                conds = ["join" in self.current_query.lower(), "sum(" in self.current_query.lower(), "as " in self.current_query.lower()]
                score = sum(conds) / len(conds)
                if score == 1: reward = 0.05
                elif score >= 0.66: reward = 0.03
                else: reward = -0.30
                return (reward > 0), reward, {"completion_score": round(score, 2)}
            
            correct = (self.current_query.strip().lower() == task["correct_query"].strip().lower())
            return correct, (0.05 if correct else -0.30), {}

        return False, -0.20, {}

    def _obs(self) -> Observation:
        # 3. Partial Observability (Mask schema until step 2)
        schema = self.task.get("schema", []) if self.step_count >= 2 else []
        return Observation(error_log=self.task.get("error_log", ""), query=self.task.get("query", ""), schema=schema, 
                           step=self.step_count, completed_actions=self.completed, total_reward=round(self.total_reward, 2), 
                           current_query=self.current_query)
