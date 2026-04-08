"""
env/models.py
=============
Pydantic v2 typed models for the OpenEnv specification.

Defines the structured types for:
  - Observation  : what the agent "sees" after each step
  - ActionType   : the 6 valid debugging actions (Enum)
  - Action       : the action the agent submits
  - StepResult   : what env.step() returns
  - ResetResult  : what env.reset() returns

These types are used by:
  - data_pipeline_env.py (internal logic)
  - server.py (FastAPI request/response bodies)
  - inference.py (parsing server responses)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────
# OBSERVATION — what the agent sees at each step
# ──────────────────────────────────────────────────────────────

class Observation(BaseModel):
    """
    The agent's view of the current environment state.

    Fields:
        error_log         : The error message from the broken pipeline.
        query             : Current SQL query (may be buggy or partially fixed).
        schema            : List of column names in the database.
        step              : Number of actions taken so far in this episode.
        completed_actions : Which of the 6 actions have been executed.
        total_reward      : Accumulated reward in this episode so far.
        current_query     : The most recently healed/optimized query.
    """

    error_log:         str            = Field(..., description="Error from the broken pipeline")
    query:             str            = Field(..., description="Current SQL query")
    schema:            List[str]      = Field(..., description="Database column names")
    step:              int            = Field(..., description="Number of steps taken")
    completed_actions: List[str]      = Field(default_factory=list,
                                              description="Actions already completed")
    total_reward:      float          = Field(default=0.0,
                                             description="Accumulated reward this episode")
    current_query:     str            = Field(default="",
                                             description="Most recent healed/optimized query")

    model_config = {"extra": "allow"}


# ──────────────────────────────────────────────────────────────
# ACTION TYPE — the 6 valid debugging actions
# ──────────────────────────────────────────────────────────────

class ActionType(str, Enum):
    """
    The six debugging actions an agent can take.

    Recommended order (following this order yields max reward = 1.0):
      1. identify_issue   → diagnose what is wrong
      2. suggest_fix      → propose a correction
      3. optimize_query   → improve query efficiency
      4. validate_data    → verify the fix works
      5. explain_fix      → explain what was done
      6. final_answer     → submit the final corrected query

    Out-of-order actions are allowed but incur a -0.1 penalty.
    Repeated actions incur a -0.1 penalty.
    Invalid action names incur a -0.2 penalty.
    """

    identify_issue  = "identify_issue"
    suggest_fix     = "suggest_fix"
    optimize_query  = "optimize_query"
    validate_data   = "validate_data"
    explain_fix     = "explain_fix"
    final_answer    = "final_answer"


# ──────────────────────────────────────────────────────────────
# ACTION — what the agent submits each step
# ──────────────────────────────────────────────────────────────

class Action(BaseModel):
    """
    The action submitted by the agent for each step.

    Fields:
        action_type : One of the 6 ActionType values.
        payload     : Optional free-text supporting the action
                      (e.g., the agent's explanation or proposed fix).
                      Not used for scoring — informational only.
    """

    action_type: ActionType        = Field(...,  description="The debugging action to perform")
    payload:     Optional[str]     = Field(None, description="Optional text supporting the action")

    model_config = {"extra": "forbid"}


# ──────────────────────────────────────────────────────────────
# STEP RESULT — returned by env.step()
# ──────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """
    What the environment returns after the agent submits an action.

    Fields:
        observation : Updated environment state after the action.
        reward      : Reward earned for this specific step (can be negative).
        done        : True if the episode is finished.
        info        : Diagnostic details (issue found, query healed, etc.).
    """

    observation: Observation       = Field(..., description="Updated state")
    reward:      float             = Field(..., description="Step reward (may be negative)")
    done:        bool              = Field(..., description="Is the episode finished?")
    info:        Dict[str, Any]    = Field(default_factory=dict, description="Step diagnostics")


# ──────────────────────────────────────────────────────────────
# RESET RESULT — returned by env.reset()
# ──────────────────────────────────────────────────────────────

class ResetResult(BaseModel):
    """
    What the environment returns after reset() is called.

    Fields:
        observation : Initial state for the new episode.
        done        : Always False after reset.
        info        : Task metadata (name, difficulty, issue_type).
    """

    observation: Observation       = Field(..., description="Initial observation")
    done:        bool              = Field(False, description="Always False after reset")
    info:        Dict[str, Any]    = Field(default_factory=dict, description="Task info")


# ──────────────────────────────────────────────────────────────
# RESET REQUEST — body for POST /reset
# ──────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """
    Request body for POST /reset.

    Fields:
        task : Task name to load. One of: easy, medium, hard, messy_schema.
               Defaults to "easy".
    """

    task: str = Field("easy", description="Task name: easy | medium | hard | messy_schema")
