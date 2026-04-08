"""
inference.py
============
Hackathon-compliant baseline inference script for the
Data Pipeline Debug Environment.

MANDATORY STDOUT FORMAT:
  [START] task=<name> env=data-pipeline-debug-env model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

ENVIRONMENT VARIABLES:
  HF_TOKEN        — Hugging Face API key (required for LLM calls)
  API_BASE_URL    — LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME      — Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  ENV_SERVER_URL  — Where the env server is running (default: http://localhost:7860)

WORKFLOW:
  1. Connects to the FastAPI environment server (api.py)
  2. Runs tasks: easy → medium → hard
  3. Uses OpenAI client to let the LLM pick each action
  4. Prints exact [START]/[STEP]/[END] log format
  5. Score = total_reward / max_possible_reward (clamped 0.0–1.0)

USAGE:
  # 1. Start the server (in a separate terminal):
  #    python api.py
  # 2. Run inference:
  #    HF_TOKEN=hf_xxx python inference.py

  # Or with Docker:
  #    docker run -p 7860:7860 data-pipeline-env
  #    HF_TOKEN=hf_xxx python inference.py
"""

from __future__ import annotations

import os
import sys
import io
import json
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ── Force UTF-8 output ───────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — read from environment variables
# ─────────────────────────────────────────────────────────────

HF_TOKEN       = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY", "dummy_for_proxy")
API_BASE_URL   = os.getenv("API_BASE_URL",   "https://router.huggingface.co/v1")
MODEL_NAME     = os.getenv("MODEL_NAME",     "Qwen/Qwen2.5-72B-Instruct")
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:7860").rstrip("/")

# Tasks to run (in order)
TASKS         = ["easy", "medium", "hard"]
BENCHMARK     = "data-pipeline-debug-env"
MAX_STEPS     = 8           # Max actions per episode (6 needed, buffer for mistakes)
MAX_REWARD    = 1.0          # Maximum possible reward per task
SUCCESS_THRESHOLD = 0.5     # score >= this → success = true
TEMPERATURE   = 0.0          # Deterministic LLM outputs
MAX_TOKENS    = 20           # Only need short action names

# The 6 valid action names
VALID_ACTIONS = [
    "identify_issue",
    "suggest_fix",
    "optimize_query",
    "validate_data",
    "explain_fix",
    "final_answer",
]

# Recommended action sequence (used as guidance in prompt)
CORRECT_SEQUENCE = VALID_ACTIONS  # same order


# ─────────────────────────────────────────────────────────────
# LOGGING (mandatory stdout format)
# ─────────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    """Print the mandatory [START] line."""
    print(
        f"[START] task={task} env={BENCHMARK} model={model}",
        flush=True,
    )


def log_step(
    step:   int,
    action: str,
    reward: float,
    done:   bool,
    error:  Optional[str],
) -> None:
    """Print a mandatory [STEP] line."""
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(
    success: bool,
    steps:   int,
    score:   float,
    rewards: List[float],
) -> None:
    """Print the mandatory [END] line."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT CLIENT — HTTP calls to api.py
# ─────────────────────────────────────────────────────────────

def env_reset(task_name: str) -> Dict[str, Any]:
    """POST /env/reset to start a new episode."""
    resp = requests.post(
        f"{ENV_SERVER_URL}/env/reset",
        json={"task": task_name},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, payload: Optional[str] = None) -> Dict[str, Any]:
    """POST /env/step to submit an action."""
    body = {"action_type": action_type}
    if payload:
        body["payload"] = payload
    resp = requests.post(
        f"{ENV_SERVER_URL}/env/step",
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_state() -> Dict[str, Any]:
    """GET /env/state to inspect current env state."""
    resp = requests.get(f"{ENV_SERVER_URL}/env/state", timeout=10)
    resp.raise_for_status()
    return resp.json()


def wait_for_server(retries: int = 10, delay: float = 2.0) -> bool:
    """
    Wait for the environment server to be ready.
    Retries every `delay` seconds up to `retries` times.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{ENV_SERVER_URL}/env/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(
            f"[DEBUG] Server not ready (attempt {attempt}/{retries}). "
            f"Waiting {delay}s...",
            flush=True,
        )
        time.sleep(delay)
    return False


# ─────────────────────────────────────────────────────────────
# LLM ACTION SELECTION — uses OpenAI client
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a data engineering expert debugging broken SQL pipelines.
    You must debug the pipeline by taking actions in this recommended order:

      1. identify_issue   — Identify what is wrong in the query/schema
      2. suggest_fix      — Propose the specific fix needed
      3. optimize_query   — Optimize the SQL query
      4. validate_data    — Validate the fix is correct
      5. explain_fix      — Explain what was fixed and why
      6. final_answer     — Submit the final corrected query

    You will be given:
      - error_log: what went wrong
      - query: the current SQL query
      - schema: available column names
      - step: how many actions have been taken
      - completed_actions: which actions are already done

    Reply with EXACTLY ONE action name from this list:
      identify_issue, suggest_fix, optimize_query,
      validate_data, explain_fix, final_answer

    Rules:
      - Output ONLY the action name — nothing else
      - No punctuation, no quotes, no explanation
      - DO NOT repeat the same action twice.
      - Follow strict sequence: identify_issue -> suggest_fix -> optimize_query -> validate_data -> explain_fix -> final_answer
      - Always call final_answer only once and only at the end
""").strip()


def build_user_prompt(obs: Dict[str, Any], step: int) -> str:
    """Build the user message for the LLM based on current observation."""
    completed = obs.get("completed_actions", [])
    remaining = [a for a in CORRECT_SEQUENCE if a not in completed]

    return textwrap.dedent(f"""
        Current State:
          error_log : {obs.get('error_log', 'N/A')}
          query     : {obs.get('query', 'N/A')}
          schema    : {obs.get('schema', [])}
          step      : {step}
          completed : {completed}
          remaining : {remaining}
          reward_so_far: {obs.get('total_reward', 0.0):.2f}

        What is the next action? (output only the action name)
    """).strip()


def get_llm_action(
    client:    OpenAI,
    obs:       Dict[str, Any],
    step:      int,
    completed: List[str],
) -> str:
    """
    Call the LLM to pick the next debugging action.

    Falls back to the next action in the correct sequence if:
      - The LLM call fails
      - The LLM output is not a valid action name
      - The LLM picks an already-completed action

    Args:
        client:    OpenAI client instance.
        obs:       Current observation dict.
        step:      Current step number (1-based).
        completed: List of already-completed action names.

    Returns:
        A valid action name string.
    """
    # Determine fallback: next action not yet completed
    fallback = next(
        (a for a in CORRECT_SEQUENCE if a not in completed),
        "final_answer",
    )

    try:
        if client is None:
            raise ValueError("OpenAI client not initialized.")

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(obs, step)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw_output = (completion.choices[0].message.content or "").strip().lower()

        # Find first valid action in the LLM output
        for action in VALID_ACTIONS:
            if action in raw_output:
                if action not in completed:
                    return action

        # If the action returned is already completed, use fallback
        print(
            f"[DEBUG] LLM output '{raw_output}' not usable. Using fallback: {fallback}",
            flush=True,
        )
        return fallback

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}. Using fallback: {fallback}", flush=True)
        return fallback


# ─────────────────────────────────────────────────────────────
# TASK RUNNER
# ─────────────────────────────────────────────────────────────

def run_task(client: OpenAI, task_name: str) -> Dict[str, Any]:
    """
    Run one complete episode for the given task.

    Args:
        client:    OpenAI client.
        task_name: Task to run ('easy', 'medium', 'hard').

    Returns:
        Dict with steps, score, success, and all rewards.
    """
    rewards:      List[float] = []
    steps_taken:  int         = 0
    score:        float       = 0.0
    success:      bool        = False
    last_error:   Optional[str] = None

    log_start(task=task_name, model=MODEL_NAME)

    try:
        # ── Reset environment ─────────────────────────────────
        reset_result = env_reset(task_name)
        obs          = reset_result.get("observation", {})
        completed    = obs.get("completed_actions", [])

        for step_num in range(1, MAX_STEPS + 1):
            # Episode already done (e.g. all 6 actions completed)
            if reset_result.get("done", False):
                break

            # ── LLM picks action ──────────────────────────────
            action_str = get_llm_action(client, obs, step_num, completed)

            # ── Step the environment ──────────────────────────
            try:
                step_result = env_step(action_str)
            except requests.HTTPError as http_err:
                last_error = str(http_err)
                log_step(
                    step=step_num, action=action_str,
                    reward=0.0, done=False, error=last_error,
                )
                break

            reward    = step_result.get("reward",  0.0)
            done      = step_result.get("done",    False)
            info      = step_result.get("info",    {})
            obs       = step_result.get("observation", obs)
            completed = obs.get("completed_actions", [])
            last_error = info.get("error")   # None if no error

            rewards.append(reward)
            steps_taken = step_num

            log_step(
                step=step_num,
                action=action_str,
                reward=reward,
                done=done,
                error=last_error,
            )

            if done:
                break

        # ── Compute final score ───────────────────────────────
        total_reward = sum(rewards)
        score        = min(max(total_reward / MAX_REWARD, 0.0), 1.0)
        success      = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        last_error = str(exc)
        print(f"[DEBUG] Task '{task_name}' error: {exc}", flush=True)

    finally:
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return {
        "task":    task_name,
        "steps":   steps_taken,
        "score":   score,
        "success": success,
        "rewards": rewards,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main() -> None:
    """
    Main entry point: run all tasks and print results.
    """
    # ── Validate credentials ──────────────────────────────────
    if not HF_TOKEN:
        print(
            "[DEBUG] WARNING: HF_TOKEN not set. "
            "LLM calls may fail without a valid API key.",
            flush=True,
        )

    # ── Wait for environment server ───────────────────────────
    print(f"[DEBUG] Connecting to env server at {ENV_SERVER_URL} ...", flush=True)
    server_ready = wait_for_server(retries=15, delay=2.0)
    if not server_ready:
        print(
            f"[DEBUG] FATAL: Server at {ENV_SERVER_URL} did not respond. "
            "Start it with: python api.py",
            flush=True,
        )
        sys.exit(1)

    print(f"[DEBUG] Server ready. Running {len(TASKS)} tasks...", flush=True)

    # ── OpenAI client ─────────────────────────────────────────
    try:
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    except Exception as e:
        print(f"[DEBUG] Failed to initialize OpenAI client: {e}. Running blind...", flush=True)
        # We must not crash! Pass a dummy client if OpenAI enforces strict logic.
        client = None

    # ── Run all tasks ─────────────────────────────────────────
    all_results = []
    for task_name in TASKS:
        result = run_task(client, task_name)
        all_results.append(result)
        print()   # blank line between tasks

    # ── Summary ───────────────────────────────────────────────
    print("=" * 52, flush=True)
    print("  BASELINE INFERENCE SUMMARY", flush=True)
    print("=" * 52, flush=True)
    print(f"  {'Task':<16} {'Score':<8} {'Steps':<8} {'Success'}", flush=True)
    print("-" * 52, flush=True)
    for r in all_results:
        print(
            f"  {r['task']:<16} {r['score']:.3f}    "
            f"{r['steps']:<8} {str(r['success']).lower()}",
            flush=True,
        )
    overall = sum(r["score"] for r in all_results) / len(all_results)
    print("-" * 52, flush=True)
    print(f"  Overall Average Score: {overall:.3f}", flush=True)
    print("=" * 52, flush=True)


if __name__ == "__main__":
    main()
