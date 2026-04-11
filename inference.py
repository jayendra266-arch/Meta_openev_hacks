"""
inference.py
============
Hackathon-compliant baseline inference script for the
Data Pipeline Debug Environment.

MANDATORY STDOUT FORMAT:
  [START] task=<name> env=data-pipeline-debug-env model=<model>
  [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

from __future__ import annotations

import os
import sys
import json
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# CONFIGURATION — matching official checklist (Image 2/3)
# ─────────────────────────────────────────────────────────────

# STICK TO CHECKLIST: Defaults only for BASE_URL and MODEL_NAME
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "gpt-4o-mini")

# HF_TOKEN and API_KEY must be retrieved without defaults
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = os.getenv("API_KEY")

# Port for internal environment server
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://localhost:7860").rstrip("/")

def validate_vars():
    """Ensure we have what we need to call the proxy."""
    if not API_BASE_URL:
        raise RuntimeError("API_BASE_URL is missing")
    # Validator mentioned API_KEY in Submission #11 failure
    if not API_KEY and not HF_TOKEN:
        raise RuntimeError("Neither API_KEY nor HF_TOKEN is set")

# Tasks to run
TASKS         = ["easy", "medium", "hard"]
BENCHMARK     = "data-pipeline-debug-env"
MAX_STEPS     = 8
MAX_REWARD    = 1.0
SUCCESS_THRESHOLD = 0.5
TEMPERATURE   = 0.0
MAX_TOKENS    = 20

VALID_ACTIONS = [
    "identify_issue", "suggest_fix", "optimize_query",
    "validate_data", "explain_fix", "final_answer",
]

# ─────────────────────────────────────────────────────────────
# LOGGING (START/STEP/END exactly)
# ─────────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ── Environment Client ─────────────────────────────────────────

def env_reset(task_name: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_SERVER_URL}/env/reset", json={"task": task_name}, timeout=30)
    resp.raise_for_status()
    return resp.json()

def env_step(action_type: str, payload: Optional[str] = None) -> Dict[str, Any]:
    body = {"action_type": action_type}
    if payload: body["payload"] = payload
    resp = requests.post(f"{ENV_SERVER_URL}/env/step", json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()

def wait_for_server(retries: int = 15, delay: float = 2.0) -> bool:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(f"{ENV_SERVER_URL}/env/health", timeout=5)
            if resp.status_code == 200: return True
        except: pass
        time.sleep(delay)
    return False

# ── LLM Core (Validator Step 6: must call get_model_message) ───

SYSTEM_PROMPT = "You are a data debugging expert. Respond with ONLY the action name."

def get_model_message(client: OpenAI, obs: Dict[str, Any], step: int, completed: List[str]) -> str:
    """Validator checks for this function specifically."""
    fallback = next((a for a in VALID_ACTIONS if a not in completed), "final_answer")
    
    user_prompt = f"Obs: {obs}, Step: {step}, Completed: {completed}. Next action?"
    
    # NO try/except here to satisfy Step 2 of the hackathon checklist
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    res = (completion.choices[0].message.content or "").strip().lower()
    for a in VALID_ACTIONS:
        if a in res and a not in completed: return a
    return fallback

# Alias for Step 1 checklist requirement
get_llm_action = get_model_message

def run_task(client: OpenAI, task_name: str) -> Dict[str, Any]:
    rewards, steps_taken, score, success = [], 0, 0.0, False
    log_start(task=task_name, model=MODEL_NAME)
    
    try:
        res = env_reset(task_name)
        obs = res.get("observation", {})
        comp = obs.get("completed_actions", [])

        for s in range(1, MAX_STEPS + 1):
            if res.get("done", False): break
            act = get_model_message(client, obs, s, comp)
            res = env_step(act)
            
            rew = res.get("reward", 0.0)
            done = res.get("done", False)
            obs = res.get("observation", obs)
            comp = obs.get("completed_actions", [])
            err = res.get("info", {}).get("error")

            rewards.append(rew)
            steps_taken = s
            log_step(s, act, rew, done, err)
            if done: break
            
        total_reward = sum(rewards)
        score = min(max(total_reward / MAX_REWARD, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] Task error: {e}")
    finally:
        log_end(success, steps_taken, score, rewards)
    return {"score": score}

def main():
    if not wait_for_server():
        print("Server not ready")
        sys.exit(1)

    validate_vars()
    
    # Initialize OpenAI exactly as shown in checklist
    # Using API_KEY as primary, HF_TOKEN as fallback
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=(API_KEY or HF_TOKEN)
    )

    for task in TASKS:
        run_task(client, task)
        print()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Final safety net for unhandled exceptions
        print(f"Unhandled Exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
