"""
api.py
FastAPI Backend for OpenEnv Hackathon
"""
import os
import json
import logging
import uvicorn
from typing import Any, Dict
from fastapi import FastAPI, HTTPException, Request

from env.models import Action, Observation, StepResult, ResetResult
from env.data_pipeline_env import DataPipelineEnv

app = FastAPI(title="Data Pipeline Debug Environment API", version="1.0.0")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State store (in memory)
active_sessions: Dict[str, DataPipelineEnv] = {}

def get_session(session_id: str = "default_session") -> DataPipelineEnv:
    if session_id not in active_sessions:
        active_sessions[session_id] = DataPipelineEnv()
    return active_sessions[session_id]

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/metadata")
async def metadata():
    return {
        "name": "data-pipeline-debug-env",
        "description": "Environment for debugging SQL data pipelines incrementally."
    }

@app.get("/schema")
async def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": Observation.model_json_schema()
    }

@app.post("/reset", response_model=ResetResult)
async def reset(request: Request):
    """Start or restart a session with a specific task."""
    body = await request.json()
    task = body.get("task", "easy")
    env = get_session()
    result = env.reset(task)
    return result

@app.post("/step", response_model=StepResult)
async def step(request: Request):
    """Execute to the next step."""
    body = await request.json()
    try:
        action = Action(**body)
    except Exception as e:
        # Pydantic validation failure represents an invalid LLM output structure
        return StepResult(
            observation=get_session()._get_obs(),
            reward=-0.20,
            done=False,
            info={"error": f"Invalid LLM format: {e}"}
        )

    env = get_session()
    result = env.step(action)
    return result

@app.get("/state", response_model=Observation)
async def state():
    """Get current observation state."""
    env = get_session()
    return env._get_obs()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=7860, reload=False)
