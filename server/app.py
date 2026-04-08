"""
api.py
FastAPI Backend for OpenEnv Hackathon - Multi-Mode Production Version
"""
import os
import sys
from pathlib import Path

# Important: Add root directory to path since app is now in server/
sys.path.append(str(Path(__file__).parent.parent))

import json
import logging
import uvicorn
from typing import Any, Dict, Optional
import logging

from fastapi import FastAPI, HTTPException, Request, Body, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# --- HOTFIX FOR GRADIO 5.1.0 & HUGGINGFACE-HUB 1.9.2 CRASH ---
# huggingface-hub removed HfFolder in 1.9.x, but Gradio 5.1.0 still tries to import it.
import huggingface_hub
if not hasattr(huggingface_hub, 'HfFolder'):
    class HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(token): pass
    huggingface_hub.HfFolder = HfFolder

import gradio as gr

from env.models import Action, Observation, StepResult, ResetResult, ResetRequest
from env.data_pipeline_env import DataPipelineEnv
from app import demo

# ─────────────────────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Data Pipeline Debug Environment API",
    version="1.0.0",
    description="Multi-mode agentic RL environment for SQL debugging."
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# State store (in memory)
active_sessions: Dict[str, DataPipelineEnv] = {}

def get_session(session_id: str = "default_session") -> DataPipelineEnv:
    if session_id not in active_sessions:
        active_sessions[session_id] = DataPipelineEnv()
    return active_sessions[session_id]

# ─────────────────────────────────────────────────────────────
# MIDDLEWARE & ERROR HANDLING
# ─────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Ensure no raw 500 errors reach the caller/platform."""
    logger.error(f"Global error caught: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal Server Error", "detail": str(exc)},
    )

# ─────────────────────────────────────────────────────────────
# CORE API ROUTER (Redundant Paths)
# ─────────────────────────────────────────────────────────────

router = APIRouter()

@router.get("/health", tags=["General"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "benchmark": "data-pipeline-debug-env"}

@router.get("/metadata", tags=["General"])
async def metadata():
    return {
        "name": "data-pipeline-debug-env",
        "description": "Environment for debugging SQL data pipelines incrementally."
    }

@router.get("/schema")
async def schema():
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": Observation.model_json_schema()
    }

@router.post("/reset", response_model=ResetResult)
async def reset(payload: Optional[ResetRequest] = Body(None)):
    """Start or restart a session with a specific task."""
    task = payload.task if payload else "easy"
    logger.info(f"Resetting environment with task: {task}")
    env = get_session()
    result = env.reset(task)
    return result

@router.post("/step", response_model=StepResult)
async def step(request: Request):
    """Execute to the next step."""
    try:
        # Handle empty/missing bodies safely
        body = await request.json()
    except Exception:
        body = {}

    try:
        action = Action(**body)
    except Exception as e:
        # Graceful handling of invalid action structures (Phase 2 compliance)
        logger.warning(f"Invalid action format: {e}")
        return StepResult(
            observation=get_session()._obs(),
            reward=-0.20,
            done=False,
            info={"error": f"Invalid format: {e}"}
        )

    env = get_session()
    result = env.step(action)
    return result

@router.get("/state", response_model=Observation)
async def state():
    """Get current observation state."""
    env = get_session()
    return env._obs()

# ─────────────────────────────────────────────────────────────
# MOUNTING & DEPLOYMENT
# ─────────────────────────────────────────────────────────────

# Include router at both root and /env to satisfy all multi-mode expectations
app.include_router(router)
app.include_router(router, prefix="/env")

# Mount Gradio UI at root (takes over visual root, API stays alive behind it)
app = gr.mount_gradio_app(app, demo, path="/")

def main():
    # Use port 7860 as mandatory for Hugging Face Spaces
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
