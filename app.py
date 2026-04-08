import gradio as gr
import requests
import json
import os

# Updated API_URL to handle both local and Space deployments
API_URL = os.getenv("API_BASE_URL", "http://localhost:7860/env")

def check_status():
    try:
        # Internal ping to the local FASTAPI engine
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            return f"🟢 Connected to OpenEnv Engine ({API_URL})"
        return f"🔴 Engine returned {response.status_code}"
    except Exception:
        return f"🔴 Engine Offline! (Expected at {API_URL})"

def start_task(task_name):
    try:
        response = requests.post(f"{API_URL}/reset", json={"task": task_name}, timeout=10)
        if response.status_code == 200:
            return f"✅ Environment Reset: {task_name}\n\n" + json.dumps(response.json(), indent=2)
        return "Reset Failed: " + response.text
    except Exception as e:
        return f"Error: {str(e)}"

with gr.Blocks(title="Data Pipeline Debug Env", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 AI Data Pipeline Debugging Environment")
    gr.Markdown(
        "This is a **Multi-Mode OpenEnv Space**. The Gradio UI is for human monitoring, "
        "while the FastAPI backend manages the RL environment for AI agents."
    )
    
    status_text = gr.Markdown(check_status())
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Control Panel")
            task_dropdown = gr.Dropdown(
                choices=["easy", "medium", "hard", "messy_schema"], 
                value="easy", 
                label="Select Debug Scenario"
            )
            reset_btn = gr.Button("Reset RL Environment", variant="primary")
        
        with gr.Column(scale=2):
            gr.Markdown("### 2. Live Agent State (JSON)")
            output_json = gr.Code(language="json")

    reset_btn.click(fn=start_task, inputs=[task_dropdown], outputs=[output_json])
    
    # Auto-refresh status on load
    demo.load(fn=check_status, outputs=[status_text])

if __name__ == "__main__":
    # Standard launch if run directly
    demo.launch(server_name="0.0.0.0", server_port=7860)
