import gradio as gr
import requests
import json
import os

API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

def check_status():
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            return f"🟢 API Backend is Online! ({API_URL})"
        return f"🔴 API Backend returned {response.status_code}"
    except Exception as e:
        return f"🔴 API Backend Offline! Error: {str(e)}"

def start_task(task_name):
    try:
        response = requests.post(f"{API_URL}/reset", json={"task": task_name})
        if response.status_code == 200:
            return f"✅ Started task: {task_name}\n\nObservation Flow:\n" + json.dumps(response.json(), indent=2)
        return "Failed: " + response.text
    except Exception as e:
        return str(e)

with gr.Blocks(title="Data Pipeline Debug Env") as demo:
    gr.Markdown("# 🚀 AI Data Pipeline Debugging Environment")
    gr.Markdown(
        "Welcome to the OpenEnv Hackathon Submission! This dashboard allows you to visually trigger tests "
        "and monitor the backend API agent solving Data Pipeline failures."
    )
    
    status_text = gr.Markdown(check_status())
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Select a Task to Reset Environment")
            task_dropdown = gr.Dropdown(choices=["easy", "medium", "hard", "messy_schema"], value="easy", label="Task")
            reset_btn = gr.Button("Reset Environment", variant="primary")
        
        with gr.Column():
            gr.Markdown("### 2. View Backend JSON State")
            output_json = gr.Code(language="json")

    reset_btn.click(fn=start_task, inputs=[task_dropdown], outputs=[output_json])
    
    demo.load(fn=check_status, outputs=[status_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
