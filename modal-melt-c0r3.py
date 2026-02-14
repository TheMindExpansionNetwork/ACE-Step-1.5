"""
MELT-C0R3 - ACE-Step 1.5 Music Generation Platform
Modal Deployment (v2 - Using official ACE-Step launchers)

Models:
- DiT: acestep-v15-turbo-rl (RL-trained, highest quality)
- LM: acestep-5Hz-lm-4B (strongest composition)
"""
import modal
import os

app = modal.App("melt-c0r3")

# =============================================================================
# VOLUMES
# =============================================================================
model_cache = modal.Volume.from_name("melt-c0r3-models", create_if_missing=True)
lora_storage = modal.Volume.from_name("melt-c0r3-loras", create_if_missing=True)
output_storage = modal.Volume.from_name("melt-c0r3-outputs", create_if_missing=True)

MODEL_DIR = "/models"
LORA_DIR = "/loras"
OUTPUT_DIR = "/outputs"

# =============================================================================
# IMAGE - CUDA 12.8 + ACE-Step dependencies
# =============================================================================
c0r3_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", python_version="3.11")
    .apt_install("git", "ffmpeg", "libsndfile1", "build-essential")
    .run_commands("pip install uv")
    .workdir("/repo")
)

# =============================================================================
# MODEL DOWNLOADER
# =============================================================================
@app.function(
    image=c0r3_image,
    volumes={MODEL_DIR: model_cache},
    timeout=3600,
)
def download_all_models():
    """Download all required ACE-Step models."""
    import subprocess
    
    os.environ["HF_HOME"] = MODEL_DIR
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
    
    models = [
        "ACE-Step/Ace-Step1.5",       # Main (turbo)
        "ACE-Step/acestep-5Hz-lm-4B", # Strongest LM
    ]
    
    print("📥 Downloading ACE-Step models...")
    for model in models:
        print(f"   → {model}")
        subprocess.run(
            ["huggingface-cli", "download", model, "--local-dir", f"{MODEL_DIR}/{model.replace('/', '_')}"],
            env=os.environ,
        )
    
    # Also download the RL variant when available
    print("✅ All models downloaded to", MODEL_DIR)

# =============================================================================
# CORE API (REST API Server)
# =============================================================================
@c0r3_image
@app.cls(
    gpu="A100",
    volumes={
        MODEL_DIR: model_cache,
        OUTPUT_DIR: output_storage,
    },
    allow_concurrent_inputs=20,
    timeout=600,
)
class MELTCoreAPI:
    """REST API for music generation."""
    
    @modal.web_server(8001, startup_timeout=300)
    def serve(self):
        import os
        import subprocess
        
        os.environ["ACE_STEP_CHECKPOINT_DIR"] = MODEL_DIR
        os.environ["HF_HOME"] = MODEL_DIR
        os.environ["OUTPUT_DIR"] = OUTPUT_DIR
        
        # Start the API server
        subprocess.Popen([
            "uv", "run", "acestep-api",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--lm-model-path", "acestep-5Hz-lm-4B",
        ])
        
        import time
        time.sleep(5)  # Give server time to start

# =============================================================================
# EXPERIENCE PORTAL (Gradio UI)
# =============================================================================
@c0r3_image
@app.cls(
    gpu="A100",
    volumes={
        MODEL_DIR: model_cache,
        OUTPUT_DIR: output_storage,
    },
    allow_concurrent_inputs=10,
    timeout=600,
)
class MELTExperience:
    """User-facing Gradio interface."""
    
    @modal.web_server(7860, startup_timeout=300)
    def serve(self):
        import os
        import subprocess
        
        os.environ["ACE_STEP_CHECKPOINT_DIR"] = MODEL_DIR
        os.environ["HF_HOME"] = MODEL_DIR
        os.environ["OUTPUT_DIR"] = OUTPUT_DIR
        
        # Start Gradio UI
        subprocess.Popen([
            "uv", "run", "acestep",
            "--server-name", "0.0.0.0",
            "--server-port", "7860",
            "--config_path", "acestep-v15-turbo",
            "--lm_model_path", "acestep-5Hz-lm-4B",
        ])
        
        import time
        time.sleep(5)

# =============================================================================
# TRAINER (LoRA Fine-tuning)
# =============================================================================
@c0r3_image
@app.cls(
    gpu="A100:2",
    volumes={
        MODEL_DIR: model_cache,
        LORA_DIR: lora_storage,
    },
    timeout=3600,
)
class MELTTrainer:
    """LoRA fine-tuning service."""
    
    @modal.web_server(7861, startup_timeout=300)
    def serve(self):
        import os
        import subprocess
        
        os.environ["ACE_STEP_CHECKPOINT_DIR"] = MODEL_DIR
        os.environ["HF_HOME"] = MODEL_DIR
        os.environ["LORA_DIR"] = LORA_DIR
        
        # Start training UI
        # Note: ACE-Step has built-in training in Gradio
        subprocess.Popen([
            "uv", "run", "acestep",
            "--server-name", "0.0.0.0",
            "--server-port", "7861",
            "--train-mode",  # Enable training UI
        ])
        
        import time
        time.sleep(5)

# =============================================================================
# DEPLOY
# =============================================================================
"""
# Download models first
modal run modal-melt-c0r3.py::download_all_models

# Deploy API (use: modal deploy modal-melt-c0r3.py::MELTCoreAPI)
modal deploy modal-melt-c0r3.py::MELTCoreAPI

# Deploy Gradio UI  
modal deploy modal-melt-c0r3.py::MELTExperience

# Deploy Trainer
modal deploy modal-melt-c0r3.py::MELTTrainer
"""
