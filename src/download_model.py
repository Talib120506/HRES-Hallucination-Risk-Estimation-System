import os
from huggingface_hub import snapshot_download, login

# Calculate absolute path to models/gemma-2-2b-it from src/ directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(BASE_DIR, "models", "gemma-2-2b-it")

hf_token = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
login(token=hf_token)

print(f"Downloading Gemma to: {save_path}")
snapshot_download(
    repo_id="google/gemma-2-2b-it",
    local_dir=save_path
)
print(f"Download complete! Model saved to: {save_path}")
