import os
from huggingface_hub import snapshot_download

# Calculate absolute path to models/TinyLlama from src/ directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_path = os.path.join(BASE_DIR, "models", "TinyLlama")

print(f"Downloading TinyLlama to: {save_path}")
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir=save_path
)
print(f"Download complete! Model saved to: {save_path}")
