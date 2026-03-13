import os
from huggingface_hub import snapshot_download
save_path = "./models/TinyLlama"
snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir=save_path
)
print("Download complete!")