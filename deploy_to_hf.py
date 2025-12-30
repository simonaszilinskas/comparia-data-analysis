#!/usr/bin/env python3
"""Deploy the Streamlit app to HuggingFace Spaces."""

from huggingface_hub import HfApi
import os

# Get token from environment or prompt
token = os.environ.get("HF_TOKEN")
if not token:
    token = input("Enter your HuggingFace token: ").strip()

# Initialize API
api = HfApi()

# Space details
repo_id = "comparIA/French-arena-dataset-preview"
repo_type = "space"

print(f"Uploading files to {repo_id}...")

# Upload all necessary files
files_to_upload = [
    "app.py",
    "requirements.txt",
    "README.md",
    "english-logo.png",
    "pages/Search the compar:IA datasets.py",
    "pages/Visualise the compar:IA datasets.py",
]

for file_path in files_to_upload:
    print(f"  Uploading {file_path}...")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token,
    )

print("\n✅ Successfully deployed to HuggingFace Spaces!")
print(f"🔗 View your Space at: https://huggingface.co/spaces/{repo_id}")
