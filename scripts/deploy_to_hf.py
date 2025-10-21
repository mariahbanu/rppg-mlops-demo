"""
Deploy to Hugging Face Spaces
"""
from huggingface_hub import HfApi, create_repo
import os
import shutil
from pathlib import Path

def deploy():
    """Deploy Streamlit app to Hugging Face Spaces"""
    
    # Configuration
    username = input("Enter your Hugging Face username: ")
    space_name = "rppg-heart-rate-estimator"
    repo_id = f"{username}/{space_name}"
    
    print(f"🚀 Deploying to: {repo_id}")
    
    # Create temporary deployment directory
    deploy_dir = Path("deploy_temp")
    deploy_dir.mkdir(exist_ok=True)
    
    try:
        # Copy necessary files
        print("📦 Preparing files...")
        
        # Copy main app
        shutil.copy("demo_app.py", deploy_dir / "app.py")
        
        # Copy model files
        shutil.copytree("src", deploy_dir / "src", dirs_exist_ok=True)
        shutil.copytree("data/models", deploy_dir / "data/models", dirs_exist_ok=True)
        
        # Create requirements.txt for HF Spaces
        hf_requirements = """
torch==2.0.1
streamlit==1.25.0
numpy==1.24.3
pandas==2.0.2
plotly==5.15.0
        """.strip()
        
        with open(deploy_dir / "requirements.txt", "w") as f:
            f.write(hf_requirements)

        # Create Dockerfile for HF Spaces
        dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
"""

        with open(deploy_dir / "Dockerfile", "w") as f:
            f.write(dockerfile_content)

        # Create README with metadata (using Docker SDK for Streamlit)
        readme_content = f"""---
title: rPPG Heart Rate Estimator
emoji: ❤️
colorFrom: red
colorTo: pink
sdk: docker
app_port: 8501
pinned: false
---

# ❤️ rPPG Heart Rate Estimator

Video-based heart rate detection using remote photoplethysmography (rPPG).

## About

This is an end-to-end MLOps demonstration project showcasing:
- Deep learning with PyTorch
- Experiment tracking with MLflow
- Containerization with Docker
- Monitoring with Prometheus & Grafana
- CI/CD with GitHub Actions

**Built for Presage Technologies Application**

## Try It

Upload a video or generate a synthetic signal to estimate heart rate!

## Links

- [GitHub Repository](https://github.com/yourusername/rppg-mlops-demo)
- [Project Documentation](#)

---

⚠️ **Disclaimer:** For demonstration purposes only. Not for medical use.
        """
        
        with open(deploy_dir / "README.md", "w") as f:
            f.write(readme_content)
        
        # Create/upload to HF Spaces
        print(f"🌐 Creating space: {repo_id}")
        
        api = HfApi()
        
        try:
            create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk="docker",
                private=False,
                exist_ok=True
            )
            print(f"✅ Created/verified space: {repo_id}")
        except Exception as e:
            if "already exists" in str(e).lower() or "409" in str(e):
                print(f"✅ Space {repo_id} already exists, updating...")
            else:
                raise
        
        # Upload all files
        print("📤 Uploading files...")
        api.upload_folder(
            folder_path=str(deploy_dir),
            repo_id=repo_id,
            repo_type="space"
        )
        
        print("\n" + "=" * 60)
        print("🎉 Deployment successful!")
        print("=" * 60)
        print(f"🔗 Your app: https://huggingface.co/spaces/{repo_id}")
        print("⏳ Building... (takes 2-3 minutes)")
        print("=" * 60)
        
    finally:
        # Cleanup
        if deploy_dir.exists():
            shutil.rmtree(deploy_dir)

if __name__ == "__main__":
    deploy()
