#!/bin/bash
set -e

# Install dependencies
pip install mlflow psutil gunicorn

# Set MLflow configuration
export BACKEND_STORE_URI="sqlite:///mlflow/mlflow.db"
export DEFAULT_ARTIFACT_ROOT="/mlflow/artifacts"

# Start MLflow with gunicorn on all interfaces
python -c "
from mlflow.server import app as mlflow_app
import os

# Configure MLflow
os.environ['BACKEND_STORE_URI'] = '${BACKEND_STORE_URI}'
os.environ['DEFAULT_ARTIFACT_ROOT'] = '${DEFAULT_ARTIFACT_ROOT}'
"

# Start gunicorn
exec gunicorn \
    -b 0.0.0.0:5000 \
    -w 4 \
    --access-logfile - \
    --error-logfile - \
    --timeout 60 \
    mlflow.server:app
