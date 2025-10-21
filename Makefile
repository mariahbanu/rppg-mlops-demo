.PHONY: help setup install data start stop clean test train deploy logs restart

# Default target
help:
	@echo "🚀 MLOps Portfolio Project - Available Commands:"
	@echo ""
	@echo "  make setup          - One-time setup (install dependencies)"
	@echo "  make install        - Install Python packages"
	@echo "  make data           - Download and prepare data"
	@echo "  make start          - Start all services (MLflow, API, monitoring)"
	@echo "  make stop           - Stop all services"
	@echo "  make train          - Train the model"
	@echo "  make test           - Run all tests"
	@echo "  make clean          - Clean up everything"
	@echo "  make deploy         - Deploy to Hugging Face"
	@echo ""

# One-time setup
setup: install data
	@echo "✅ Setup complete! Run 'make start' to begin."

# Install dependencies
install:
	@echo "📦 Creating virtual environment..."
	python3 -m venv venv || true
	@echo "📦 Installing Python dependencies..."
	. venv/bin/activate && python3 -m pip install --upgrade pip
	. venv/bin/activate && python3 -m pip install -r requirements.txt
	@echo "✅ Dependencies installed"
	@echo "💡 Activate with: source venv/bin/activate"

# Download/create data
data:
	@echo "📥 Preparing dataset..."
	. venv/bin/activate && python3 scripts/prepare_data.py
	@echo "✅ Data ready"

# Start all services
start:
	@echo "🚀 Starting MLOps stack..."
	docker compose up -d
	@echo ""
	@echo "✅ Services started! Access:"
	@echo "   📊 MLflow:     http://localhost:5000"
	@echo "   🤖 API:        http://localhost:8080"
	@echo "   📈 Prometheus: http://localhost:9090"
	@echo "   📉 Grafana:    http://localhost:3001 (admin/admin)"
	@echo ""
	@echo "📝 Logs: docker compose logs -f"

# Stop all services
stop:
	@echo "🛑 Stopping services..."
	docker compose down
	@echo "✅ Services stopped"

# Train model
train:
	@echo "🎓 Training model..."
	. venv/bin/activate && python3 src/training/train.py
	@echo "✅ Training complete! Check MLflow at http://localhost:5000"

# Run tests
test:
	@echo "🧪 Running tests..."
	. venv/bin/activate && pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Tests complete! Coverage report: htmlcov/index.html"

# Clean everything
clean:
	@echo "🧹 Cleaning up..."
	docker compose down -v
	rm -rf mlflow/mlruns mlflow/mlflow.db
	rm -rf htmlcov .coverage .pytest_cache
	rm -rf venv
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete"

# Deploy to Hugging Face
deploy:
	@echo "🚀 Deploying to Hugging Face Spaces..."
	. venv/bin/activate && python3 scripts/deploy_to_hf.py
	@echo "✅ Deployment complete!"

# Show logs
logs:
	docker compose logs -f

# Restart services
restart: stop start
