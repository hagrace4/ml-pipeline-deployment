# ML Pipeline Deployment System

End-to-end machine learning pipeline deployment system for homelab Kubernetes infrastructure. Demonstrates production-grade ML operations using local NVIDIA GPUs (GTX 4070, RTX 3060) with free and low-cost tools.

## Overview

This project implements a complete MLOps pipeline with:
- **Data Ingestion**: Automated data fetching and validation
- **Model Training**: XGBoost and PyTorch with GPU acceleration
- **Model Registry**: MLflow for versioning and lineage tracking
- **Inference Serving**: FastAPI-based REST API endpoints
- **Monitoring**: Prometheus metrics and Grafana dashboards
- **CI/CD**: GitHub Actions with self-hosted runners

## Architecture

```
GitHub Actions → Data Ingestion → Training (GPU) → MLflow Registry → Inference API
                                                                    ↓
                                                              Prometheus/Grafana
```

The system runs on homelab Kubernetes (K3s) with:
- Dell R430 (control plane + worker)
- Minisforum (worker node)
- NVIDIA GTX 4070 and RTX 3060 GPUs

## Project Status

🚧 **In Development** - Core infrastructure complete, implementing services

### Completed
- ✅ Project structure and shared utilities
- ✅ Configuration management (env vars + YAML/JSON files)
- ✅ Pydantic data models for type safety
- ✅ Prometheus metrics base classes
- ✅ Structured logging utilities
- ✅ Property-based tests for configuration (Hypothesis)
- ✅ Unit tests for core functionality

### In Progress
- 🔄 Data ingestion service
- 🔄 Training service (XGBoost/PyTorch)
- 🔄 MLflow integration
- 🔄 Inference endpoint

## Quick Start

### Prerequisites

- Python 3.10+
- Kubernetes cluster (K3s recommended for homelab)
- NVIDIA GPU with CUDA support (optional, falls back to CPU)
- Docker with NVIDIA Container Toolkit

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ml-pipeline-deployment
```

2. Create virtual environment and install dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

3. Configure environment:
```bash
cp config/default.yaml config/local.yaml
# Edit config/local.yaml with your settings
```

### Running Tests

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit -v

# Run property-based tests with statistics
pytest tests/properties -v --hypothesis-show-statistics

# Run with coverage
pytest --cov=shared --cov=services
```

## Configuration

Configuration is loaded from:
1. Environment variables (highest priority)
2. Configuration files (YAML/JSON)
3. Default values (lowest priority)

### Environment Variables

```bash
# General settings
ML_PIPELINE_ENVIRONMENT=development  # development, staging, production
ML_PIPELINE_LOG_LEVEL=INFO
ML_PIPELINE_GPU_ENABLED=true

# Data source
DATA_SOURCE_SOURCE_TYPE=local  # local, url, s3
DATA_SOURCE_SOURCE_PATH=/data/training
DATA_SOURCE_FORMAT=csv

# Model training
MODEL_MODEL_TYPE=xgboost  # xgboost, pytorch
MODEL_EPOCHS=100
MODEL_BATCH_SIZE=32
MODEL_LEARNING_RATE=0.001

# MLflow registry
REGISTRY_MLFLOW_TRACKING_URI=http://localhost:5000
REGISTRY_MLFLOW_EXPERIMENT_NAME=ml-pipeline

# Inference
INFERENCE_MODEL_NAME=ml-model
INFERENCE_PORT=8000
```

### Configuration File

See `config/default.yaml` for full configuration options.

## Project Structure

```
ml-pipeline-deployment/
├── config/
│   └── default.yaml                   # Default configuration
├── services/
│   ├── data_ingestion/                # Data fetching and validation
│   ├── training/                      # Model training (XGBoost/PyTorch)
│   └── inference/                     # FastAPI prediction endpoint
├── shared/
│   ├── config.py                      # Configuration management
│   ├── models.py                      # Pydantic data models
│   ├── metrics.py                     # Prometheus metrics
│   └── logging.py                     # Structured logging
├── tests/
│   ├── unit/                          # Unit tests
│   └── properties/                    # Property-based tests (Hypothesis)
├── k8s/                               # Kubernetes manifests (coming soon)
├── .github/workflows/                 # CI/CD pipelines (coming soon)
└── pyproject.toml                     # Python project configuration
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| ML Frameworks | PyTorch 2.x, XGBoost 2.x |
| API Framework | FastAPI |
| Data Validation | Pydantic |
| Testing | pytest, Hypothesis (property-based) |
| Metrics | Prometheus Client |
| Model Registry | MLflow 2.x |
| Container Runtime | Docker + NVIDIA Container Toolkit |
| Orchestration | Kubernetes (K3s) |
| CI/CD | GitHub Actions |

## Development

### Testing Strategy

**Dual Testing Approach:**
- **Unit Tests**: Specific examples and edge cases
- **Property-Based Tests**: Universal properties across all inputs (100+ iterations)

Property tests validate correctness properties like:
- Configuration loading from any valid source
- Configuration application to all services
- Invalid configuration handling with clear errors

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy shared services
```

## Kubernetes Deployment

### Prerequisites

1. K3s cluster with GPU support
2. NVIDIA Device Plugin installed
3. Local container registry or Docker Hub access

### Deploy Services

```bash
# Deploy MLflow
kubectl apply -f k8s/mlflow/

# Deploy data ingestion
kubectl apply -f k8s/data-ingestion/

# Deploy training job
kubectl apply -f k8s/training/

# Deploy inference endpoint
kubectl apply -f k8s/inference/

# Deploy monitoring
kubectl apply -f k8s/monitoring/
```

See the `docs/` directory for detailed infrastructure setup instructions.

## Monitoring

Access monitoring dashboards:

- **MLflow UI**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

Metrics exposed:
- Data ingestion: rows processed, validation errors, duration
- Training: GPU utilization, loss, accuracy, F1 score
- Inference: request latency, throughput, error rates

## Contributing

This is a learning project demonstrating ML pipeline patterns. Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built as a learning platform for:
- Production ML operations patterns
- Kubernetes-native ML workloads
- Property-based testing for correctness
- Homelab infrastructure optimization

## Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [K3s Documentation](https://docs.k3s.io/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
