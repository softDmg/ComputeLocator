# ComputeLocator

A Kubernetes-based platform that collects runtime metrics from heterogeneous containerized workloads for execution performance prediction experiments, supporting dynamic compute placement decisions.

## Research context

This project extends the work of Kimovski et al. in *"Cloud, Fog, or Edge: Where to Compute?"* (IEEE Internet Computing, 2021) by implementing a dynamic, data-driven approach to compute placement. Where the paper provides static placement recommendations based on application and infrastructure characteristics, this system deploys real workloads across resource-constrained environments, collects execution data, and trains predictive models from observed performance.

## Architecture

![Architecture](project_resources/img.png)

## How it works

1. **Deploy** benchmark workloads on Kubernetes pods with different resource constraints (small: 100m CPU / 1Gi RAM, medium: 250m CPU / 2Gi RAM, large: 500m CPU / 4Gi RAM)
2. **Execute** diverse compute functions and collect 14+ process metrics at 1-second resolution via Prometheus, plus structured request logs with nanosecond timestamps via Loki
3. **Aggregate** and temporally align metrics with request windows through a custom aggregation API
4. **Train** an XGBoost multi-output regression model to predict execution performance from pod size, function type, and initial metric state

## Benchmark workloads

| Workload | Endpoint | Description | Key Tech |
|----------|----------|-------------|----------|
| Dense Network | `/dense` | MNIST classifier with configurable layers and units, memory-efficient streaming batch loading | TensorFlow |
| Pi Estimation | `/pi` | Monte Carlo method, 100M vectorized samples in batches of 100K | NumPy |
| Video Encoding | `/video` | H.264 encoding at 720p / 1080p / 1440p profiles from a 4-second segment | FFmpeg |
| CPU Stress | `/ml` | Parameterized arithmetic loop for variable-load testing | Python |

## Collected metrics

All collected at 1-second resolution via Prometheus scraping `psutil`-based exporters:

| Category | Metrics |
|----------|---------|
| CPU | Usage %, user time, system time |
| Memory | RSS, VMS, usage % |
| I/O | Read bytes, write bytes |
| Threading | Thread count, voluntary context switches, involuntary context switches |
| Network | Open connections |
| Runtime | Process uptime |

Each API request also produces a structured JSON log (collected via Promtail → Loki) containing start/end timestamps in nanoseconds, endpoint, method, payload, and status.

## ML pipeline

1. **Data collection** — The aggregation API joins Prometheus metric timeseries with Loki request logs, producing per-request records with aligned metric snapshots
2. **Feature engineering** — Flatten nested metric dictionaries, compute deltas for duration and CPU time, drop constant columns, one-hot encode pod size and function type
3. **Training** — XGBoost `MultiOutputRegressor` (100 estimators, max depth 5, learning rate 0.1) with 80/20 train-test split
4. **Evaluation** — MAE, RMSE, R², and MAPE per output metric; feature importance ranking; actual-vs-predicted scatter plots

## Project structure

```
ComputeLocator/
├── kubernetes/
│   ├── target_apis/                  # Benchmark workload deployments
│   │   ├── api-small.yaml            # Pod: 100m CPU, 1Gi RAM
│   │   ├── api-medium.yaml           # Pod: 250m CPU, 2Gi RAM
│   │   ├── api-large.yaml            # Pod: 500m CPU, 4Gi RAM
│   │   └── api_code/
│   │       ├── server.py             # Flask API exposing /ml, /pi, /dense, /video
│   │       ├── functions/            # Benchmark implementations
│   │       │   ├── ML.py
│   │       │   ├── pi_estimation.py
│   │       │   ├── dense_network.py
│   │       │   └── video_encoding.py
│   │       ├── resources/            # MNIST dataset, source video
│   │       └── Dockerfile
│   ├── service_registry/             # Observability stack
│   │   ├── prometheus-server.yaml    # Prometheus, Loki, Promtail, aggregation API
│   │   └── custom_aggregation_api/
│   │       ├── server.py             # /history and /endpoints aggregation API
│   │       └── Dockerfile
│   └── load-balancer/                # Client gateway and ML training
│       ├── client-gateway.yaml       # Kubernetes Job manifest
│       └── code/
│           ├── main.py               # Orchestrates benchmark execution
│           ├── model_exploration.py   # XGBoost training and evaluation
│           ├── tested_api_client.py   # API client for benchmark functions
│           ├── cluster_data_client.py # Client for metrics aggregation
│           └── Dockerfile
└── project_resources/
    └── img.png                       # Architecture diagram
```

## Current Tech stack

Python, Flask, pandas, Kubernetes, Docker, Prometheus, Loki, Promtail

## Getting started

### Prerequisites

- [minikube](https://minikube.sigs.k8s.io/)
- Docker
- kubectl

### Build images

```bash
docker build -t sample_api ./kubernetes/target_apis/api_code/
minikube image load sample_api

docker build -t cluster_api ./kubernetes/service_registry/custom_aggregation_api/
minikube image load cluster_api

docker build -t client-gateway ./kubernetes/load-balancer/code/
minikube image load client-gateway
```

### Deploy

```bash
kubectl apply -f ./kubernetes/target_apis/
kubectl apply -f ./kubernetes/service_registry/
kubectl apply -f ./kubernetes/load-balancer/
```

### Access

Forward the aggregation API to your host:

```bash
kubectl port-forward deployment/prometheus-server 5000:5000
```

Retrieve collected data:

```bash
curl http://localhost:5000/history
```

Send a request to a benchmark pod:

```bash
kubectl port-forward deployment/python-api-large 8000:8000
curl -X POST http://localhost:8000/ml -H 'Content-Type: application/json' -d '{"input": 15}'
curl -X POST http://localhost:8000/pi -H 'Content-Type: application/json' -d '{"num_samples": 100000000}'
curl -X POST http://localhost:8000/dense -H 'Content-Type: application/json' -d '{"epochs": 5, "num_layers": 2}'
curl -X POST http://localhost:8000/video -H 'Content-Type: application/json' -d '{"profile": 1}'
```
