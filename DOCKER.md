# Docker Setup Guide

Run the Smart Money Detection system in Docker without any local installation!

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Run the live test
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Docker CLI

```bash
# Build image
docker build -t smart-money-detector .

# Run live test
docker run -it --rm smart-money-detector

# Run with .env file
docker run -it --rm -v $(pwd)/.env:/app/.env:ro smart-money-detector

# Run interactive shell
docker run -it --rm smart-money-detector /bin/bash
```

## Running Different Commands

### Live Test
```bash
docker-compose run --rm smart-money-detector python test_live.py
```

### Basic Example
```bash
docker-compose run --rm smart-money-detector python examples/basic_usage.py
```

### Kalshi Integration
```bash
docker-compose run --rm smart-money-detector python examples/kalshi_integration.py
```

### Interactive Python Shell
```bash
docker-compose run --rm smart-money-detector python
```

Then in Python:
```python
from smart_money_detection import SmartMoneyDetector
from smart_money_detection.kalshi_client import KalshiClient

# Your code here
```

## Environment Variables

Create a `.env` file:

```bash
# Kalshi API
KALSHI_API_KEY=your_key_here
# Optional: override the default Kalshi API base URL
KALSHI_API_BASE=https://api.elections.kalshi.com
KALSHI_DEMO_MODE=false  # Set to true for demo mode

# Detection settings
DETECTION_THRESHOLD=0.75
MIN_VOLUME_THRESHOLD=100

# Logging
LOG_LEVEL=INFO
```

If `KALSHI_API_BASE` is set in your environment or `.env` file, the Kalshi client now uses it automatically—no need to pass `api_base` manually when instantiating `KalshiClient`.

## Persisting Data

Data is automatically saved to `./data/` which is mounted as a volume.

```bash
# Detector state files
./data/detector_state.pkl

# Feedback data
./data/feedback_history.json

# Results
./data/smart_money_trades_*.csv
```

## Development

### Mount source code for live editing:

```bash
docker run -it --rm \
  -v $(pwd):/app \
  smart-money-detector \
  /bin/bash
```

### Rebuild after changes:

```bash
docker-compose build
```

## Troubleshooting

### Permission errors:
```bash
# Fix data directory permissions
sudo chown -R $USER:$USER ./data/
```

### Container won't start:
```bash
# Check logs
docker-compose logs

# Remove old containers
docker-compose down -v
docker-compose up --build
```

### API connection issues:
```bash
# Test in demo mode first
export KALSHI_DEMO_MODE=true
docker-compose up
```

## Production Deployment

### Using Docker Swarm:

```bash
docker stack deploy -c docker-compose.yml smart-money
```

### Using Kubernetes:

```bash
kubectl apply -f kubernetes/
```

(Create kubernetes manifests as needed)

## Resource Limits

Add to docker-compose.yml:

```yaml
services:
  smart-money-detector:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## Monitoring

### View resource usage:
```bash
docker stats kalshi-smart-money
```

### Export logs:
```bash
docker-compose logs > logs.txt
```

## Next Steps

1. Run live test: `docker-compose up`
2. Review results in `./data/`
3. Set up real API key in `.env`
4. Integrate into your pipeline
5. Scale with Kubernetes/Swarm as needed
