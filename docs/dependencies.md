# Project Dependencies

Use this list to provision every Python package imported in the codebase (mirrors `pyproject.toml`). Pin the versions to avoid drift, especially for RL and brokerage connectors.

## Python Version
- Python 3.11+

## Core Machine Learning & Reinforcement Learning
- gym >= 0.26.2
- gymnasium == 0.29.1
- stable-baselines3 == 2.3.2
- torch == 2.5.1
- optuna == 4.2.0
- shimmy == 0.2.1

## Data Processing & Analysis
- numpy == 2.1.3
- pandas == 2.2.3
- scipy == 1.15.1
- scikit-learn == 1.6.1
- ta == 0.11.0
- yfinance == 0.2.50

## Visualization & Web Interface
- streamlit == 1.40.2
- plotly == 5.24.1
- ipywidgets == 8.1.5

## Databases & Persistence
- sqlalchemy == 2.0.36
- psycopg2-binary == 2.9.10

## Broker & Market Data Connectivity
- ib-insync == 0.9.86
- alpha-vantage == 3.0.0
- aiohttp == 3.11.12
- PyYAML == 6.0.2

## Agent Integrations & Retrieval
- openai == 1.60.2
- trafilatura == 2.0.0

## Development & Testing
- pytest == 8.3.4
- pytest-cov == 6.0.0
- tensorboard == 2.18.0
- tqdm (latest)
- jupyter == 1.1.1
- notebook == 7.3.2

## Installation

Install Python 3.11 or newer, then install the full set in one step (CPU-only PyTorch wheel shown):

```bash
# Optional: configure the PyTorch CPU index from pyproject.toml if needed
pip install \
  gym>=0.26.2 gymnasium==0.29.1 stable-baselines3==2.3.2 torch==2.5.1 \
  optuna==4.2.0 shimmy==0.2.1 numpy==2.1.3 pandas==2.2.3 scipy==1.15.1 \
  scikit-learn==1.6.1 ta==0.11.0 yfinance==0.2.50 streamlit==1.40.2 \
  plotly==5.24.1 ipywidgets==8.1.5 sqlalchemy==2.0.36 psycopg2-binary==2.9.10 \
  ib-insync==0.9.86 alpha-vantage==3.0.0 aiohttp==3.11.12 PyYAML==6.0.2 \
  openai==1.60.2 trafilatura==2.0.0 pytest==8.3.4 pytest-cov==6.0.0 \
  tensorboard==2.18.0 tqdm jupyter==1.1.1 notebook==7.3.2
```

## System Requirements
- Git for version control
- Optional PostgreSQL server if you use the database storage
- CUDA-compatible GPU recommended for faster training; CPU is supported (pyproject uses the CPU index by default)
- ~10GB free disk space and 8GB+ RAM recommended for training workloads

## Environment Variables
- `DATABASE_URL` when using PostgreSQL storage
- API keys for broker/data providers (e.g., IBKR, Alpha Vantage, OpenAI) as required by your workflow

## Troubleshooting
- If PyTorch/GPU wheels fail, install the CPU wheel or confirm NVIDIA drivers/CUDA version
- For database connectivity, verify PostgreSQL is running and `DATABASE_URL` is correct
- Reduce batch sizes or sequence lengths if you hit memory limits during training