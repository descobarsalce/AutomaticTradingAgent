# Project Dependencies

## Python Version
- Python 3.11

## Core Machine Learning & Reinforcement Learning
- gymnasium==0.29.1
- stable-baselines3==2.3.2
- torch==2.5.1
- optuna==4.2.0
- shimmy==0.2.1

## Data Processing & Analysis
- numpy==2.1.3
- pandas==2.2.3
- scipy==1.15.1
- scikit-learn==1.6.1
- ta==0.11.0
- yfinance==0.2.50

## Visualization & Web Interface
- streamlit==1.40.2
- plotly==5.24.1
- ipywidgets==8.1.5

## Database
- sqlalchemy==2.0.36

## Development & Testing
- pytest==8.3.4
- pytest-cov==6.0.0
- tensorboard==2.18.0
- tqdm (latest)
- jupyter==1.1.1
- notebook==7.3.2

## Installation

This project uses the Replit package manager for dependency management. To install dependencies:

1. Install Python 3.11
2. Install the required packages using the package manager:
   ```bash
   # Core ML packages
   pip install gymnasium==0.29.1 stable-baselines3==2.3.2 torch==2.5.1 optuna==4.2.0 shimmy==0.2.1

   # Data processing
   pip install numpy==2.1.3 pandas==2.2.3 scipy==1.15.1 scikit-learn==1.6.1 ta==0.11.0 yfinance==0.2.50

   # Visualization
   pip install streamlit==1.40.2 plotly==5.24.1 ipywidgets==8.1.5

   # Database
   pip install sqlalchemy==2.0.36

   # Development tools
   pip install pytest==8.3.4 pytest-cov==6.0.0 tensorboard==2.18.0 tqdm jupyter==1.1.1 notebook==7.3.2
   ```

## System Requirements

### Required System Dependencies
- CUDA-compatible GPU recommended for PyTorch acceleration
- PostgreSQL database server
- Git for version control

### Environment Variables
The following environment variables need to be set:
- `DATABASE_URL`: PostgreSQL database connection string
- Other configuration variables will be prompted during setup

## Notes
- Version numbers are specified for compatibility
- Some packages may have additional system-level dependencies
- The project has been tested with these specific versions
- Make sure to have sufficient disk space for model training (recommend at least 10GB)
- Minimum 8GB RAM recommended for training
- For CPU-only installation, PyTorch will automatically install the CPU version

## Development Setup
1. Clone the repository
2. Install dependencies as listed above
3. Set up PostgreSQL database
4. Set required environment variables
5. Run the Streamlit application:
   ```bash
   streamlit run main.py --server.port=5000 --server.address=0.0.0.0
   ```

## Troubleshooting
- If you encounter CUDA/GPU issues, ensure you have compatible NVIDIA drivers installed
- For database connection issues, verify PostgreSQL is running and the connection string is correct
- Memory issues during training may require reducing batch sizes in the configuration