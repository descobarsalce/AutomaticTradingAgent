# Algorithmic Trading Research Platform

An advanced algorithmic trading research platform that leverages reinforcement learning for sophisticated portfolio management and multi-stock analysis. The system implements cutting-edge machine learning techniques with comprehensive logging and monitoring capabilities.

## Key Features

- Reinforcement Learning Framework (Gymnasium, Stable Baselines 3)
- Advanced Trading Algorithms (DDPG, PPO)
- Comprehensive Documentation and Type Hinting
- Web Visualization (Streamlit, Plotly)
- Market Data Management (SQL Database)
- Modular Machine Learning Agent Design
- Hyperparameter Optimization
- Robust Test Suite

## System Requirements & Installation

Please refer to [dependencies.md](dependencies.md) for a complete list of:
- Required packages and their versions
- System dependencies
- Hardware requirements
- Environment variables
- Installation instructions
- Troubleshooting guide

## Quick Start

1. Install all required dependencies as specified in [dependencies.md](dependencies.md)
2. Set up environment variables as documented
3. Run the application:
   ```bash
   streamlit run main.py --server.port=5000 --server.address=0.0.0.0
   ```
4. Access the web interface at `http://localhost:5000`

## Project Structure

- `/core` - Core trading agent implementation
- `/environment` - Trading environment simulation
- `/models` - Database models and data structures
- `/utils` - Utility functions and helpers
- `/data` - Data handling and preprocessing
- `/tests` - Test suite

## Development

1. Install development dependencies as specified in [dependencies.md](dependencies.md)
2. Run tests:
   ```bash
   python -m pytest tests/
   ```
3. Start Jupyter notebook for development:
   ```bash
   jupyter notebook
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.