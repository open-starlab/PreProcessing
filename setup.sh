#!/bin/bash
# Quick start script for Defense Line Analysis pipeline

set -e  # Exit on error

echo "=========================================="
echo "Defense Line Analysis - Quick Start"
echo "=========================================="

# Check if Python 3.8+ is installed
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

echo ""
echo "=========================================="
echo "Pipeline Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Organize your data:"
echo "     python preprocessing/data_reorganization.py --raw_dir /path/to/raw/data --output_dir ./data/Laliga2023/24"
echo ""
echo "  2. Run the pipeline:"
echo "     python main.py --help                    # See all options"
echo "     python main.py                           # Default configuration"
echo "     python main.py --data_match barcelona_madrid --back_four back_four"
echo ""
echo "  3. Or use Python API:"
echo "     python -c \"from preprocessing.config import PipelineConfig; from preprocessing.main import run_pipeline\""
echo ""
echo "Documentation:"
echo "  - README.md for overview"
echo "  - DEVELOPMENT.md for extending the pipeline"
echo "  - example_notebook.ipynb for examples"
echo ""
