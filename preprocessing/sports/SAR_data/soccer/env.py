import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parents[1] / "data"))
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).parents[1]))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", Path(__file__).parents[1] / "output"))
