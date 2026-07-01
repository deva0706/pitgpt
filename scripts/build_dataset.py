"""Thin CLI: build the raw dataset, then validate + clean it.

Usage:  python scripts/build_dataset.py
"""

from pitgpt.data.build_dataset import main as build
from pitgpt.data.validate_dataset import main as validate

if __name__ == "__main__":
    build()
    print()
    validate()
