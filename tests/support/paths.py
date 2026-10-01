"""Locations of the test data, independent of where a test file lives."""

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TESTS_DIR / "data"
