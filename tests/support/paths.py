"""Locations of the test data, wherever a test file lives."""

from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = TESTS_DIR / "data"
# Real SEM images and the particle files the readers are tested on
IMAGES_DIR = DATA_DIR / "images"
PARTICLES_DIR = DATA_DIR / "particles"
EXAMPLES_DIR = TESTS_DIR.parent / "examples"
DOCS_DIR = TESTS_DIR.parent / "docs"
