"""
Download Solar Panel Defects dataset from Roboflow.

Dataset: https://universe.roboflow.com/solarpanel-2me5p/solar-panel-defects-lnge0
Classes: Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered

Usage:
    pip install roboflow
    # Optional: set RF_API_KEY environment variable (defaults to project key)
    python data/download_dataset.py
"""

import os

# Read API key from environment variable; fall back to the project key if unset.
RF_API_KEY = os.environ.get("RF_API_KEY", "0Bj5V6Y2QAIK6qrhAf2L")
RF_WORKSPACE = "solarpanel-2me5p"
RF_PROJECT = "solar-panel-defects-lnge0"
RF_VERSION = 1
DOWNLOAD_FORMAT = "folder"   # folder keeps per-class sub-directories
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "rf_dataset")


def download():
    try:
        from roboflow import Roboflow
    except ImportError:
        raise SystemExit(
            "roboflow package not found. Install it with:\n"
            "  pip install roboflow"
        )

    rf = Roboflow(api_key=RF_API_KEY)
    project = rf.workspace(RF_WORKSPACE).project(RF_PROJECT)
    version = project.version(RF_VERSION)

    print(f"Downloading '{RF_PROJECT}' (version {RF_VERSION}) …")
    dataset = version.download(DOWNLOAD_FORMAT, location=DOWNLOAD_DIR)
    print(f"Dataset saved to: {dataset.location}")
    return dataset


if __name__ == "__main__":
    download()
