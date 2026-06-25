"""Configuration for the CLIP CoOp prompt-learning experiments."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = Path(
    os.environ.get(
        "COOP_DATA_ROOT",
        str(PROJECT_ROOT / "data"),
    )
).expanduser().resolve()

DATASET_ROOTS = {
    "oxford_pets": DATA_ROOT / "oxford_pets",
    "dtd": DATA_ROOT / "dtd",
    "eurosat": DATA_ROOT / "eurosat",
    "oxford_flowers": DATA_ROOT / "oxford_flowers",
    "food101": DATA_ROOT / "food-101",
    "caltech101": DATA_ROOT / "caltech-101",
    "ucf101": DATA_ROOT / "ucf101",
    "fgvc_aircraft": DATA_ROOT / "fgvc-aircraft-2013b",
}

IMAGE_DIRS = {
    "oxford_pets": DATASET_ROOTS["oxford_pets"] / "images",
    "dtd": DATASET_ROOTS["dtd"] / "images",
    "eurosat": DATASET_ROOTS["eurosat"] / "2750",
    "oxford_flowers": DATASET_ROOTS["oxford_flowers"] / "jpg",
    "food101": DATASET_ROOTS["food101"] / "images",
    "caltech101": DATASET_ROOTS["caltech101"] / "101_ObjectCategories",
    "ucf101": DATASET_ROOTS["ucf101"] / "UCF-101-midframes",
    "fgvc_aircraft": DATASET_ROOTS["fgvc_aircraft"] / "data",
}

SPLIT_FILES = {
    "oxford_pets": DATASET_ROOTS["oxford_pets"] / "split_zhou_OxfordPets.json",
    "dtd": DATASET_ROOTS["dtd"] / "split_zhou_DescribableTextures.json",
    "eurosat": DATASET_ROOTS["eurosat"] / "split_zhou_EuroSAT.json",
    "oxford_flowers": DATASET_ROOTS["oxford_flowers"] / "split_zhou_OxfordFlowers.json",
    "food101": DATASET_ROOTS["food101"] / "split_zhou_Food101.json",
    "caltech101": DATASET_ROOTS["caltech101"] / "split_zhou_Caltech101.json",
    "ucf101": DATASET_ROOTS["ucf101"] / "split_zhou_UCF101.json",
    "fgvc_aircraft": DATASET_ROOTS["fgvc_aircraft"] / "split_zhou_FGVCAircraft.json",
}

OUTPUT_DIR = PROJECT_ROOT / "results"

CLIP_MODEL = "ViT-B/16"
SEED = 42
BATCH_SIZE = 64
