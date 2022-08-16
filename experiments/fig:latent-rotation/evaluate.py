import logging
from collections import defaultdict
from enum import auto
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import rich
import torch
import typer

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

logging.getLogger().setLevel(logging.ERROR)


BATCH_SIZE = 32


EXPERIMENT_ROOT = Path(__file__).parent
EXPERIMENT_CHECKPOINTS = EXPERIMENT_ROOT / "checkpoints"
PREDICTIONS_TSV = EXPERIMENT_ROOT / "predictions.tsv"
PERFORMANCE_TSV = EXPERIMENT_ROOT / "performance.tsv"

DATASET_SANITY = {
    "mnist": ("rae.data.vision.mnist.MNISTDataset", "test"),
    "fmnist": ("rae.data.vision.fmnist.FashionMNISTDataset", "test"),
    "cifar10": ("rae.data.vision.cifar10.CIFAR10Dataset", "test"),
    "cifar100": ("rae.data.vision.cifar100.CIFAR100Dataset", "test"),
}
MODEL_SANITY = {
    "abs": "rae.modules.vision.resnet.ResNet",
    "rel": "rae.modules.vision.relresnet.RelResNet",
}


def parse_checkpoint_id(ckpt: Path) -> str:
    return ckpt.with_suffix("").with_suffix("").name


# Parse checkpoints tree
checkpoints = defaultdict(dict)
RUNS = defaultdict(dict)
for dataset_abbrv in EXPERIMENT_CHECKPOINTS.iterdir():
    checkpoints[dataset_abbrv.name] = defaultdict(list)
    RUNS[dataset_abbrv.name] = defaultdict(list)
    for model_abbrv in dataset_abbrv.iterdir():
        for ckpt in model_abbrv.iterdir():
            checkpoints[dataset_abbrv.name][model_abbrv.name].append(ckpt)
            RUNS[dataset_abbrv.name][model_abbrv.name].append(parse_checkpoint_id(ckpt))


MODELS = checkpoints["mnist"]["ae"]
print(MODELS)
