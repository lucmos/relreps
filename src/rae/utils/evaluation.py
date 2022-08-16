from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Type, Union

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from nn_core.serialization import NNCheckpointIO


def parse_checkpoint_id(ckpt: Path) -> str:
    return ckpt.with_suffix("").with_suffix("").name


def parse_checkpoints_tree(checkpoints_tree) -> Tuple[Dict[str, Dict[str, Path]], Dict[str, Dict[str, str]]]:
    checkpoints = defaultdict(dict)
    RUNS = defaultdict(dict)
    for dataset_abbrv in sorted(checkpoints_tree.iterdir()):
        checkpoints[dataset_abbrv.name] = defaultdict(list)
        RUNS[dataset_abbrv.name] = defaultdict(list)
        for model_abbrv in sorted(dataset_abbrv.iterdir()):
            for ckpt in sorted(model_abbrv.iterdir()):
                checkpoints[dataset_abbrv.name][model_abbrv.name].append(ckpt)
                RUNS[dataset_abbrv.name][model_abbrv.name].append(parse_checkpoint_id(ckpt))
    return checkpoints, RUNS


def get_dataset(pl_module, ckpt):
    _, cfg = parse_checkpoint(
        module_class=pl_module,
        checkpoint_path=ckpt,
        map_location="cpu",
    )
    datamodule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    datamodule.setup()
    val_dataset = datamodule.val_datasets[0]
    return val_dataset


def parse_checkpoint(
    module_class: Type[nn.Module],
    checkpoint_path: Path,
    map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
) -> Tuple[nn.Module, DictConfig]:
    if checkpoint_path.name.endswith(".ckpt.zip"):
        checkpoint = NNCheckpointIO.load(path=checkpoint_path, map_location=map_location)
        model = module_class._load_model_state(
            checkpoint=checkpoint, metadata=checkpoint.get("metadata", None), strict=False
        )
        model.eval()
        return (
            model,
            OmegaConf.create(checkpoint["cfg"]),
        )
    raise ValueError(f"Wrong checkpoint: {checkpoint_path}")


def plot_bg(
    ax,
    df,
    cmap,
    norm,
    size=0.5,
    bg_alpha=0.01,
):
    """Create and return a plot of all our movie embeddings with very low opacity.

    (Intended to be used as a basis for further - more prominent - plotting of a
    subset of movies. Having the overall shape of the map space in the background is
    useful for context.)
    """
    ax.scatter(df.x, df.y, c=cmap(norm(df["target"])), alpha=bg_alpha, s=size)
    return ax


def hightlight_cluster(
    ax,
    df,
    target,
    alpha,
    cmap,
    norm,
    size=0.5,
):
    cluster_df = df[df["target"] == target]
    ax.scatter(cluster_df.x, cluster_df.y, c=cmap(norm(cluster_df["target"])), alpha=alpha, s=size)


def plot_latent_space(ax, df, targets, size, cmap, norm, bg_alpha=0.1, alpha=0.5):
    ax = plot_bg(ax, df, bg_alpha=bg_alpha, cmap=cmap, norm=norm)
    for target in targets:
        hightlight_cluster(ax, df, target, alpha=alpha, size=size, cmap=cmap, norm=norm)
    return ax
