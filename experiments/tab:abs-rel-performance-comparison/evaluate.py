import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Type, Union

import hydra
import pandas as pd
import torch
import typer
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_core.serialization import NNCheckpointIO

from rae.data.vision.datamodule import MyDataModule
from rae.modules.enumerations import Output
from rae.pl_modules.vision.pl_gclassifier import LightningClassifier

logging.getLogger().setLevel(logging.ERROR)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EXPERIMENT_ROOT = Path(__file__).parent
EXPERIMENT_CHECKPOINTS = EXPERIMENT_ROOT / "checkpoints"
PREDICTIONS_CSV = EXPERIMENT_ROOT / "predictions.tsv"

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

# Parse checkpoints tree
checkpoints = defaultdict(dict)
for dataset_abbrv in EXPERIMENT_CHECKPOINTS.iterdir():
    checkpoints[dataset_abbrv.name] = defaultdict(list)
    for model_abbrv in dataset_abbrv.iterdir():
        for ckpt in model_abbrv.iterdir():
            checkpoints[dataset_abbrv.name][model_abbrv.name].append(ckpt)


def parse_checkpoint_id(ckpt: Path) -> str:
    return ckpt.with_suffix("").with_suffix("").name


DATASETS = sorted(checkpoints.keys())
MODELS = sorted(checkpoints[DATASETS[0]].keys())
RUN_IDS = sorted(
    parse_checkpoint_id(ckpt) for dataset, model in checkpoints.items() for _, runs in model.items() for ckpt in runs
)


def parse_checkpoint(
    module_class: Type[LightningModule],
    checkpoint_path: Path,
    map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
) -> Tuple[LightningModule, DictConfig]:
    if checkpoint_path.name.endswith(".ckpt.zip"):
        checkpoint = NNCheckpointIO.load(path=checkpoint_path, map_location=map_location)
        model = module_class._load_model_state(checkpoint=checkpoint, metadata=checkpoint.get("metadata", None))
        model.eval()
        return (
            model,
            OmegaConf.create(checkpoint["cfg"]),
        )
    raise ValueError(f"Wrong checkpoint: {checkpoint_path}")


def compute_predictions(force: bool = False) -> pd.DataFrame:
    if PREDICTIONS_CSV.exists() and not force:
        return pd.read_csv(PREDICTIONS_CSV)
    PREDICTIONS_CSV.unlink(missing_ok=True)

    predictions = {x: [] for x in ("run_id", "model_type", "dataset_name", "sample_idx", "pred", "target")}
    for dataset_name in (dataset_tqdm := tqdm(DATASETS, leave=True)):
        dataset_tqdm.set_description(f"Dataset ({dataset_name})")
        _, data_cfg = parse_checkpoint(
            module_class=LightningClassifier,
            checkpoint_path=checkpoints[dataset_name][MODELS[0]][0],
            map_location="cpu",
        )

        datamodule: MyDataModule = hydra.utils.instantiate(data_cfg.nn.data, _recursive_=False)
        datamodule.setup()
        val_dataset = datamodule.val_datasets[0]
        val_dataloder = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            persistent_workers=True,
        )
        assert (
            f"{val_dataset.__module__}.{val_dataset.__class__.__name__}" == DATASET_SANITY[dataset_name][0]
        ), f"{val_dataset.__module__}.{val_dataset.__class__.__name__}!={DATASET_SANITY[dataset_name][0]}"
        assert val_dataset.split == DATASET_SANITY[dataset_name][1]

        for model_type in (model_type_tqdm := tqdm(MODELS, leave=True)):
            model_type_tqdm.set_description(f"Model type ({model_type})")

            for ckpt in (ckpt_tqdm := tqdm(checkpoints[dataset_name][model_type], leave=False)):
                run_id = parse_checkpoint_id(ckpt)
                ckpt_tqdm.set_description(f"Run id ({run_id})")

                model, cfg = parse_checkpoint(
                    module_class=LightningClassifier,
                    checkpoint_path=ckpt,
                    map_location="cpu",
                )
                assert (
                    f"{model.model.__module__}.{model.model.__class__.__name__}" == MODEL_SANITY[model_type]
                ), f"{model.model.__module__}.{model.model.__class__.__name__}!={MODEL_SANITY[model_type]}"
                model = model.to(DEVICE)
                for batch in tqdm(val_dataloder, desc="Batch", leave=False):
                    model_out = model(batch["image"].to(DEVICE))
                    pred = model_out[Output.LOGITS].argmax(-1).cpu()

                    batch_size = len(batch["index"].cpu().tolist())
                    predictions["run_id"].extend([run_id] * batch_size)
                    predictions["model_type"].extend([model_type] * batch_size)
                    predictions["dataset_name"].extend([dataset_name] * batch_size)
                    predictions["sample_idx"].extend(batch["index"].cpu().tolist())
                    predictions["pred"].extend(pred.cpu().tolist())
                    predictions["target"].extend(batch["target"].cpu().tolist())
                    del model_out
                    del batch
                model = model.cpu()
                del model

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(PREDICTIONS_CSV, sep="\t")
    return predictions_df


# print(compute_predictions())
# exit()
#
#
# metric_collection = MetricCollection(
#     [
#         Accuracy(average="weighted"),
#         Accuracy(average="micro"),
#         Accuracy(average="macro"),
#         F1Score(average="macro"),
#     ]
# )


def evaluate(force: bool = False):
    compute_predictions(force=force)


if __name__ == "__main__":
    typer.run(evaluate)
