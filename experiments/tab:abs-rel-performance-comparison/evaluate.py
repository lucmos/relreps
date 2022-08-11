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
from torchmetrics import Accuracy, F1Score, MetricCollection
from tqdm import tqdm

from nn_core.serialization import NNCheckpointIO

from rae.data.vision.datamodule import MyDataModule
from rae.modules.enumerations import Output
from rae.pl_modules.vision.pl_gclassifier import LightningClassifier

logging.getLogger().setLevel(logging.ERROR)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
CONSIDERED_METRICS = {
    "acc/weighted": lambda num_classes: Accuracy(average="weighted", num_classes=num_classes),
    "acc/micro": lambda num_classes: Accuracy(average="macro", num_classes=num_classes),
    "acc/macro": lambda num_classes: Accuracy(average="micro", num_classes=num_classes),
    "f1/macro": lambda num_classes: F1Score(average="macro", num_classes=num_classes),
    "f1/micro": lambda num_classes: F1Score(average="micro", num_classes=num_classes),
}

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


DATASETS = sorted(checkpoints.keys())
DATASET_NUM_CLASSES = {
    "mnist": 10,
    "fmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
}
MODELS = sorted(checkpoints[DATASETS[0]].keys())


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


def compute_predictions(force_predict: bool) -> pd.DataFrame:
    if PREDICTIONS_TSV.exists() and not force_predict:
        return pd.read_csv(PREDICTIONS_TSV, sep="\t", index_col=0)
    PREDICTIONS_TSV.unlink(missing_ok=True)

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
                model.cpu()
                del model

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(PREDICTIONS_TSV, sep="\t")
    return predictions_df


def measure_predictions(predictions_df: pd.DataFrame, force_measure: bool) -> pd.DataFrame:
    if PERFORMANCE_TSV.exists() and not force_measure:
        return pd.read_csv(PERFORMANCE_TSV, sep="\t", index_col=0)
    PERFORMANCE_TSV.unlink(missing_ok=True)

    performance = {**{x: [] for x in ("model_type", "dataset_name")}, **{k: [] for k in CONSIDERED_METRICS.keys()}}

    for dataset_name, dataset_pred in RUNS.items():
        for model_type, run_ids in dataset_pred.items():
            for run_id in run_ids:
                metrics = MetricCollection(
                    {
                        key: metric(num_classes=DATASET_NUM_CLASSES[dataset_name])
                        for key, metric in CONSIDERED_METRICS.items()
                    }
                )
                run_df = predictions_df[predictions_df["run_id"] == run_id]
                run_predictions = torch.as_tensor(run_df["pred"].values)
                run_targets = torch.as_tensor(run_df["target"].values)

                metrics.update(run_predictions, run_targets)

                performance["dataset_name"].append(dataset_name)
                performance["model_type"].append(model_type)
                for metric_name, metric_value in metrics.compute().items():
                    performance[metric_name].append(metric_value.item())

    performance_df = pd.DataFrame(performance)
    performance_df.to_csv(PERFORMANCE_TSV, sep="\t")
    return performance_df


def evaluate(force_predict: bool = False, force_measure: bool = True):
    predictions = compute_predictions(force_predict=force_predict)
    performance_df = measure_predictions(predictions, force_measure=force_measure)
    print(performance_df)


if __name__ == "__main__":
    typer.run(evaluate)
