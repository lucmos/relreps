import logging
from enum import auto
from pathlib import Path

import numpy as np
import pandas as pd
import rich
import torch
import typer

from rae.utils.evaluation import parse_checkpoint_id, parse_checkpoints_tree

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


checkpoints, RUNS = parse_checkpoints_tree(EXPERIMENT_CHECKPOINTS)


DATASETS = sorted(checkpoints.keys())
DATASET_NUM_CLASSES = {
    "mnist": 10,
    "fmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
}
MODELS = sorted(checkpoints[DATASETS[0]].keys())


def compute_predictions(force_predict: bool) -> pd.DataFrame:
    if PREDICTIONS_TSV.exists() and not force_predict:
        return pd.read_csv(PREDICTIONS_TSV, sep="\t", index_col=0)
    rich.print("Computing the predictions")
    PREDICTIONS_TSV.unlink(missing_ok=True)

    import hydra
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from rae.data.vision.datamodule import MyDataModule
    from rae.modules.enumerations import Output
    from rae.pl_modules.vision.pl_gclassifier import LightningClassifier
    from rae.utils.evaluation import parse_checkpoint

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    rich.print("Computing the performance")

    from torchmetrics import Accuracy, F1Score, MetricCollection

    CONSIDERED_METRICS = {
        "acc/weighted": lambda num_classes: Accuracy(average="weighted", num_classes=num_classes),
        "acc/micro": lambda num_classes: Accuracy(average="macro", num_classes=num_classes),
        "acc/macro": lambda num_classes: Accuracy(average="micro", num_classes=num_classes),
        "f1/macro": lambda num_classes: F1Score(average="macro", num_classes=num_classes),
        "f1/micro": lambda num_classes: F1Score(average="micro", num_classes=num_classes),
    }

    PERFORMANCE_TSV.unlink(missing_ok=True)

    performance = {
        **{x: [] for x in ("run_id", "model_type", "dataset_name")},
        **{k: [] for k in CONSIDERED_METRICS.keys()},
    }

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

                performance["run_id"].append(run_id)
                performance["dataset_name"].append(dataset_name)
                performance["model_type"].append(model_type)
                for metric_name, metric_value in metrics.compute().items():
                    performance[metric_name].append(metric_value.item())

    performance_df = pd.DataFrame(performance)
    performance_df.to_csv(PERFORMANCE_TSV, sep="\t")
    return performance_df


class Display(StrEnum):
    DF = auto()
    LATEX = auto()


def display_performance(performance_df, display: Display):
    aggregated_performnace = performance_df.drop(columns=["run_id"])
    aggregated_perfomance = aggregated_performnace.groupby(
        [
            "dataset_name",
            "model_type",
        ]
    ).agg([np.mean, np.std])
    aggregated_perfomance = (aggregated_perfomance * 100).round(2)

    if display == Display.DF:
        rich.print(aggregated_perfomance)

    elif display == Display.LATEX:
        COLUMN_ORDER = ["mnist", "cmnist", "fmnist", "cifar10", "cifar100", "shapenet", "faust", "coma", "amz"]
        METRIC_CONSIDERED = "f1/macro"

        df = aggregated_perfomance[METRIC_CONSIDERED]
        classification_rel = r"Relative & {} & {} & {} & {} & {} & {} & {} & {} & {} \\[1ex]"
        classification_abs = r"Absolute & {} & {} & {} & {} & {} & {} & {} & {} & {} \\[1ex]"

        def extract_mean_std(df: pd.DataFrame, dataset_name: str, model_type: str) -> str:
            try:
                mean_std = df.loc[dataset_name, model_type]
                return rf"${mean_std['mean']:.2f} \pm {mean_std['std']:.2f}$"
            except (AttributeError, KeyError):
                return "?"

        classification_rel = classification_rel.format(
            *[extract_mean_std(df, dataset_name, "rel") for dataset_name in COLUMN_ORDER]
        )
        classification_abs = classification_abs.format(
            *[extract_mean_std(df, dataset_name, "abs") for dataset_name in COLUMN_ORDER]
        )

        print(classification_rel)
        print(classification_abs)


def evaluate(force_predict: bool = False, force_measure: bool = False, display: Display = Display.DF):
    predictions_df = compute_predictions(force_predict=force_predict)
    performance_df = measure_predictions(predictions_df, force_measure=force_measure)
    display_performance(performance_df, display=display)


if __name__ == "__main__":
    typer.run(evaluate)
