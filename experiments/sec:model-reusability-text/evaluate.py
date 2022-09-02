import itertools
import logging
from collections import defaultdict
from enum import auto
from pathlib import Path
from typing import Dict, Sequence, Tuple

import hydra
import pandas as pd
import rich
import torch
import typer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from tqdm import tqdm

from rae.data.text.datamodule import MyDataModule
from rae.modules.enumerations import Output
from rae.pl_modules.pl_stitching_module import StitchingModule
from rae.pl_modules.pl_text_classifier import LightningTextClassifier
from rae.utils.evaluation import parse_checkpoint, parse_checkpoint_id
from rae.utils.utils import to_device

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

logging.getLogger().setLevel(logging.ERROR)


BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXPERIMENT_ROOT = Path(__file__).parent
EXPERIMENT_CHECKPOINTS = EXPERIMENT_ROOT / "checkpoints"
PREDICTIONS_TSV = EXPERIMENT_ROOT / "predictions.tsv"
PERFORMANCE_TSV = EXPERIMENT_ROOT / "performance.tsv"

DATASET_SANITY = {
    "trec": ("rae.data.text.trec.TREC", "test"),
}

EMBEDDER_SANITY = ("fasttext", "word2vec")
MODEL_SANITY = {
    "absolute": "rae.modules.text.classifier.AbsoluteTextClassifier",
    "relative": "rae.modules.text.classifier.TextClassifier",
}


def parse_checkpoints_tree(
    checkpoints_tree,
) -> Tuple[Dict[str, Dict[str, Dict[str, Sequence[Path]]]], Dict[str, Dict[str, Dict[str, Sequence[str]]]]]:
    checkpoints = defaultdict(dict)
    RUNS = defaultdict(dict)
    for dataset_abbrv in sorted(checkpoints_tree.iterdir()):
        checkpoints[dataset_abbrv.name] = defaultdict(dict)
        RUNS[dataset_abbrv.name] = defaultdict(dict)

        for model_abbrv in sorted(dataset_abbrv.iterdir()):
            checkpoints[dataset_abbrv.name][model_abbrv.name] = defaultdict(list)
            RUNS[dataset_abbrv.name][model_abbrv.name] = defaultdict(list)

            for embedding_abbrv in sorted(model_abbrv.iterdir()):
                for ckpt in sorted(embedding_abbrv.iterdir()):
                    checkpoints[dataset_abbrv.name][model_abbrv.name][embedding_abbrv.name].append(ckpt)
                    RUNS[dataset_abbrv.name][model_abbrv.name][embedding_abbrv.name].append(parse_checkpoint_id(ckpt))

    return checkpoints, RUNS


checkpoints, RUNS = parse_checkpoints_tree(EXPERIMENT_CHECKPOINTS)

DATASETS = sorted(checkpoints.keys())
MODELS = sorted(checkpoints[DATASETS[0]].keys())


def compute_predictions(force_predict: bool) -> pd.DataFrame:
    if PREDICTIONS_TSV.exists() and not force_predict:
        return pd.read_csv(PREDICTIONS_TSV, sep="\t", index_col=0)
    rich.print("Computing the predictions")
    PREDICTIONS_TSV.unlink(missing_ok=True)

    predictions = {
        x: []
        for x in (
            "stitching",
            "run_id_a",
            "run_id_b",
            "model_type",
            "dataset_name",
            "embedder_a",
            "embedder_b",
            "sample_idx",
            "pred",
            "target",
        )
    }
    for dataset_name in (dataset_tqdm := tqdm(DATASETS, leave=True)):
        dataset_tqdm.set_description(f"Dataset ({dataset_name})")
        _, data_cfg = parse_checkpoint(
            module_class=LightningTextClassifier,
            checkpoint_path=checkpoints[dataset_name][MODELS[0]][EMBEDDER_SANITY[0]][0],
            map_location="cpu",
        )

        datamodule: MyDataModule = hydra.utils.instantiate(data_cfg.nn.data, _recursive_=False)
        datamodule.setup()
        val_dataset = datamodule.val_datasets[0]
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=16,
            persistent_workers=True,
            collate_fn=lambda x: x,
        )
        assert (
            f"{val_dataset.__module__}.{val_dataset.__class__.__name__}" == DATASET_SANITY[dataset_name][0]
        ), f"{val_dataset.__module__}.{val_dataset.__class__.__name__}!={DATASET_SANITY[dataset_name][0]}"
        assert val_dataset.split == DATASET_SANITY[dataset_name][1]

        for model_type in (model_type_tqdm := tqdm(MODELS, leave=True)):
            model_type_tqdm.set_description(f"Model type ({model_type})")

            for embedder_type_a, embedder_type_b in (
                embedder_tqdm := tqdm(
                    itertools.product(
                        checkpoints[dataset_name][model_type],
                        checkpoints[dataset_name][model_type],
                    ),
                    leave=False,
                )
            ):
                embedder_tqdm.set_description(f"Stitching: ({embedder_type_a}, {embedder_type_b})")
                assert len(checkpoints[dataset_name][model_type][embedder_type_a]) == len(
                    checkpoints[dataset_name][model_type][embedder_type_b]
                )
                for ckpt_a, ckpt_b in (
                    ckpt_tqdm := tqdm(
                        zip(
                            checkpoints[dataset_name][model_type][embedder_type_a],
                            checkpoints[dataset_name][model_type][embedder_type_b],
                        )
                    )
                ):
                    run_id_a = parse_checkpoint_id(ckpt_a)
                    run_id_b = parse_checkpoint_id(ckpt_b)
                    ckpt_tqdm.set_description(f"Stitching: ({run_id_a}, {run_id_b})")

                    model_a, _ = parse_checkpoint(
                        module_class=LightningTextClassifier,
                        checkpoint_path=ckpt_a,
                        map_location=DEVICE,
                    )
                    assert (
                        f"{model_a.model.__module__}.{model_a.model.__class__.__name__}" == MODEL_SANITY[model_type]
                    ), f"{model_a.model.__module__}.{model_a.model.__class__.__name__}!={MODEL_SANITY[model_type]}"

                    model_b, _ = parse_checkpoint(
                        module_class=LightningTextClassifier,
                        checkpoint_path=ckpt_b,
                        map_location=DEVICE,
                    )
                    assert (
                        f"{model_b.model.__module__}.{model_b.model.__class__.__name__}" == MODEL_SANITY[model_type]
                    ), f"{model_b.model.__module__}.{model_b.model.__class__.__name__}!={MODEL_SANITY[model_type]}"

                    model_ab = StitchingModule(model_a, model_b)
                    model_ab = model_ab.to(DEVICE)

                    for batch in tqdm(val_dataloader, desc="Batch", leave=False):
                        batch = to_device(model_a.model.text_encoder.collate_fn(batch=batch), device=DEVICE)

                        model_out = model_ab(batch, device=DEVICE)
                        pred = model_out[Output.INT_PREDICTIONS].cpu()

                        batch_size = len(batch["index"])
                        predictions["stitching"].extend([embedder_type_a != embedder_type_b] * batch_size)
                        predictions["run_id_a"].extend([run_id_a] * batch_size)
                        predictions["run_id_b"].extend([run_id_b] * batch_size)
                        predictions["embedder_a"].extend([embedder_type_a] * batch_size)
                        predictions["embedder_b"].extend([embedder_type_b] * batch_size)
                        predictions["model_type"].extend([model_type] * batch_size)
                        predictions["dataset_name"].extend([dataset_name] * batch_size)
                        predictions["sample_idx"].extend(batch["index"])
                        predictions["pred"].extend(pred.cpu().tolist())
                        predictions["target"].extend(batch["targets"].cpu().tolist())
                        del model_out
                        del batch
                    model_ab.cpu()
                    del model_ab

    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv(PREDICTIONS_TSV, sep="\t")
    return predictions_df


def compute_performance(predictions_df: pd.DataFrame, force_measure: bool) -> pd.DataFrame:
    if PERFORMANCE_TSV.exists() and not force_measure:
        return pd.read_csv(PERFORMANCE_TSV, sep="\t", index_col=0)
    rich.print("Computing the performance")

    CONSIDERED_METRICS = {
        "acc/macro": lambda num_classes: Accuracy(average="micro", num_classes=num_classes),
        "acc/micro": lambda num_classes: Accuracy(average="macro", num_classes=num_classes),
        "acc/weighted": lambda num_classes: Accuracy(average="weighted", num_classes=num_classes),
        "f1/macro": lambda num_classes: F1Score(average="macro", num_classes=num_classes),
        "f1/micro": lambda num_classes: F1Score(average="micro", num_classes=num_classes),
        "f1/weighted": lambda num_classes: F1Score(average="weighted", num_classes=num_classes),
        "recall/macro": lambda num_classes: Recall(average="macro", num_classes=num_classes),
        "recall/micro": lambda num_classes: Recall(average="micro", num_classes=num_classes),
        "recall/weighted": lambda num_classes: Recall(average="weighted", num_classes=num_classes),
        "precision/macro": lambda num_classes: Precision(average="macro", num_classes=num_classes),
        "precision/micro": lambda num_classes: Precision(average="micro", num_classes=num_classes),
        "precision/weighted": lambda num_classes: Precision(average="weighted", num_classes=num_classes),
    }

    PERFORMANCE_TSV.unlink(missing_ok=True)

    DATASET_NUM_CLASSES = {"trec-coarse": 6, "trec-fine": 24, "trec": 6}

    KEYS = ["stitching", "run_id_a", "run_id_b", "model_type", "dataset_name", "embedder_a", "embedder_b"]

    performance = {
        **{x: [] for x in KEYS},
        **{k: [] for k in CONSIDERED_METRICS.keys()},
    }

    predictions_df = predictions_df.groupby(KEYS)
    for (values, aggregate_df) in predictions_df:
        key2value = dict(zip(KEYS, values))
        aggregate_df: pd.DataFrame

        metrics = MetricCollection(
            {
                key: metric(num_classes=DATASET_NUM_CLASSES[key2value["dataset_name"]])
                for key, metric in CONSIDERED_METRICS.items()
            }
        )
        run_predictions = torch.as_tensor(aggregate_df["pred"].values)
        run_targets = torch.as_tensor(aggregate_df["target"].values)

        metrics.update(run_predictions, run_targets)

        for key, value in key2value.items():
            if key in performance:
                performance[key].append(value)

        for metric_name, metric_value in metrics.compute().items():
            performance[metric_name].append(metric_value.item())
    performance_df = pd.DataFrame(performance)

    return performance_df


class Display(StrEnum):
    DF = auto()
    LATEX = auto()


def display_performance(performance_df, display):
    raise NotImplementedError


def evaluate(force_predict: bool = False, force_measure: bool = False, display: Display = Display.DF):
    predictions_df = compute_predictions(force_predict=force_predict)
    performance_df = compute_performance(predictions_df, force_measure=force_measure)
    display_performance(performance_df, display=display)


if __name__ == "__main__":
    typer.run(evaluate)
