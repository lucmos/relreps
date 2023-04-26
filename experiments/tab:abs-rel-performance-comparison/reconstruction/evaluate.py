import logging
from enum import auto
from pathlib import Path

import numpy as np
import pandas as pd
import rich
import torch
import typer
from torchmetrics import (
    ErrorRelativeGlobalDimensionlessSynthesis,
    MeanSquaredError,
    MetricCollection,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)

from rae.modules.enumerations import Output
from rae.pl_modules.pl_gautoencoder import LightningAutoencoder
from rae.utils.evaluation import parse_checkpoint_id, parse_checkpoints_tree

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

logging.getLogger().setLevel(logging.ERROR)


BATCH_SIZE = 256


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
    "vae": "rae.modules.vae.VanillaVAE",
    "ae": "rae.modules.ae.VanillaAE",
    "rel_vae": "rae.modules.rel_vae.VanillaRelVAE",
    "rel_ae": "rae.modules.rel_ae.VanillaRelAE",
}


checkpoints, RUNS = parse_checkpoints_tree(EXPERIMENT_CHECKPOINTS)

DATASETS = sorted(checkpoints.keys())
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
    from rae.utils.evaluation import parse_checkpoint

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CONSIDERED_METRICS = MetricCollection(
        {
            "mse": MeanSquaredError(),
            "ergas": ErrorRelativeGlobalDimensionlessSynthesis(),
            "psnr": PeakSignalNoiseRatio(),
            "ssim": StructuralSimilarityIndexMeasure(),
        }
    )

    predictions = {
        **{x: [] for x in ("run_id", "model_type", "dataset_name", "sample_idx")},
        **{k: [] for k in CONSIDERED_METRICS.keys()},
    }
    for dataset_name in (dataset_tqdm := tqdm(DATASETS, leave=True)):
        dataset_tqdm.set_description(f"Dataset ({dataset_name})")
        _, data_cfg = parse_checkpoint(
            module_class=LightningAutoencoder,
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
                    module_class=LightningAutoencoder,
                    checkpoint_path=ckpt,
                    map_location="cpu",
                )
                assert (
                    f"{model.autoencoder.__module__}.{model.autoencoder.__class__.__name__}" == MODEL_SANITY[model_type]
                ), f"{model.autoencoder.__module__}.{model.autoencoder.__class__.__name__}!={MODEL_SANITY[model_type]}"
                model = model.to(DEVICE)
                for batch in tqdm(val_dataloder, desc="Batch", leave=False):
                    model_out = model(batch["image"].to(DEVICE))

                    # COMPUTE ERRORS FOR EACH SAMPLE
                    for idx in range(batch["image"].shape[0]):
                        metrics = CONSIDERED_METRICS.clone()
                        metrics.update(model_out[Output.RECONSTRUCTION][[idx]].cpu(), batch["image"][[idx]].cpu())
                        for metric_name, metric_value in metrics.compute().items():
                            predictions[metric_name].append(metric_value.item())

                    batch_size = len(batch["index"].cpu().tolist())
                    predictions["run_id"].extend([run_id] * batch_size)
                    predictions["model_type"].extend([model_type] * batch_size)
                    predictions["dataset_name"].extend([dataset_name] * batch_size)
                    predictions["sample_idx"].extend(batch["index"].cpu().tolist())
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

    performance_df: pd.DataFrame = predictions_df.groupby(["run_id", "model_type", "dataset_name"]).agg([np.mean])
    performance_df = performance_df.droplevel(1, axis=1)
    performance_df = performance_df.drop(columns=["sample_idx"])
    performance_df = performance_df.reset_index()

    performance_df.to_csv(PERFORMANCE_TSV, sep="\t")
    return performance_df


class Display(StrEnum):
    DF = auto()
    LATEX = auto()


def display_performance(performance_df, display: Display):
    aggregated_performnace = performance_df.reset_index().drop(columns=["run_id"])
    aggregated_perfomance = aggregated_performnace.groupby(
        [
            "dataset_name",
            "model_type",
        ]
    ).agg([np.mean, np.std])
    aggregated_perfomance = aggregated_perfomance.round(6)
    aggregated_perfomance = (
        aggregated_perfomance[["mse", "ergas", "psnr", "ssim"]]
        .reindex(["ae", "rel_ae", "vae", "rel_vae"], level="model_type")
        .reindex(["mnist", "fmnist", "cifar10", "cifar100"], level="dataset_name")
    )

    if display == Display.DF:
        rich.print(aggregated_perfomance)

    elif display == Display.LATEX:
        COLUMN_ORDER = ["mnist", "fmnist", "cifar10", "cifar100"]
        METRIC_CONSIDERED = "mse"

        df = aggregated_perfomance[METRIC_CONSIDERED]
        reconstruction_str = r"{} & {} & {} & {} & {}  \\[1ex]"

        def latex_float(f):
            float_str = "{0:.2f}".format(f)
            if "e" in float_str:
                base, exponent = float_str.split("e")
                return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
            else:
                return float_str

        def extract_mean_std(df: pd.DataFrame, dataset_name: str, model_type: str, p=3) -> str:
            try:
                mean_std = df.loc[dataset_name, model_type]
                return rf"${latex_float(mean_std['mean'] * 10**p)} \pm {latex_float(mean_std['std'] *  10**p)} \times 10^{{{-p}}}$"
            except (AttributeError, KeyError):
                return "?"

        for available_model_type, available_model_name in zip(
            ("ae", "rel_ae", "vae", "rel_vae"), ("AE", "Relative AE", "VAE", "Relative VAE")
        ):
            s = reconstruction_str.format(
                available_model_name,
                *[extract_mean_std(df, dataset_name, available_model_type) for dataset_name in COLUMN_ORDER],
            )
            print(s)


def evaluate(force_predict: bool = False, force_measure: bool = False, display: Display = Display.DF):
    predictions_df = compute_predictions(force_predict=force_predict)
    performance_df = measure_predictions(predictions_df, force_measure=force_measure)
    display_performance(performance_df, display=display)


if __name__ == "__main__":
    typer.run(evaluate)
