import logging
from typing import Any, Dict, List, Mapping, Optional, Set

import hydra
import omegaconf
import pandas as pd
import plotly.express as px
import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from rae.data.text.datamodule import MetaData
from rae.modules.enumerations import Output, Stage, SupportedViz
from rae.pl_modules.pl_abstract_module import AbstractLightningModule

pylogger = logging.getLogger(__name__)


def _build_metrics(stage: Stage, num_classes: int) -> nn.ModuleDict:
    metrics = MetricCollection(
        [
            MetricCollection(
                [metric(num_classes=num_classes, average=average) for metric in (F1Score, Precision, Recall, Accuracy)],
                prefix=f"{stage}/",
                postfix=f"/{average if average != 'none' else ''}",
            )
            for average in ("macro", "micro", "weighted", "none")
        ]
    )

    return metrics


class LightningTextClassifier(AbstractLightningModule):
    logger: NNLogger
    metadata: MetaData

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata, *args, **kwargs)

        self.model = hydra.utils.instantiate(kwargs["model"], metadata=metadata, _recursive_=False)
        self.loss = hydra.utils.instantiate(self.hparams.loss)

        # self.register_buffer("anchor_samples", self.metadata.anchor_samples)
        # self.register_buffer("anchor_latents", self.metadata.anchor_latents)
        # self.register_buffer("fixed_samples", self.metadata.fixed_samples)

        self.df_columns = [
            "sample_index",
            "class",
            "target",
            "latent_0",
            "latent_1",
            "latent_0_normalized",
            "latent_1_normalized",
            "epoch",
            "is_anchor",
            "anchor",
        ]

        self.validation_stats_df: pd.DataFrame = pd.DataFrame(columns=self.df_columns)

        num_classes: int = len(self.metadata.class_to_idx)
        self.train_stage_metrics: nn.ModuleDict = _build_metrics(stage=Stage.TRAIN_STAGE, num_classes=num_classes)
        self.val_stage_metrics: nn.ModuleDict = _build_metrics(stage=Stage.VAL_STAGE, num_classes=num_classes)

        # self.supported_viz = self.supported_viz()
        # pylogger.info(f"Enabled visualizations: {str(sorted(x.value for x in self.supported_viz))}")

    def supported_viz(self) -> Set[SupportedViz]:
        supported_viz = set()

        # supported_viz.add(SupportedViz.LATENT_SPACE_PCA)

        return supported_viz

    def encode(self, *args, **kwargs):
        return self.model.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.model.decode(*args, **kwargs)

    def forward(self, batch: Mapping[str, Any], *args, **kwargs) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        encoding = self.encode(batch, *args, device=self.device, **kwargs)
        return self.decode(**encoding)

    def step(self, batch, batch_index: int, stage: Stage) -> Mapping[str, Any]:
        out = self(batch)

        loss = self.loss(out[Output.LOGITS], batch["targets"])

        self.log_dict(
            {f"loss/{stage}": loss.cpu().detach()},
            on_step=stage == Stage.TRAIN_STAGE,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["targets"].shape[0],
        )

        metrics = eval(f"self.{stage.lower()}_metrics")
        metrics_out = metrics(out[Output.INT_PREDICTIONS], batch["targets"].flatten())
        for metric_name, metric_out in metrics_out.items():
            # If this metric had no previous aggregation...
            if metric_name.endswith("/"):
                metric_out = {
                    f"{metric_name}{label}": score
                    for label, score in zip(self.metadata.class_to_idx.keys(), metric_out)
                }

            if isinstance(metric_out, torch.Tensor):
                self.log(name=metric_name, value=metric_out)
            elif isinstance(metric_out, dict):
                self.log_dict(dictionary=metric_out)
            else:
                assert False

        return {
            Output.LOSS: loss,
            # Output.BATCH: {key: detach_tensors(value) for key, value in batch.items()},
            # **{key: detach_tensors(value) for key, value in out.items()},
        }

    def on_epoch_start(self) -> None:
        self.model.set_finetune_mode()

    def on_fit_start(self) -> None:
        # TODO: where does self.fixed_images come from?!
        # on_fit_start_viz(lightning_module=self, fixed_images=self.fixed_images, anchors_images=self.anchors_images)
        pass

    def on_fit_end(self) -> None:
        pass
        # on_fit_end_viz(lightning_module=self, validation_stats_df=None)

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self.step(batch, batch_idx, stage=Stage.TRAIN_STAGE)

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self.step(batch, batch_idx, stage=Stage.VAL_STAGE)

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        self.train_stage_metrics.reset()

    def state_dict(self, *args, **kwargs):
        result = super(LightningTextClassifier, self).state_dict(*args, **kwargs)
        result = {k: v for k, v in result.items() if "transformer" not in k}
        return result

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        self.val_stage_metrics.reset()

        if self.trainer.sanity_checking:
            return

        # data = {
        #     "is_anchor": [],
        #     "image_index": [],
        #     "anchor_index": [],
        #     "class_name": [],
        #     "latent_dim_0": [],
        #     "latent_dim_1": [],
        # }
        # latents: List = []
        #
        # data["class_name"] = self.metadata.anchor_classes + self.metadata.fixed_sample_classes
        # data["image_index"] = self.metadata.anchor_idxs + self.metadata.fixed_sample_idxs
        # data["anchor_index"] = self.metadata.anchor_idxs + len(self.metadata.fixed_sample_idxs) * [None]
        # data["is_anchor"] = len(self.metadata.anchor_idxs) * [True] + len(self.metadata.fixed_sample_idxs) * [False]
        #
        # for batch in chunk_iterable(self.metadata.anchor_samples + self.metadata.fixed_samples, 128):
        #     batch = to_device(self.model.text_encoder.collate_fn(batch=batch), device=self.device)
        #     batch_encoding = self.encode(batch=batch, device=self.device)
        #     latents.append(batch_encoding[Output.DEFAULT_LATENT])
        #
        # latents: torch.Tensor = torch.cat(latents, dim=0)
        # pca_latents = PCA(n_components=2).fit_transform(X=latents.detach().cpu().numpy())
        # data["latent_dim_0"] = pca_latents[:, 0]
        # data["latent_dim_1"] = pca_latents[:, 1]
        #
        # latent_val_fig = plot_latent_space(
        #     metadata=self.metadata,
        #     validation_stats_df=pd.DataFrame(data),
        #     x_data="latent_dim_0",
        #     y_data="latent_dim_1",
        # )
        # self.logger.experiment.log({"sPaCe1!1": latent_val_fig}, step=self.global_step)


def plot_latent_space(metadata, validation_stats_df, x_data: str, y_data: str):
    color_discrete_map = {
        class_name: color
        for class_name, color in zip(metadata.class_to_idx, px.colors.qualitative.Plotly[: len(metadata.class_to_idx)])
    }

    latent_val_fig = px.scatter(
        validation_stats_df,
        x=x_data,
        y=y_data,
        category_orders={"class_name": metadata.class_to_idx.keys()},
        #             # size='std_0',  # TODO: fixme, plotly crashes with any column name to set the anchor size
        color="class_name",
        hover_name="image_index",
        hover_data=["image_index", "anchor_index"],
        facet_col="is_anchor",
        color_discrete_map=color_discrete_map,
        # symbol="is_anchor",
        # symbol_map={False: "circle", True: "star"},
        size_max=40,
        range_x=[-5, 5],
        color_continuous_scale=None,
        range_y=[-5, 5],
    )
    return latent_val_fig


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.module,
        metadata=datamodule.metadata,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
