import logging
from typing import Any, Dict, List, Mapping, Optional

import hydra
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger
from torch import nn
from torchmetrics import Accuracy, F1Score, Precision, Recall

from rae.data.text.datamodule import MetaData
from rae.modules.enumerations import Output, Stage
from rae.pl_modules.pl_abstract_module import AbstractLightningModule

pylogger = logging.getLogger(__name__)


def _build_metrics(stage: Stage, num_classes: int) -> nn.ModuleDict:
    metrics: nn.ModuleDict = nn.ModuleDict(
        {
            f"f1/{stage}/{average if average != 'none' else ''}": F1Score(num_classes=num_classes, average=average)
            for average in ("macro", "micro", "weighted", "none")
        }
    )
    for average in ("macro", "micro", "weighted", "none"):
        metrics[f"precision/{stage}/{average if average != 'none' else ''}"] = Precision(
            num_classes=num_classes, average=average
        )

    for average in ("macro", "micro", "weighted", "none"):
        metrics[f"recall/{stage}/{average if average != 'none' else ''}"] = Recall(
            num_classes=num_classes, average=average
        )

    for average in ("macro", "micro", "weighted", "none"):
        metrics[f"accuracy/{stage}/{average if average != 'none' else ''}"] = Accuracy(
            num_classes=num_classes, average=average
        )

    return metrics


class LightningTextClassifier(AbstractLightningModule):
    logger: NNLogger
    metadata: MetaData

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata, *args, **kwargs)

        self.model = hydra.utils.instantiate(kwargs["model"], metadata=metadata, _recursive_=False)
        self.loss = hydra.utils.instantiate(self.hparams.loss)

        self.register_buffer("anchor_samples", self.metadata.anchor_samples)
        self.register_buffer("anchor_latents", self.metadata.anchor_latents)
        self.register_buffer("fixed_samples", self.metadata.fixed_samples)

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

    # def supported_viz(self) -> Set[SupportedViz]:
    #     supported_viz = set()
    #
    #     if self.fixed_images is not None:
    #         supported_viz.add(SupportedViz.VALIDATION_IMAGES_SOURCE)
    #
    #     if self.anchors_images is not None:
    #         supported_viz.add(SupportedViz.ANCHORS_SOURCE)
    #
    #     supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT)
    #     supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT)
    #     supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT_NORMALIZED)
    #     supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT_NORMALIZED)
    #     return supported_viz

    def forward(self, batch: Mapping[str, Any], *args, **kwargs) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        return self.model(batch, *args, **kwargs)

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
        for metric_name, metric in metrics.items():
            metric_out = metric(out[Output.INT_PREDICTIONS], batch["targets"].flatten())
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
        for metric_name, metric in self.train_stage_metrics.items():
            metric.reset()

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        for metric_name, metric in self.val_stage_metrics.items():
            metric.reset()

        if self.trainer.sanity_checking:
            return

        # validation_aggregation = {}
        # for output in outputs:
        #     aggregate(
        #         validation_aggregation,
        #         sample_index=output["batch"]["index"].cpu().tolist(),
        #         class_name=output["batch"]["classes"],
        #         target=output["batch"]["targets"].cpu(),
        #         latents=output[Output.DEFAULT_LATENT].cpu(),
        #         epoch=[self.current_epoch] * len(output["batch"]["index"]),
        #         is_anchor=[False] * len(output["batch"]["index"]),
        #         anchor_index=[None] * len(output["batch"]["index"]),
        #     )
        #
        # if self.anchor_samples is not None:
        #     anchors_num = len(self.metadata.anchor_idxs)
        #     anchors_out = self(
        #         dict(
        #             encodings=self.anchor_samples,
        #             sections=dict(
        #                 words_per_sentence=self.model.anchors_words_per_sentence,
        #                 words_per_text=self.model.anchors_words_per_text,
        #                 sentences_per_text=self.model.anchors_sentences_per_text,
        #             ),
        #         )
        #     )
        #     if Output.ANCHORS_LATENT in anchors_out:
        #         anchors_latents = anchors_out[Output.ANCHORS_LATENT]
        #     else:
        #         anchors_latents = anchors_out[Output.DEFAULT_LATENT]
        # else:
        #     raise NotImplementedError()
        #
        # non_elements = ["none"] * anchors_num
        # aggregate(
        #     validation_aggregation,
        #     sample_index=self.metadata.anchor_idxs
        #     if self.metadata.anchor_idxs is not None
        #     else list(range(anchors_num)),
        #     class_name=self.metadata.anchor_classes if self.metadata.anchor_classes is not None else non_elements,
        #     target=self.metadata.anchor_targets.cpu() if self.metadata.anchor_targets is not None else non_elements,
        #     latents=anchors_latents.cpu(),
        #     epoch=[self.current_epoch] * anchors_num,
        #     is_anchor=[True] * anchors_num,
        #     anchor_index=list(range(anchors_num)),
        # )
        #
        # latents = validation_aggregation["latents"]
        # self.fit_pca(latents)
        # add_2D_latents(validation_aggregation, latents=latents, pca=self.validation_pca)
        # del validation_aggregation["latents"]

        # validation_epoch_end_viz(
        #     lightning_module=self,
        #     outputs=outputs,
        #     validation_stats_df=pd.DataFrame(validation_aggregation),
        #     anchors_reconstructed=None,
        #     anchors_latents=anchors_latents,
        #     fixed_samples_out=self(self.fixed_samples),
        # )


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
