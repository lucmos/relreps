import logging
from typing import Any, Dict, List, Mapping, Optional, Set

import hydra
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from rae.data.vision.datamodule import MetaData
from rae.modules.enumerations import Output, Stage, SupportedViz
from rae.pl_modules.pl_abstract_module import AbstractLightningModule
from rae.pl_modules.pl_visualizations import on_fit_end_viz, on_fit_start_viz
from rae.utils.tensor_ops import detach_tensors

pylogger = logging.getLogger(__name__)


class LightningClassifier(AbstractLightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata, *args, **kwargs)

        self.model = hydra.utils.instantiate(
            kwargs["model"],
            metadata=metadata,
            input_channels=metadata.anchors_images.shape[1],
            n_classes=len(metadata.class_to_idx),
            _recursive_=False,
        )
        self.loss = hydra.utils.instantiate(self.hparams.loss)

        self.register_buffer("anchors_images", self.metadata.anchors_images)
        self.register_buffer("anchors_latents", self.metadata.anchors_latents)
        self.register_buffer("fixed_images", self.metadata.fixed_images)

        self.df_columns = [
            "image_index",
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

        metric = torchmetrics.Accuracy()
        self.accuracies = nn.ModuleDict(
            {
                Stage.TRAIN_STAGE: metric.clone(),
                Stage.VAL_STAGE: metric.clone(),
            }
        )

        self.supported_viz = self.supported_viz()
        pylogger.info(f"Enabled visualizations: {str(sorted(x.value for x in self.supported_viz))}")

    def supported_viz(self) -> Set[SupportedViz]:
        supported_viz = set()

        # if self.fixed_images is not None:
        #     supported_viz.add(SupportedViz.VALIDATION_IMAGES_SOURCE)
        #
        # if self.anchors_images is not None:
        #     supported_viz.add(SupportedViz.ANCHORS_SOURCE)
        #
        # supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT)
        # supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT)
        # supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT_NORMALIZED)
        # supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT_NORMALIZED)
        return supported_viz

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.model(x, **kwargs)

    def step(self, batch, batch_index: int, stage: Stage) -> Mapping[str, Any]:
        image_batch = batch["image"]
        out = self(image_batch)

        loss = self.loss(out[Output.LOGITS], batch["target"])

        self.log_dict(
            {f"loss/{stage}": loss.cpu().detach()},
            on_step=stage == Stage.TRAIN_STAGE,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["image"].shape[0],
        )

        self.accuracies[stage](torch.softmax(out[Output.LOGITS], dim=-1), batch["target"])
        self.log_dict(
            {
                f"acc/{stage}": self.accuracies[stage],
            },
            on_epoch=True,
        )

        return {
            Output.LOSS: loss,
            Output.BATCH: {key: detach_tensors(value) for key, value in batch.items()},
            **{key: detach_tensors(value) for key, value in out.items()},
        }

    def on_epoch_start(self) -> None:
        self.model.set_finetune_mode()

    def on_fit_start(self) -> None:
        on_fit_start_viz(lightning_module=self, fixed_images=self.fixed_images, anchors_images=self.anchors_images)

    def on_fit_end(self) -> None:
        on_fit_end_viz(lightning_module=self, validation_stats_df=None)

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self.step(batch, batch_idx, stage=Stage.TRAIN_STAGE)

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self.step(batch, batch_idx, stage=Stage.VAL_STAGE)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        if self.trainer.sanity_checking:
            return

        # validation_aggregation = {}
        # for output in outputs:
        #     aggregate(
        #         validation_aggregation,
        #         image_index=output["batch"]["index"].cpu().tolist(),
        #         class_name=output["batch"]["class"],
        #         target=output["batch"]["target"].cpu(),
        #         latents=output[Output.DEFAULT_LATENT].cpu(),
        #         epoch=[self.current_epoch] * len(output["batch"]["index"]),
        #         is_anchor=[False] * len(output["batch"]["index"]),
        #         anchor_index=[None] * len(output["batch"]["index"]),
        #     )
        #
        # if self.anchors_images is not None:
        #     anchors_num = self.anchors_images.shape[0]
        #     anchors_out = self(self.anchors_images)
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
        #     image_index=self.metadata.anchors_idxs
        #     if self.metadata.anchors_idxs is not None
        #     else list(range(anchors_num)),
        #     class_name=self.metadata.anchors_classes if self.metadata.anchors_classes is not None else non_elements,
        #     target=self.metadata.anchors_targets.cpu() if self.metadata.anchors_targets is not None else non_elements,
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
        #
        # validation_epoch_end_viz(
        #     lightning_module=self,
        #     outputs=outputs,
        #     validation_stats_df=pd.DataFrame(validation_aggregation),
        #     anchors_reconstructed=None,
        #     anchors_latents=anchors_latents,
        #     fixed_images_out=self(self.fixed_images),
        # )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1")
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
