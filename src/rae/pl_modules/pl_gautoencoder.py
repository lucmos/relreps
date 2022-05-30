import logging
from typing import Any, Dict, List, Mapping, Optional, Set

import hydra
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from rae.data.datamodule import MetaData
from rae.modules.ae import AE
from rae.modules.enumerations import Output, Stage, SupportedViz
from rae.modules.rae_model import RAE, RaeDecoder
from rae.pl_modules.pl_abstract_module import AbstractLightningModule
from rae.pl_modules.pl_visualizations import on_fit_end_viz, on_fit_start_viz, validation_epoch_end_viz
from rae.utils.dataframe_op import cat_anchors_stats_to_dataframe, cat_output_to_dataframe
from rae.utils.tensor_ops import detach_tensors

pylogger = logging.getLogger(__name__)


class LightningAutoencoder(AbstractLightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata, *args, **kwargs)

        self.autoencoder = hydra.utils.instantiate(
            kwargs["model"] if "model" in kwargs else kwargs["autoencoder"], metadata=metadata
        )

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

        self.reconstruction_quality_metrics = {
            "mse": F.mse_loss,
            "l1": F.l1_loss,
        }

        self.register_buffer("anchors_images", self.metadata.anchors_images)
        self.register_buffer("anchors_latents", self.metadata.anchors_latents)
        self.register_buffer("fixed_images", self.metadata.fixed_images)

        self.validation_pca: Optional[PCA] = None

        self.loss = hydra.utils.instantiate(self.hparams.loss)

        self.supported_viz = self.supported_viz()
        pylogger.info(f"Enabled visualizations: {str(sorted(x.value for x in self.supported_viz))}")

    def supported_viz(self) -> Set[SupportedViz]:
        supported_viz = set()

        if self.fixed_images is not None:
            supported_viz.add(SupportedViz.VALIDATION_IMAGES_SOURCE)

        if self.anchors_images is not None:
            supported_viz.add(SupportedViz.ANCHORS_SOURCE)

        if isinstance(self.autoencoder, RAE):
            supported_viz.add(SupportedViz.INVARIANT_LATENT_DISTRIBUTION)

        supported_viz.add(SupportedViz.LATENT_EVOLUTION_PLOTLY_ANIMATION)
        supported_viz.add(SupportedViz.ANCHORS_RECONSTRUCTED)
        supported_viz.add(SupportedViz.VALIDATION_IMAGES_RECONSTRUCTED)
        supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT)
        supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT)
        supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT_NORMALIZED)
        supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT_NORMALIZED)
        supported_viz.add(SupportedViz.LATENT_SPACE)
        supported_viz.add(SupportedViz.LATENT_SPACE_NORMALIZED)
        supported_viz.add(SupportedViz.LATENT_SPACE_PCA)
        return supported_viz

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["validation_pca"] = self.validation_pca

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "validation_pca" in checkpoint:
            self.validation_pca = checkpoint["validation_pca"]
        else:
            self.validation_pca = PCA(n_components=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.autoencoder(x)

    def step(self, batch, batch_index: int, stage: Stage) -> Mapping[str, Any]:
        image_batch = batch["image"]
        out = self(image_batch)

        for metric_name, metric in self.reconstruction_quality_metrics.items():
            metric_value = metric(image_batch, out[Output.RECONSTRUCTION])
            self.log(
                f"{stage}/{metric_name}", metric_value, on_step=False, on_epoch=True, batch_size=image_batch.shape[0]
            )

        if isinstance(self.autoencoder, AE):
            loss = self.loss(
                out[Output.RECONSTRUCTION],
                image_batch,
            )
        else:
            loss = self.loss(
                out[Output.RECONSTRUCTION],
                image_batch,
                out[Output.LATENT_MU],
                out[Output.LATENT_LOGVAR],
            )

        self.log_dict(
            {f"loss/{stage}": loss.cpu().detach()},
            on_step=stage == Stage.TRAIN,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["image"].shape[0],
        )

        return {
            Output.LOSS: loss,
            Output.BATCH: batch,
            **{key: detach_tensors(value) for key, value in out.items()},
        }

    def on_fit_start(self) -> None:
        on_fit_start_viz(lightning_module=self, fixed_images=self.fixed_images, anchors_images=self.anchors_images)

    def on_fit_end(self) -> None:
        on_fit_end_viz(lightning_module=self, validation_stats_df=self.validation_stats_df)

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self.step(batch, batch_idx, stage=Stage.TRAIN)

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self.step(batch, batch_idx, stage=Stage.VAL)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        if self.trainer.sanity_checking:
            return

        if self.validation_pca is None or self.hparams.fit_pca_each_epoch:
            all_default_latents = torch.cat([x[Output.DEFAULT_LATENT] for x in outputs], dim=0)
            self.validation_pca = PCA(n_components=2)
            self.validation_pca.fit(all_default_latents)

        for output in outputs:
            self.validation_stats_df = cat_output_to_dataframe(
                validation_stats_df=self.validation_stats_df,
                output=output,
                current_epoch=self.current_epoch,
                pca=self.validation_pca,
            )

        if self.anchors_images is not None:
            anchors_num = self.anchors_images.shape[0]
            anchors_out = self(self.anchors_images)
            if Output.ANCHORS_LATENT in anchors_out:
                anchors_latents = anchors_out[Output.ANCHORS_LATENT]
            else:
                anchors_latents = anchors_out[Output.DEFAULT_LATENT]
            anchors_reconstructed = anchors_out.get(Output.RECONSTRUCTION, None)

        else:
            assert self.anchors_latents is not None
            anchors_num = self.anchors_latents.shape[0]
            # TODO: these latents should be normalized if the normalization is enabled..
            anchors_latents = self.anchors_latents
            if isinstance(self.autoencoder.decoder, RaeDecoder):
                anchors_reconstructed, _ = self.autoencoder.decoder(
                    self.anchors_latents, anchors_latents=self.anchors_latents
                )
            else:
                anchors_reconstructed = self.autoencoder.decoder(self.anchors_latents)

        self.validation_stats_df = cat_anchors_stats_to_dataframe(
            validation_stats_df=self.validation_stats_df,
            anchors_num=anchors_num,
            anchors_latents=anchors_latents,
            metadata=self.metadata,
            current_epoch=self.current_epoch,
            pca=self.validation_pca,
        )

        fixed_images_out = self(self.fixed_images)

        validation_epoch_end_viz(
            lightning_module=self,
            outputs=outputs,
            validation_stats_df=self.validation_stats_df,
            anchors_reconstructed=anchors_reconstructed,
            anchors_latents=anchors_latents,
            fixed_images_out=fixed_images_out,
        )


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
