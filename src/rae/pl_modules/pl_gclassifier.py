import logging
from typing import Any, Mapping, Optional, Sequence, Set, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from sklearn.decomposition import PCA
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from rae.data.datamodule import MetaData
from rae.modules.enumerations import Output, SupportedViz
from rae.utils.plotting import plot_images

pylogger = logging.getLogger(__name__)


class LightningClassifier(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        self.model = hydra.utils.instantiate(
            kwargs["model"],
            metadata=metadata,
            input_channels=metadata.anchors_images.shape[1],
            n_classes=len(metadata.class_to_idx),
        )

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

        self.supported_viz = self._determine_supported_viz()
        pylogger.info(f"Enabled visualizations: {str(sorted(x.value for x in self.supported_viz))}")

        self.validation_pca: Optional[PCA] = None

        metric = torchmetrics.Accuracy()
        # FIXME: workaround to avoid lightnign error of missing attribute
        self.train_accuracy = metric.clone()
        self.validation_accuracy = metric.clone()

        self.accuracies = {
            "train": self.train_accuracy,
            "validation": self.validation_accuracy,
        }

    def _determine_supported_viz(self) -> Set[SupportedViz]:
        supported_viz = set()

        if self.fixed_images is not None:
            supported_viz.add(SupportedViz.VALIDATION_IMAGES_SOURCE)

        if self.anchors_images is not None:
            supported_viz.add(SupportedViz.ANCHORS_SOURCE)

        supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT)
        supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT)
        supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT_NORMALIZED)
        supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT_NORMALIZED)
        return supported_viz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.model(x)

    def step(self, batch, batch_index: int, stage: str) -> Mapping[str, Any]:
        image_batch = batch["image"]
        out = self(image_batch)

        loss = F.cross_entropy(out[Output.LOGITS], batch["target"])

        self.accuracies[stage](torch.softmax(out[Output.LOGITS], dim=-1), batch["target"])
        self.log_dict(
            {
                f"acc/{stage}": self.accuracies[stage],
            },
            on_epoch=True,
        )

        return {
            Output.LOSS: loss,
            Output.BATCH: batch,
            **{key: self.normalize_output(value) for key, value in out.items()},
        }

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]

    def on_fit_start(self) -> None:
        to_log = {}
        to_close = set()
        if SupportedViz.VALIDATION_IMAGES_SOURCE in self.supported_viz:
            to_log["images/source"] = plot_images(self.fixed_images, "Source images")
            to_close.add(to_log["images/source"])

        if SupportedViz.ANCHORS_SOURCE in self.supported_viz:
            to_log["anchors/source"] = plot_images(self.anchors_images, "Anchors images")
            to_close.add(to_log["anchors/source"])

        if to_log:
            self.logger.experiment.log(to_log, step=self.global_step)

        for fig in to_close:
            plt.close(fig)

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch, batch_idx, stage="train")

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["image"].shape[0],
        )
        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch, batch_idx, stage="validation")

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["image"].shape[0],
        )
        return step_out

    # def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
    #     if self.trainer.sanity_checking:
    #         return
    #
    #     if self.validation_pca is None or self.hparams.fit_pca_each_epoch:
    #         all_default_latents = torch.cat([x[Output.DEFAULT_LATENT] for x in outputs], dim=0)
    #         self.validation_pca = PCA(n_components=2)
    #         self.validation_pca.fit(all_default_latents)
    #
    #     for output in outputs:
    #         self.validation_stats_df = cat_output_to_dataframe(
    #             validation_stats_df=self.validation_stats_df,
    #             output=output,
    #             current_epoch=self.current_epoch,
    #             pca=self.validation_pca,
    #         )
    #
    #     if self.anchors_images is not None:
    #         anchors_num = self.anchors_images.shape[0]
    #         anchors_out = self(self.anchors_images)
    #         anchors_latents = anchors_out[Output.ANCHORS_LATENT]
    #         anchors_reconstructed = anchors_out[Output.RECONSTRUCTION]
    #
    #     else:
    #         assert self.anchors_latents is not None
    #         anchors_num = self.anchors_latents.shape[0]
    #         # TODO: these latents should be normalized if the normalization is enabled..
    #         anchors_latents = self.anchors_latents
    #         if isinstance(self.autoencoder.decoder, RaeDecoder):
    #             anchors_reconstructed, _ = self.autoencoder.decoder(
    #                 self.anchors_latents, anchors_latents=self.anchors_latents
    #             )
    #         else:
    #             anchors_reconstructed = self.autoencoder.decoder(self.anchors_latents)
    #
    #     self.validation_stats_df = cat_anchors_stats_to_dataframe(
    #         validation_stats_df=self.validation_stats_df,
    #         anchors_num=anchors_num,
    #         anchors_latents=anchors_latents,
    #         metadata=self.metadata,
    #         current_epoch=self.current_epoch,
    #         pca=self.validation_pca,
    #     )
    #
    #     fixed_images_out = self(self.fixed_images)
    #
    #     to_log = {}
    #     to_close = set()
    #     if SupportedViz.ANCHORS_RECONSTRUCTED in self.supported_viz:
    #         to_log["anchors/reconstructed"] = plot_images(anchors_reconstructed, "Anchors reconstructed")
    #         to_close.add(to_log["anchors/reconstructed"])
    #
    #     if SupportedViz.VALIDATION_IMAGES_RECONSTRUCTED in self.supported_viz:
    #         to_log["images/reconstructed"] = plot_images(
    #             fixed_images_out[Output.RECONSTRUCTION], "Reconstructed images"
    #         )
    #         to_close.add(to_log["images/reconstructed"])
    #
    #     if SupportedViz.ANCHORS_SELF_INNER_PRODUCT in self.supported_viz:
    #         anchors_self_inner_product = anchors_latents @ anchors_latents.T
    #         to_log["anchors-vs-anchors/inner"] = plot_matrix(
    #             anchors_self_inner_product,
    #             title="Anchors vs Anchors inner products",
    #             labels={"x": "anchors", "y": "anchors"},
    #         )
    #
    #     if SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT in self.supported_viz:
    #         batch_latent = fixed_images_out[Output.BATCH_LATENT]
    #         anchors_batch_latents_inner_product = anchors_latents @ batch_latent.T
    #         to_log["anchors-vs-samples/inner"] = plot_matrix(
    #             anchors_batch_latents_inner_product,
    #             title="Anchors vs Samples images inner products",
    #             labels={"x": "samples", "y": "anchors"},
    #         )
    #
    #     if SupportedViz.ANCHORS_SELF_INNER_PRODUCT_NORMALIZED in self.supported_viz:
    #         anchors_latents_normalized = F.normalize(anchors_latents, p=2, dim=-1)
    #         anchors_self_inner_product_normalized = anchors_latents_normalized @ anchors_latents_normalized.T
    #         to_log["anchors-vs-anchors/inner-normalized"] = plot_matrix(
    #             anchors_self_inner_product_normalized,
    #             title="Anchors vs Anchors inner products",
    #             labels={"x": "anchors", "y": "anchors"},
    #         )
    #
    #     if SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT_NORMALIZED in self.supported_viz:
    #         batch_latent = fixed_images_out[Output.BATCH_LATENT]
    #         anchors_latents_normalized = F.normalize(anchors_latents, p=2, dim=-1)
    #         batch_latent_normalized = F.normalize(batch_latent, p=2, dim=-1)
    #         anchors_batch_latents_inner_product_normalized = anchors_latents_normalized @ batch_latent_normalized.T
    #         to_log["anchors-vs-samples/inner-normalized"] = plot_matrix(
    #             anchors_batch_latents_inner_product_normalized,
    #             title="Anchors vs Samples images inner products",
    #             labels={"x": "samples", "y": "anchors"},
    #         )
    #
    #     if SupportedViz.INVARIANT_LATENT_DISTRIBUTION in self.supported_viz:
    #         fig = plot_violin(
    #             torch.cat([output[Output.INV_LATENTS] for output in outputs], dim=0),
    #             title="Relative Latent Space distribution",
    #             y_label="validation distribution",
    #             x_label="anchors",
    #         )
    #         to_log["distributions/invariant-latent-space"] = wandb.Image(fig)
    #         to_close.add(fig)
    #
    #     if SupportedViz.LATENT_SPACE in self.supported_viz:
    #         to_log["latent/space"] = plot_latent_space(
    #             metadata=self.metadata,
    #             validation_stats_df=self.validation_stats_df,
    #             epoch=self.current_epoch,
    #             x_data="latent_0",
    #             y_data="latent_1",
    #             n_samples=self.hparams.plot_n_val_samples,
    #         )
    #
    #     if SupportedViz.LATENT_SPACE_NORMALIZED in self.supported_viz:
    #         to_log["latent/space-normalized"] = plot_latent_space(
    #             metadata=self.metadata,
    #             validation_stats_df=self.validation_stats_df,
    #             epoch=self.current_epoch,
    #             x_data="latent_0_normalized",
    #             y_data="latent_1_normalized",
    #             n_samples=self.hparams.plot_n_val_samples,
    #         )
    #
    #     if SupportedViz.LATENT_SPACE_PCA in self.supported_viz:
    #         to_log["latent/space-pca"] = plot_latent_space(
    #             metadata=self.metadata,
    #             validation_stats_df=self.validation_stats_df,
    #             epoch=self.current_epoch,
    #             x_data="latent_0_pca",
    #             y_data="latent_1_pca",
    #             n_samples=self.hparams.plot_n_val_samples,
    #         )
    #
    #     if to_log:
    #         self.logger.experiment.log(to_log, step=self.global_step)
    #
    #     for fig in to_close:
    #         plt.close(fig)

    def normalize_output(self, x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu()
        else:
            return x


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
