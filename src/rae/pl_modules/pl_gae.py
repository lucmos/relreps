import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import hydra
import matplotlib.pyplot as plt
import omegaconf
import pandas as pd
import plotly
import plotly.express as px
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision
import wandb
from matplotlib.figure import Figure
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from rae.data.datamodule import MetaData
from rae.modules.output_keys import Output
from rae.modules.rae import RaeDecoder

pylogger = logging.getLogger(__name__)


class LightningGAE(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        self.autoencoder = hydra.utils.instantiate(kwargs["autoencoder"], metadata=metadata)

        self.df_columns = [
            "image_index",
            "class",
            "target",
            "latent_0",
            "latent_1",
            "epoch",
            "is_anchor",
        ]

        self.validation_stats_df: pd.DataFrame = pd.DataFrame(columns=self.df_columns)

        self.reconstruction_quality_metrics = {
            "mse": F.mse_loss,
            "l1": F.l1_loss,
        }

        self.register_buffer("anchors_images", self.metadata.anchors_images)
        self.register_buffer("anchors_latents", self.metadata.anchors_latents)
        self.register_buffer("fixed_images", self.metadata.fixed_images)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.autoencoder(x)

    def step(self, batch, batch_index: int, stage: str) -> Mapping[str, Any]:
        image_batch = batch["image"]
        out = self(image_batch)

        for metric_name, metric in self.reconstruction_quality_metrics.items():
            metric_value = metric(image_batch, out[Output.OUT])
            self.log(f"{stage}/{metric_name}", metric_value, on_step=False, on_epoch=True)

        return out

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        if self.trainer.sanity_checking:
            return

        for output in outputs:
            self.validation_stats_df = pd.concat(
                [
                    self.validation_stats_df,
                    pd.DataFrame(
                        {
                            "image_index": output["batch"]["index"].cpu(),
                            "class": output["batch"]["class"],
                            "target": output["batch"]["target"].cpu(),
                            "latent_0": output["default_latent"][:, 0].cpu(),
                            "latent_1": output["default_latent"][:, 1].cpu(),
                            # "std_0": output["latent_logvar"][:, 0],
                            # "std_1": output["latent_logvar"][:, 1],
                            "epoch": [self.current_epoch] * len(output["batch"]["index"]),
                            "is_anchor": [False] * len(output["batch"]["index"]),
                        }
                    ),
                ],
                ignore_index=False,
            )

        if self.anchors_images is not None:
            anchors_out = self(self.anchors_images)
            anchors_num = self.anchors_images.shape[0]
            non_elements = ["none"] * anchors_num
            self.validation_stats_df = pd.concat(
                [
                    self.validation_stats_df,
                    pd.DataFrame(
                        {
                            "image_index": self.metadata.anchors_idxs
                            if self.metadata.anchors_idxs is not None
                            else list(range(anchors_num)),
                            "class": self.metadata.anchors_classes
                            if self.metadata.anchors_classes is not None
                            else non_elements,
                            "target": self.metadata.anchors_targets.cpu()
                            if self.metadata.anchors_targets is not None
                            else non_elements,
                            "latent_0": anchors_out[anchors_out[Output.DEFAULT_LATENT]][:, 0].cpu(),
                            "latent_1": anchors_out[anchors_out[Output.DEFAULT_LATENT]][:, 1].cpu(),
                            # "std_0": anchors_latent_std[:, 0],
                            # "std_1": anchors_latent_std[:, 1],
                            "epoch": [self.current_epoch] * anchors_num,
                            "is_anchor": [True] * anchors_num,
                        }
                    ),
                ],
                ignore_index=False,
            )
            anchors_reconstructed = anchors_out[Output.OUT]
        else:
            assert self.anchors_latents is not None
            if isinstance(self.autoencoder.decoder, RaeDecoder):
                anchors_reconstructed, _ = self.autoencoder.decoder(
                    self.anchors_latents, anchors_latents=self.anchors_latents
                )
            else:
                anchors_reconstructed = self.autoencoder.decoder(self.anchors_latents)

            anchors_num = self.anchors_latents.shape[0]
            non_elements = ["none"] * anchors_num

            self.validation_stats_df = pd.concat(
                [
                    self.validation_stats_df,
                    pd.DataFrame(
                        {
                            "image_index": self.metadata.anchors_idxs
                            if self.metadata.anchors_idxs is not None
                            else list(range(anchors_num)),
                            "class": non_elements,
                            "target": non_elements,
                            "latent_0": self.anchors_latents[:, 0].cpu(),
                            "latent_1": self.anchors_latents[:, 1].cpu(),
                            # "std_0": anchors_latent_std[:, 0],
                            # "std_1": anchors_latent_std[:, 1],
                            "epoch": [self.current_epoch] * anchors_num,
                            "is_anchor": [True] * anchors_num,
                        }
                    ),
                ],
                ignore_index=False,
            )

        fixed_images_out = self(self.fixed_images)[Output.OUT]

        self.logger.experiment.log(
            {
                "images/reconstructed": self.plot_images(fixed_images_out, "Reconstructed images"),
                "anchors/reconstructed": self.plot_images(
                    anchors_reconstructed, "Anchors reconstructed", figsize=(17, 4)
                ),
            },
            step=self.global_step,
        )

    def on_fit_start(self) -> None:
        fixed_images_fig = self.plot_images(self.fixed_images, "Source images")
        self.logger.experiment.log(
            {"images/source": fixed_images_fig},
            step=self.global_step,
        )

        if self.anchors_images is not None:
            self.logger.experiment.log(
                {"anchors/source": self.plot_images(self.anchors_images, "Anchors images", figsize=(17, 4))},
                step=self.global_step,
            )

    @staticmethod
    def plot_images(images: torch.Tensor, title: str, figsize: Optional[Tuple[int, int]] = None) -> Figure:
        fig, ax = plt.subplots(1, 1, figsize=(17, 9) if figsize is None else figsize)
        ax.set_title(title)
        ax.axis("off")
        fig.set_tight_layout(tight=True)
        ax.imshow(torchvision.utils.make_grid(images.cpu(), 10, 5).permute(1, 2, 0))

        # Plotly version
        # fig = px.imshow(torchvision.utils.make_grid(images.cpu(), 10, 5).permute(1, 2, 0), title=title)
        return fig

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch, batch_idx, stage="train")

        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        step_out = self.step(batch, batch_idx, stage="validation")

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return step_out

    def on_fit_end(self) -> None:
        n_samples = self.hparams.plot_n_val_samples
        color_discrete_map = {
            class_name: color
            for class_name, color in zip(
                self.metadata.class_to_idx, px.colors.qualitative.Plotly[: len(self.metadata.class_to_idx)]
            )
        }

        latent_val_fig = px.scatter(
            self.validation_stats_df.loc[self.validation_stats_df["image_index"] < n_samples],
            x="latent_0",
            y="latent_1",
            animation_frame="epoch",
            animation_group="image_index",
            category_orders={"class": self.metadata.class_to_idx.keys()},
            #             # size='std_0',  # TODO: fixme, plotly crashes with any column name to set the anchor size
            color="class",
            hover_name="image_index",
            facet_col="is_anchor",
            color_discrete_map=color_discrete_map,
            # symbol="is_anchor",
            # symbol_map={False: "circle", True: "star"},
            size_max=40,
            range_x=[-5, 5],
            color_continuous_scale=None,
            range_y=[-5, 5],
        )

        # Convert to HTML as a workaround to https://github.com/wandb/client/issues/2191
        self.logger.experiment.log(
            {
                "latent": wandb.Html(plotly.io.to_html(latent_val_fig), inject=True),
            }
        )

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
