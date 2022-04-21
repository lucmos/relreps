import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import pandas as pd
import plotly
import plotly.express as px
import pytorch_lightning as pl
import torch
import wandb
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from rae.data.datamodule import MetaData
from rae.losses.vae_loss import vae_loss

pylogger = logging.getLogger(__name__)


class RAE(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        self.vae = hydra.utils.instantiate(kwargs["autoencoder"], metadata=metadata)

        self.df_columns = ["image_index", "class", "target", "mu_0", "mu_1", "std_0", "std_1", "epoch", "is_anchor"]

        self.validation_stats_df: pd.DataFrame = pd.DataFrame(columns=self.df_columns)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.vae(x)

    def step(self, batch, batch_index: int, stage: str) -> Mapping[str, Any]:
        image_batch = batch["image"]
        image_batch_recon, latent_mu, latent_logvar = self.vae(image_batch)

        loss = vae_loss(
            image_batch_recon,
            image_batch,
            latent_mu,
            latent_logvar,
            variational_beta=self.hparams.loss.variational_beta,
        )

        return {
            "loss": loss,
            "batch": batch,
            "image_batch_recon": image_batch_recon.detach(),
            "latent_mu": latent_mu.detach(),
            "latent_logvar": latent_logvar.detach(),
        }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        for output in outputs:
            self.validation_stats_df = pd.concat(
                [
                    self.validation_stats_df,
                    pd.DataFrame(
                        {
                            "image_index": output["batch"]["index"],
                            "class": output["batch"]["class"],
                            "target": output["batch"]["target"],
                            "mu_0": output["latent_mu"][:, 0],
                            "mu_1": output["latent_mu"][:, 1],
                            "std_0": output["latent_logvar"][:, 0],
                            "std_1": output["latent_logvar"][:, 1],
                            "epoch": [self.current_epoch] * len(output["batch"]["index"]),
                            "is_anchor": [False] * len(output["batch"]["index"]),
                        }
                    ),
                ],
                ignore_index=False,
            )
        anchors_recon, anchors_latent_mu, anchors_latent_std = self(self.metadata.anchors)
        self.validation_stats_df = pd.concat(
            [
                self.validation_stats_df,
                pd.DataFrame(
                    {
                        "image_index": self.metadata.anchors_idxs,
                        "class": self.metadata.anchors_classes,
                        "target": self.metadata.anchors_targets,
                        "mu_0": anchors_latent_mu[:, 0],
                        "mu_1": anchors_latent_mu[:, 1],
                        "std_0": anchors_latent_std[:, 0],
                        "std_1": anchors_latent_std[:, 1],
                        "epoch": [self.current_epoch] * len(self.metadata.anchors_idxs),
                        "is_anchor": [True] * len(self.metadata.anchors_idxs),
                    }
                ),
            ],
            ignore_index=False,
        )

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
            x="mu_0",
            y="mu_1",
            animation_frame="epoch",
            animation_group="image_index",
            category_orders={"class": self.metadata.class_to_idx.keys()},
            # size='std_0',  # TODO: fixme, plotly crashes with any column name to set the anchor size
            color="class",
            hover_name="image_index",
            facet_col="is_anchor",
            color_discrete_map=color_discrete_map,
            symbol="is_anchor",
            symbol_map={False: "circle", True: "star"},
            size_max=40,
            range_x=[-5, 5],
            color_continuous_scale=None,
            range_y=[-5, 5],
        )

        # Convert to HTML as a workaround to https://github.com/wandb/client/issues/2191
        self.logger.experiment.log(
            {"latent/val": wandb.Html(plotly.io.to_html(latent_val_fig), inject=False)}, step=self.global_step
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
