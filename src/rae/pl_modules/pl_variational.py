import logging
from typing import Any, Mapping

from rae.losses.vae_loss import vae_loss
from rae.modules.enumerations import Output
from rae.pl_modules.pl_gae import LightningGAE

pylogger = logging.getLogger(__name__)


class LightningVariational(LightningGAE):
    def step(self, batch, batch_index: int, stage: str) -> Mapping[str, Any]:
        out = super().step(batch, batch_index, stage)
        image_batch = batch["image"]

        loss = vae_loss(
            out[Output.RECONSTRUCTION],
            image_batch,
            out[Output.LATENT_MU],
            out[Output.LATENT_LOGVAR],
            variational_beta=self.hparams.loss.variational_beta,
        )
        loss = loss - out[Output.INV_LATENTS].clamp(max=0).sum() * 100

        return {
            Output.LOSS: loss,
            Output.BATCH: batch,
            **{key: self.normalize_output(value) for key, value in out.items()},
        }
