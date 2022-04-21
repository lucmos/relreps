import logging
from typing import Any, Mapping

from rae.losses.vae_loss import vae_loss
from rae.modules.output_keys import Output
from rae.pl_modules.pl_gae import LightningGAE

pylogger = logging.getLogger(__name__)


class LightningVariational(LightningGAE):
    def step(self, batch, batch_index: int, stage: str) -> Mapping[str, Any]:
        out = super().step(batch, batch_index, stage)
        image_batch = batch["image"]

        image_batch_recon, latent_mu, latent_logvar = out[Output.OUT], out[Output.LATENT_MU], out[Output.LATENT_LOGVAR]
        default_latent = out[out[Output.DEFAULT_LATENT]]

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
            "default_latent": default_latent.detach(),
            "latent_mu": latent_mu.detach(),
            "latent_logvar": latent_logvar.detach(),
        }
