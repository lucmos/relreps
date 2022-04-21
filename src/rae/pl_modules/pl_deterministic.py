import logging
from typing import Any, Mapping

import torch.nn.functional as F

from rae.modules.output_keys import Output
from rae.pl_modules.pl_gae import LightningGAE

pylogger = logging.getLogger(__name__)


class LightningDeterministic(LightningGAE):
    def step(self, batch, batch_index: int, stage: str) -> Mapping[str, Any]:
        image_batch = batch["image"]
        out = self(image_batch)
        image_batch_recon = out[Output.OUT]
        default_latent = out[out[Output.DEFAULT_LATENT]]

        loss = F.mse_loss(
            image_batch_recon,
            image_batch,
        )

        return {
            "loss": loss,
            "batch": batch,
            "image_batch_recon": image_batch_recon.detach(),
            "default_latent": default_latent.detach(),
        }
