import logging
from typing import Any, Mapping

import torch.nn.functional as F

from rae.modules.enumerations import Output
from rae.pl_modules.pl_gae import LightningGAE
from rae.utils.tensor_ops import detach_tensors

pylogger = logging.getLogger(__name__)


class LightningDeterministic(LightningGAE):
    def step(self, batch, batch_index: int, stage: str) -> Mapping[str, Any]:
        out = super().step(batch, batch_index, stage)
        image_batch = batch["image"]

        loss = F.mse_loss(
            out[Output.RECONSTRUCTION],
            image_batch,
        )

        return {
            Output.LOSS: loss,
            Output.BATCH: batch,
            **{key: detach_tensors(value) for key, value in out.items()},
        }
