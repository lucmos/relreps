from collections import deque
from typing import Set

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from nn_core.common import PROJECT_ROOT

from rae.data.datamodule import MetaData
from rae.pl_modules.pl_abstract_module import AbstractLightningModule


class StaticMemoryLoss:
    def __init__(
        self,
        metadata: MetaData,
        module: AbstractLightningModule,
        running_average_n: int,
        start_epoch: int,
        limit_target_representation: bool,
        loss_weight: float,
    ):
        super().__init__()
        self.metadata = metadata
        self.module = module
        self.running_average_n = running_average_n
        self.start_epoch = start_epoch
        self.limit_target_representation = limit_target_representation
        self.loss_weight = loss_weight
        self.targets = None
        self.index2latent = None

    def update(
        self,
        image_latents: torch.Tensor,
        image_targets: torch.Tensor,
        targets_to_consider: torch.Tensor,
    ) -> None:
        self.targets = image_targets
        if self.index2latent is None:
            self.index2latent = [deque([latent], maxlen=self.running_average_n) for latent in image_latents]

        for i, (latent, target) in enumerate(zip(image_latents, image_targets)):
            if target not in targets_to_consider:
                continue
            self.index2latent[i].append(latent)

    # TODO: take into account the anchors target in the image latents to compute the loss only according to the known targets!
    # TODO: if using fancy rel attention parameters, the anchors targets may be different! Still need to implement the
    # TODO: targets update for those operations (stratified mean, sampling, etc.)
    def compute(
        self, image_latents: torch.Tensor, image_targets: torch.Tensor, targets_to_consider: torch.Tensor
    ) -> torch.Tensor:
        if (
            targets_to_consider is None
            or targets_to_consider.size() == 0
            or self.targets is None
            or self.index2latent is None
            or self.start_epoch > self.module.current_epoch
        ):
            return 0
        memorized_latents = torch.stack(
            [torch.stack(tuple(self.index2latent[i])).mean(0) for i in range(len(self.index2latent))]
        )
        target_masks = torch.isin(image_targets, targets_to_consider)

        selected_image_latents, selected_memorized_latents = (
            image_latents[target_masks, :],
            memorized_latents[target_masks, :],
        )

        if self.limit_target_representation:
            selected_image_latents, selected_memorized_latents = (
                selected_image_latents[:, target_masks],
                selected_memorized_latents[:, target_masks],
            )

        return F.mse_loss(selected_image_latents, selected_memorized_latents) * self.loss_weight


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.2")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    m: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    m.setup()
    m.metadata
    StaticMemoryLoss(metadata=m.metadata)


if __name__ == "__main__":
    main()
