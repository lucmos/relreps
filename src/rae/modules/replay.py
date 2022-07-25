import random
from typing import Dict, Set

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, default_collate, default_convert
from torch.utils.data.dataset import Dataset

from nn_core.common import PROJECT_ROOT

from rae.data.datamodule import MetaData
from rae.pl_modules.pl_abstract_module import AbstractLightningModule


class ReplayBuffer(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        module: AbstractLightningModule,
        max_size: int = 50,
        substitute_p: float = 0.5,
        anchors_p: float = 0.5,
        batch_keys: Set[str] = ("index", "image", "target", "class"),
    ) -> None:
        """Image buffer to lessen the catastrophic forgetting by replaying examples.

        All the anchors are always in the replay buffer by default
        """
        super().__init__()
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.substitute_p = substitute_p
        self.anchors_p = anchors_p

        self.batch_keys = batch_keys

        self.anchors = [
            {
                "index": default_convert(metadata.anchors_idxs[i]),
                "image": module.anchors_images[i],
                "target": default_convert(metadata.anchors_targets[i]).to(module.device),
                "class": default_convert(metadata.anchors_classes[i]),
            }
            for i in range(metadata.anchors_images.shape[0])
        ]
        self.buffer = []

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        to_return = []
        images = batch["image"]

        for i in range(images.shape[0]):
            element = {key: batch[key][i] for key in self.batch_keys}

            if len(self.buffer) < self.max_size:
                self.buffer.append(element)

            if random.uniform(a=0, b=1) > self.substitute_p:
                if random.uniform(a=0, b=1) > self.anchors_p:
                    for key in self.batch_keys:
                        i = random.randint(0, len(self.anchors) - 1)
                        element[key] = self.anchors[i][key]
                else:
                    for key in self.batch_keys:
                        i = random.randint(0, len(self.buffer) - 1)
                        self.buffer[i][key], element[key] = element[key], self.buffer[i][key]

            to_return.append(element)

        return default_collate(to_return)
