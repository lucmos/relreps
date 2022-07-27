import random
from typing import Dict, Set

import torch
from torch.utils.data import default_collate, default_convert

from rae.data.datamodule import MetaData
from rae.pl_modules.pl_abstract_module import AbstractLightningModule


class ReplayBuffer:
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
        self.metadata = metadata
        self.module = module

        self.max_size = max_size
        self.substitute_p = substitute_p
        self.anchors_p = anchors_p

        self.batch_keys = batch_keys

        # TODO: duplicated anchors storing, present both in model and module
        self.anchors = [
            {
                "index": default_convert(metadata.anchors_idxs[i]),
                "image": module.anchors_images[i],
                "target": default_convert(metadata.anchors_targets[i]),
                "class": default_convert(metadata.anchors_classes[i]),
            }
            for i in range(metadata.anchors_images.shape[0])
        ]
        self.buffer = []

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        to_return = []

        for i in range(batch["image"].shape[0]):
            element = {key: batch[key][i] for key in self.batch_keys}
            element["replay"] = False

            if len(self.buffer) < self.max_size:
                self.buffer.append(element)

            if random.uniform(a=0, b=1) > self.substitute_p:
                if random.uniform(a=0, b=1) > self.anchors_p:
                    element["replay"] = True
                    i = random.randint(0, len(self.anchors) - 1)
                    for key in self.batch_keys:
                        element[key] = self.anchors[i][key]
                        if isinstance(element[key], torch.Tensor):
                            element[key] = element[key].to(self.module.device)
                elif len(self.buffer):
                    element["replay"] = True
                    i = random.randint(0, len(self.buffer) - 1)
                    for key in self.batch_keys:
                        self.buffer[i][key], element[key] = element[key], self.buffer[i][key]

            to_return.append(element)

        return default_collate(to_return)
