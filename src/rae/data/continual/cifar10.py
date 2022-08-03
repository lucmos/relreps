import logging
import random
from bisect import bisect
from typing import Iterable

import hydra
import numpy as np
import omegaconf
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CIFAR10

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

pylogger = logging.getLogger(__name__)


class ContinualCIFAR10Dataset(Dataset):
    def __init__(
        self,
        split: Split,
        tasks_epochs: Iterable[int],
        tasks_progression: Iterable[Iterable[int]],
        datamodule: LightningDataModule,
        **kwargs,
    ):
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}> ('{split}')")

        self.split: Split = split
        self.tasks_epochs = tasks_epochs
        self._task_epochs_cumsum = np.asarray(tasks_epochs).cumsum()
        self.tasks_progression = tasks_progression
        self.datamodule = datamodule

        # example
        self.cifar = CIFAR10(
            kwargs["path"],
            train=split == "train",
            download=True,
            transform=kwargs["transform"],
        )

        if len(tasks_epochs) != len(tasks_progression):
            raise ValueError("Each tasks should be associated with its epochs number and targets")

        if not all(isinstance(task, Iterable) for task in tasks_progression):
            raise ValueError("Each task in the task progression should be an iterable!")

        for task in tasks_progression:
            if len(task) != len(set(task)):
                raise ValueError(f"A task cannot contain repeated indices! Found: {task}")

        self.task_idx2samples_indices = {
            task_idx: np.flatnonzero(np.isin(np.asarray(self.cifar.targets), tasks_progression[task_idx]))
            for task_idx in range(len(tasks_progression))
        }
        self.task_idx2dataset = {
            task: Subset(self.cifar, indices=indices) for task, indices in self.task_idx2samples_indices.items()
        }

    @property
    def current_task(self):
        return min(
            bisect(self._task_epochs_cumsum, self.datamodule.trainer.current_epoch),
            len(self.tasks_progression) - 1,
        )

    @property
    def classes(self):
        return self.cifar.classes

    @property
    def targets(self):
        if self.split == "train":
            return np.asarray(self.cifar.targets)[self.task_idx2samples_indices[self.current_task]].tolist()
        else:
            return self.cifar.targets

    @property
    def class_vocab(self):
        return self.cifar.class_to_idx

    def __len__(self) -> int:
        if self.split == "train":
            return len(self.task_idx2dataset[self.current_task])
        else:
            return len(self.cifar)

    def __getitem__(self, index: int):
        if self.split == "train":
            dataset = self.task_idx2dataset[self.current_task]
        else:
            dataset = self.cifar

        # example
        image, target = dataset[index]
        sample = {
            "index": index,
            "image": image,
            "target": target,
            "class": self.cifar.classes[target],
        }
        if self.split == "train":
            sample["task"] = self.current_task
        return sample

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.split=}, {self.tasks_epochs=}, {self.tasks_progression=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.2")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    from torchvision.transforms import transforms

    def fake_datamodule():
        pass

    datamodule = fake_datamodule
    datamodule.trainer = lambda x: x
    datamodule.trainer.current_epoch = 0

    dataset: Dataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train,
        split="train",
        datamodule=datamodule,
        path=PROJECT_ROOT / "data",
        transform=transforms.Compose([transforms.ToTensor()]),  # , transforms.Normalize((0.1307,), (0.3081,))]),
        _recursive_=False,
    )
    _ = dataset[0]

    for epoch in range(110):
        datamodule.trainer.current_epoch = epoch

        print(f"TASK: {dataset.current_task}")
        for step in range(3):
            i = random.randint(0, len(dataset))
            x = dataset[i]

            print(x["target"])


if __name__ == "__main__":
    main()
