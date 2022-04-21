import logging
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(
        self,
        anchors_idxs: List[int],
        anchors: torch.Tensor,
        anchors_targets: torch.Tensor,
        anchors_classes: torch.Tensor,
        fixed_images_idxs: List[int],
        fixed_images: torch.Tensor,
        fixed_images_targets: torch.Tensor,
        fixed_images_classes: torch.Tensor,
        class_to_idx: Dict[str, int],
        idx_to_class: Dict[int, str],
    ):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.
        """
        self.class_to_idx: Dict[str, int] = class_to_idx
        self.idx_to_class: Dict[int, str] = idx_to_class
        self.anchors_idxs: List[int] = anchors_idxs
        self.anchors: torch.Tensor = anchors
        self.anchors_targets: torch.Tensor = anchors_targets
        self.anchors_classes: torch.Tensor = anchors_classes

        self.fixed_images_idxs: torch.Tensor = fixed_images_idxs
        self.fixed_images: torch.Tensor = fixed_images
        self.fixed_images_targets: torch.Tensor = fixed_images_targets
        self.fixed_images_classes: torch.Tensor = fixed_images_classes

    def __repr__(self):
        return f"MetaData(anchors_idxs={self.anchors_idxs}, ...)"

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        torch.save(self.class_to_idx, f=dst_path / "class_to_idx.pt")
        torch.save(self.idx_to_class, f=dst_path / "idx_to_class.pt")

        torch.save(self.anchors_idxs, f=dst_path / "anchors_idxs.pt")
        torch.save(self.anchors, f=dst_path / "anchors.pt")
        torch.save(self.anchors_targets, f=dst_path / "anchors_targets.pt")
        torch.save(self.anchors_classes, f=dst_path / "anchors_classes.pt")

        torch.save(self.fixed_images_idxs, f=dst_path / "fixed_images_idxs.pt")
        torch.save(self.fixed_images, f=dst_path / "fixed_images.pt")
        torch.save(self.fixed_images_targets, f=dst_path / "fixed_images_targets.pt")
        torch.save(self.fixed_images_classes, f=dst_path / "fixed_images_classes.pt")

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        class_to_idx = torch.load(f=src_path / "class_to_idx.pt")
        idx_to_class = torch.load(f=src_path / "idx_to_class.pt")

        anchors_idxs = torch.load(f=src_path / "anchors_idxs.pt")
        anchors = torch.load(f=src_path / "anchors.pt")
        anchors_targets = torch.load(f=src_path / "anchors_targets.pt")
        anchors_classes = torch.load(f=src_path / "anchors_classes.pt")

        fixed_images_idxs = torch.load(f=src_path / "fixed_images_idxs.pt")
        fixed_images = torch.load(f=src_path / "fixed_images.pt")
        fixed_images_targets = torch.load(f=src_path / "fixed_images_targets.pt")
        fixed_images_classes = torch.load(f=src_path / "fixed_images_classes.pt")

        return MetaData(
            anchors_idxs=anchors_idxs,
            anchors=anchors,
            anchors_targets=anchors_targets,
            anchors_classes=anchors_classes,
            fixed_images_idxs=fixed_images_idxs,
            fixed_images=fixed_images,
            fixed_images_targets=fixed_images_targets,
            fixed_images_classes=fixed_images_classes,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
        )


def collate_fn(samples: List, split: Split, metadata: MetaData):
    """Custom collate function for dataloaders with access to split and metadata.

    Args:
        samples: A list of samples coming from the Dataset to be merged into a batch
        split: The data split (e.g. train/val/test)
        metadata: The MetaData instance coming from the DataModule or the restored checkpoint

    Returns:
        A batch generated from the given samples
    """
    return default_collate(samples)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        anchors_idxs: List[int],
        val_images_fixed_idxs: List[int],
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.anchors_idxs: List[int] = anchors_idxs
        self.val_images_fixed_idxs: List[int] = val_images_fixed_idxs

    def extract_batch(self, dataset: Dataset, indices: Sequence[int]) -> Dict[str, Any]:
        images = []
        targets = []
        classes = []
        for index in indices:
            sample = dataset[index]
            images.append(sample["image"])
            targets.append(sample["target"])
            classes.append(sample["class"])

        images = torch.stack(images, dim=0)
        targets = torch.as_tensor(targets)

        return {
            "images": images,
            "targets": targets,
            "classes": classes,
        }

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        class_to_idx = self.train_dataset.mnist.class_to_idx
        idx_to_class = {x: y for y, x in class_to_idx.items()}

        anchors = self.extract_batch(self.train_dataset, self.anchors_idxs)
        fixed_images = self.extract_batch(self.val_datasets[0], self.val_images_fixed_idxs)

        return MetaData(
            anchors_idxs=self.anchors_idxs,
            anchors=anchors["images"],
            anchors_targets=anchors["targets"],
            anchors_classes=anchors["classes"],
            fixed_images_idxs=self.val_images_fixed_idxs,
            fixed_images=fixed_images["images"],
            fixed_images_targets=fixed_images["targets"],
            fixed_images_classes=fixed_images["classes"],
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
        )

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose([transforms.ToTensor()])  # , transforms.Normalize((0.1307,), (0.3081,))])

        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):
            # example
            self.train_dataset = hydra.utils.instantiate(
                self.datasets.train,
                split="train",
                transform=transform,
                path=PROJECT_ROOT / "data",
            )

            self.val_datasets = [
                hydra.utils.instantiate(
                    self.datasets.train,
                    split="test",
                    transform=transform,
                    path=PROJECT_ROOT / "data",
                )
            ]

        if stage is None or stage == "test":
            raise NotImplementedError()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="train", metadata=self.metadata),
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
                collate_fn=partial(collate_fn, split="val", metadata=self.metadata),
            )
            for dataset in self.val_datasets
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)


if __name__ == "__main__":
    main()
