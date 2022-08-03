import logging

import hydra
import omegaconf
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

from rae.utils.plotting import plot_images

pylogger = logging.getLogger(__name__)


class CIFAR10Dataset(Dataset):
    def __init__(self, split: Split, **kwargs):
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}> ('{split}')")

        self.split: Split = split

        # example
        self.cifar = CIFAR10(
            kwargs["path"],
            train=split == "train",
            download=True,
            transform=kwargs["transform"],
        )

    @property
    def classes(self):
        return self.cifar.classes

    @property
    def targets(self):
        return self.cifar.targets

    @property
    def class_vocab(self):
        return self.cifar.class_to_idx

    def __len__(self) -> int:
        # example
        return len(self.cifar)

    def __getitem__(self, index: int):
        # example
        image, target = self.cifar[index]
        return {"index": index, "image": image, "target": target, "class": self.cifar.classes[target]}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.2")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    from torchvision.transforms import transforms

    dataset: Dataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train,
        split="train",
        path=PROJECT_ROOT / "data",
        transform=transforms.Compose([transforms.ToTensor()]),  # , transforms.Normalize((0.1307,), (0.3081,))]),
        _recursive_=False,
    )
    _ = dataset[0]

    for x in dataset:
        plot_images(x["image"][None], title="s").show()
        # break


if __name__ == "__main__":
    main()
