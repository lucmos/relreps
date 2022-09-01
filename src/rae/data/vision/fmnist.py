import logging

import hydra
import omegaconf
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import transforms

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

pylogger = logging.getLogger(__name__)


class FashionMNISTDataset(Dataset):
    def __init__(self, split: Split, **kwargs):
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}> ('{split}')")

        self.split: Split = split

        # example
        self.mnist = FashionMNIST(
            kwargs["path"],
            train=split == "train",
            download=True,
            transform=kwargs["transform"],
        )

    @property
    def targets(self):
        return self.mnist.targets

    @property
    def class_vocab(self):
        return self.mnist.class_to_idx

    def __len__(self) -> int:
        # example
        return len(self.mnist)

    def __getitem__(self, index: int):
        # example
        image, target = self.mnist[index]
        return {"index": index, "image": image, "target": target, "class": self.mnist.classes[target]}

    def __repr__(self) -> str:
        return f"MNIST({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
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
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        _recursive_=False,
    )
    _ = dataset[0]


if __name__ == "__main__":
    main()
