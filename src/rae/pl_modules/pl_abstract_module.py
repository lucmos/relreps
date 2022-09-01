import abc
import logging
from typing import Any, Dict, Optional, Sequence, Set, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
from sklearn.decomposition import PCA
from torch.optim import Optimizer

from nn_core.model_logging import NNLogger

from rae.data.vision.datamodule import MetaData
from rae.modules.enumerations import SupportedViz

pylogger = logging.getLogger(__name__)


class AbstractLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata

        self.validation_pca: Optional[PCA] = None

    @abc.abstractmethod
    def encode(self, *args, **kwargs):
        raise NotImplementedError

    # @property
    # def encode_output(self) -> Set[str]:
    #     raise NotImplementedError
    #
    # @property
    # def decode_input(self) -> Set[str]:
    #     raise NotImplementedError

    @abc.abstractmethod
    def decode(self, *args, **kwargs):
        raise NotImplementedError

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint["validation_pca"] = self.validation_pca

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if "validation_pca" in checkpoint:
            self.validation_pca = checkpoint["validation_pca"]
        else:
            self.validation_pca = PCA(n_components=2)

    def fit_pca(self, latents: torch.Tensor) -> None:
        if self.validation_pca is None or self.hparams.fit_pca_each_epoch:
            self.validation_pca = PCA(n_components=2)
            self.validation_pca.fit(latents)

    @abc.abstractmethod
    def supported_viz(self) -> Set[SupportedViz]:
        raise NotImplementedError

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]
