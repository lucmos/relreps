import logging
from typing import Any, Dict, List, Mapping, Optional, Set

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torchmetrics

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from rae.data.datamodule import MetaData
from rae.modules.enumerations import Output, Stage, SupportedViz
from rae.pl_modules.pl_abstract_module import AbstractLightningModule
from rae.pl_modules.pl_visualizations import on_fit_end_viz, on_fit_start_viz, validation_epoch_end_viz
from rae.utils.tensor_ops import detach_tensors

pylogger = logging.getLogger(__name__)


class LightningContinualClassifier(AbstractLightningModule):
    logger: NNLogger

    def __init__(self, metadata: Optional[MetaData] = None, *args, **kwargs) -> None:
        super().__init__(metadata, *args, **kwargs)

        self.model = hydra.utils.instantiate(
            kwargs["model"],
            metadata=metadata,
            input_channels=metadata.anchors_images.shape[1],
            n_classes=len(metadata.class_to_idx),
        )
        self.loss = hydra.utils.instantiate(self.hparams.loss)

        self.register_buffer("anchors_images", self.metadata.anchors_images)
        self.register_buffer("anchors_latents", self.metadata.anchors_latents)
        self.register_buffer("anchors_targets", self.metadata.anchors_targets)
        self.register_buffer("fixed_images", self.metadata.fixed_images)

        self.replay_buffer = hydra.utils.instantiate(
            kwargs["replay"],
            metadata=metadata,
            module=self,
        )

        self.memory_loss = hydra.utils.instantiate(
            kwargs["memory"],
            metadata=metadata,
            module=self,
        )
        self.targets_seen_in_epoch = None
        self.learned_targets = None

        micro_metric = torchmetrics.Accuracy(num_classes=len(metadata.class_to_idx))
        # FIXME: workaround to avoid lightnign error of missing attribute
        self.micro_train_accuracy = micro_metric.clone()
        self.micro_validation_accuracy = micro_metric.clone()
        self.micro_accuracies = {
            Stage.TRAIN: self.micro_train_accuracy,
            Stage.VAL: self.micro_validation_accuracy,
        }

        macro_metric = torchmetrics.Accuracy(num_classes=len(metadata.class_to_idx), average="none")
        # FIXME: workaround to avoid lightnign error of missing attribute
        self.macro_train_accuracy = macro_metric.clone()
        self.macro_validation_accuracy = macro_metric.clone()
        self.macro_accuracies = {
            Stage.TRAIN: self.macro_train_accuracy,
            Stage.VAL: self.macro_validation_accuracy,
        }

        self.supported_viz = self.supported_viz()
        pylogger.info(f"Enabled visualizations: {str(sorted(x.value for x in self.supported_viz))}")

    def supported_viz(self) -> Set[SupportedViz]:
        supported_viz = set()

        if self.fixed_images is not None:
            supported_viz.add(SupportedViz.VALIDATION_IMAGES_SOURCE)

        if self.anchors_images is not None:
            supported_viz.add(SupportedViz.ANCHORS_SOURCE)

        supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT)
        supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT)
        supported_viz.add(SupportedViz.ANCHORS_SELF_INNER_PRODUCT_NORMALIZED)
        supported_viz.add(SupportedViz.ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT_NORMALIZED)
        return supported_viz

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        # example
        return self.model(x, **kwargs)

    def step(self, batch, batch_index: int, stage: Stage) -> Mapping[str, Any]:
        image_batch = batch["image"]
        out = self(image_batch)

        out_anchors = self(self.anchors_images)

        classification_loss = self.loss(out[Output.LOGITS], batch["target"])
        mem_loss = self.memory_loss.compute(
            out_anchors[Output.DEFAULT_LATENT],
            self.anchors_targets,
            targets_to_consider=self.learned_targets,
        )

        loss = classification_loss + mem_loss

        probs = torch.softmax(out[Output.LOGITS], dim=-1)
        self.micro_accuracies[stage].update(probs, batch["target"])
        self.macro_accuracies[stage].update(probs, batch["target"])
        self.log_dict(
            {
                "task": float(self.trainer.datamodule.train_dataset.current_task) if self.trainer is not None else None,
                f"loss/{stage}": detach_tensors(loss),
                f"loss/{stage}/mem_loss": detach_tensors(mem_loss),
                f"loss/{stage}/classification_loss": detach_tensors(classification_loss),
                f"acc/{stage}": detach_tensors(self.micro_accuracies[stage].compute()),
            },
            on_step=stage == Stage.TRAIN,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["image"].shape[0],
        )
        self.log_dict(
            {
                f"acc/{stage}/class{class_idx}": class_acc
                for class_idx, class_acc in enumerate(self.macro_accuracies[stage].compute().cpu().tolist())
            },
            on_step=stage == Stage.TRAIN,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch["image"].shape[0],
        )

        return {
            Output.LOSS: loss,
            Output.BATCH: {key: detach_tensors(value) for key, value in batch.items()},
            **{key: detach_tensors(value) for key, value in out.items()},
            Output.ANCHORS_OUT: {key: detach_tensors(value) for key, value in out_anchors.items()},
        }

    def on_epoch_start(self) -> None:
        self.model.set_finetune_mode()

    def on_fit_start(self) -> None:
        on_fit_start_viz(lightning_module=self, fixed_images=self.fixed_images, anchors_images=self.anchors_images)

    def on_fit_end(self) -> None:
        on_fit_end_viz(lightning_module=self, validation_stats_df=None)

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        batch = self.replay_buffer(batch)
        return self.step(batch, batch_idx, stage=Stage.TRAIN)

    def training_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        self.micro_accuracies[Stage.TRAIN].reset()
        self.macro_accuracies[Stage.TRAIN].reset()

        self.targets_seen_in_epoch = torch.cat([output[Output.BATCH]["target"] for output in outputs]).unique()
        self.memory_loss.update(
            outputs[-1][Output.ANCHORS_OUT][Output.DEFAULT_LATENT],
            self.anchors_targets,
            targets_to_consider=self.targets_seen_in_epoch,
        )
        if self.learned_targets is None:
            self.learned_targets = self.targets_seen_in_epoch
        else:
            self.learned_targets = torch.cat([self.learned_targets, self.targets_seen_in_epoch]).unique()

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self.step(batch, batch_idx, stage=Stage.VAL)

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]) -> None:
        if self.trainer.sanity_checking:
            return

        self.micro_accuracies[Stage.VAL].reset()
        self.macro_accuracies[Stage.VAL].reset()

        if self.anchors_images is not None:
            anchors_out = self(self.anchors_images)
            if Output.ANCHORS_LATENT in anchors_out:
                anchors_latents = anchors_out[Output.ANCHORS_LATENT]
            else:
                anchors_latents = anchors_out[Output.DEFAULT_LATENT]
        else:
            raise NotImplementedError()

        validation_epoch_end_viz(
            lightning_module=self,
            outputs=outputs,
            validation_stats_df=None,
            anchors_reconstructed=None,
            anchors_latents=anchors_latents,
            fixed_images_out=self(self.fixed_images),
        )


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Lightning Module.

    Args:
        cfg: the hydra configuration
    """
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    _: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.module,
        metadata=datamodule.metadata,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
