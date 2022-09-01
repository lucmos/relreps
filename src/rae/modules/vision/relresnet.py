import logging
from typing import Dict, Optional

import hydra.utils
import torch
from torch import nn

from rae.data.vision.datamodule import MetaData
from rae.modules.attention import AbstractRelativeAttention, AttentionOutput
from rae.modules.enumerations import Output
from rae.modules.passthrough import PassThrough
from rae.utils.tensor_ops import freeze, get_resnet_model

pylogger = logging.getLogger(__name__)


class RelResNet(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        resnet_size: int,
        hidden_features: int,
        use_pretrained: bool,
        finetune: bool,
        relative_attention: AbstractRelativeAttention,
        **kwargs,
    ) -> None:
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata

        self.finetune = finetune
        self.resnet, self.resnet_features = get_resnet_model(resnet_size=resnet_size, use_pretrained=use_pretrained)
        self.resnet.fc = PassThrough()
        if not finetune:
            freeze(self.resnet)

        self.resnet_post_fc = nn.Sequential(
            nn.Linear(in_features=self.resnet_features, out_features=self.resnet_features),
            nn.BatchNorm1d(num_features=self.resnet_features),
            nn.Tanh(),
            nn.Linear(in_features=self.resnet_features, out_features=hidden_features),
            nn.Tanh(),
        )

        self.relative_attention = (
            hydra.utils.instantiate(
                relative_attention,
                n_anchors=self.metadata.anchors_targets.shape[0],
                n_classes=len(self.metadata.class_to_idx),
            )
            if not isinstance(relative_attention, AbstractRelativeAttention)
            else relative_attention
        )

        self.final_layer = nn.Linear(
            in_features=self.relative_attention.output_dim, out_features=len(self.metadata.class_to_idx)
        )

        # TODO: these buffers are duplicated in the pl_gclassifier. Remove one of the two.
        self.register_buffer("anchors_images", metadata.anchors_images)
        self.register_buffer("anchors_latents", metadata.anchors_latents)
        self.register_buffer("anchors_targets", metadata.anchors_targets)

    def set_finetune_mode(self) -> None:
        if not self.finetune:
            self.resnet.eval()

    def forward(
        self,
        x: torch.Tensor,
        new_anchors_images: Optional[torch.Tensor] = None,
        new_anchors_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            anchors_latents = self.resnet(self.anchors_images if new_anchors_images is None else new_anchors_images)
            anchors_latents = self.resnet_post_fc(anchors_latents)

        batch_latents = self.resnet(x)
        batch_latents = self.resnet_post_fc(batch_latents)

        attention_output = self.relative_attention(
            x=batch_latents,
            anchors=anchors_latents,
            anchors_targets=self.anchors_targets if new_anchors_targets is None else new_anchors_targets,
        )

        output = self.final_layer(attention_output[AttentionOutput.OUTPUT])

        return {
            Output.LOGITS: output,
            Output.DEFAULT_LATENT: batch_latents,
            Output.BATCH_LATENT: batch_latents,
            Output.ANCHORS_LATENT: anchors_latents,
            Output.INV_LATENTS: attention_output[AttentionOutput.SIMILARITIES],
        }
