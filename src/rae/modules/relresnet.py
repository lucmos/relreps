import logging
from typing import Dict, Optional

import torch
from torch import nn

from rae.data.datamodule import MetaData
from rae.modules.attention import RelativeTransformerBlock
from rae.modules.enumerations import (
    AttentionOutput,
    NormalizationMode,
    Output,
    RelativeEmbeddingMethod,
    SimilaritiesQuantizationMode,
    ValuesMethod,
)
from rae.modules.passthrough import PassThrough
from rae.utils.tensor_ops import freeze, get_resnet_model

pylogger = logging.getLogger(__name__)


class RelResNet(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        use_pretrained: bool,
        finetune: bool,
        hidden_features: int,
        dropout_p: float,
        normalization_mode: NormalizationMode,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
        similarities_quantization_mode: Optional[SimilaritiesQuantizationMode] = None,
        similarities_bin_size: Optional[float] = None,
        resnet_size: int = 18,
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

        self.relative_attention_block = RelativeTransformerBlock(
            in_features=self.resnet_features,
            hidden_features=hidden_features,
            dropout_p=dropout_p,
            out_features=hidden_features,
            n_anchors=metadata.anchors_images.shape[0],
            normalization_mode=normalization_mode,
            similarity_mode=similarity_mode,
            values_mode=values_mode,
            similarities_quantization_mode=similarities_quantization_mode,
            similarities_bin_size=similarities_bin_size,
        )

        self.final_layer = nn.Linear(in_features=hidden_features, out_features=len(self.metadata.class_to_idx))

        self.register_buffer("anchors_images", metadata.anchors_images)
        self.register_buffer("anchors_latents", metadata.anchors_latents)

    def set_finetune_mode(self) -> None:
        if not self.finetune:
            self.resnet.eval()

    def forward(self, x: torch.Tensor, new_anchors_images: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            anchors_latents = self.resnet(self.anchors_images if new_anchors_images is None else new_anchors_images)

        batch_latents = self.resnet(x)

        attention_output = self.relative_attention_block(x=batch_latents, anchors=anchors_latents)

        output = self.final_layer(attention_output[AttentionOutput.OUTPUT])

        return {
            Output.LOGITS: output,
            Output.DEFAULT_LATENT: batch_latents,
            Output.BATCH_LATENT: batch_latents,
            Output.ANCHORS_LATENT: anchors_latents,
            Output.INV_LATENTS: attention_output[AttentionOutput.SIMILARITIES],
        }
