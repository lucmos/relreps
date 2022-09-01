import math
from typing import Dict, List

import torch
from torch import Tensor, nn

from rae.modules.blocks import build_dynamic_encoder_decoder
from rae.modules.enumerations import Output


class VanillaCNN(nn.Module):
    def __init__(
        self,
        metadata,
        input_size,
        latent_dim: int,
        hidden_dims: List = None,
        **kwargs,
    ) -> None:
        """https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

        Args:
            in_channels:
            latent_dim:
            hidden_dims:
            **kwargs:
        """
        super().__init__()

        self.metadata = metadata
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.encoder, self.encoder_out_shape, _ = build_dynamic_encoder_decoder(
            width=metadata.width, height=metadata.height, n_channels=metadata.n_channels, hidden_dims=hidden_dims
        )
        encoder_out_numel = math.prod(self.encoder_out_shape[1:])

        self.encoder_out = nn.Sequential(
            nn.Linear(encoder_out_numel, latent_dim),
        )

        self.projection = nn.Sequential(
            nn.Linear(self.latent_dim, encoder_out_numel),
            nn.Tanh(),
            nn.Linear(
                encoder_out_numel,
                len(metadata.class_to_idx),
            ),
            nn.ReLU(),
        )

    def encode(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.encoder_out(result)
        return result

    def forward(self, x: Tensor, **kwargs) -> Dict[Output, Tensor]:
        latent = self.encode(x)
        logits = self.projection(latent)
        return {
            Output.LOGITS: logits,
            Output.DEFAULT_LATENT: latent,
            Output.BATCH_LATENT: latent,
        }

    def set_finetune_mode(self) -> None:
        pass
