import math
from typing import Dict, List, Optional

import hydra
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rae.modules.attention import AbstractRelativeAttention, AttentionOutput
from rae.modules.blocks import build_dynamic_encoder_decoder
from rae.modules.enumerations import Output


class VanillaRelAE(nn.Module):
    def __init__(
        self,
        metadata,
        input_size,
        latent_dim: int,
        relative_attention: AbstractRelativeAttention,
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

        self.encoder, self.encoder_out_shape, self.decoder = build_dynamic_encoder_decoder(
            width=metadata.width, height=metadata.height, n_channels=metadata.n_channels, hidden_dims=hidden_dims
        )
        encoder_out_numel = math.prod(self.encoder_out_shape[1:])

        self.relative_attention: AbstractRelativeAttention = (
            hydra.utils.instantiate(
                relative_attention,
                n_anchors=self.metadata.anchors_targets.size(0),
                n_classes=len(self.metadata.class_to_idx),
                _recursive_=False,
            )
            if not isinstance(relative_attention, AbstractRelativeAttention)
            else relative_attention
        )

        # Build Decoder
        self.decoder_input = nn.Linear(self.relative_attention.output_dim, encoder_out_numel)

        # TODO: these buffers are duplicated in the pl_gclassifier. Remove one of the two.
        self.register_buffer("anchors_images", metadata.anchors_images)
        self.register_buffer("anchors_latents", metadata.anchors_latents)
        self.register_buffer("anchors_targets", metadata.anchors_targets)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        return result

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, *self.encoder_out_shape[1:])
        result = self.decoder(result)
        return result

    def forward(
        self,
        x: Tensor,
        new_anchors_images: Optional[torch.Tensor] = None,
        new_anchors_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[Output, Tensor]:
        x_embedded = self.encode(x)

        with torch.no_grad():
            anchors_embedded = self.encode(self.anchors_images if new_anchors_images is None else new_anchors_images)

        attention_output = self.relative_attention(
            x=x_embedded,
            anchors=anchors_embedded,
            anchors_targets=self.anchors_targets if new_anchors_targets is None else new_anchors_targets,
        )

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        x_recon = self.decode(attention_output[AttentionOutput.OUTPUT])

        return {
            Output.RECONSTRUCTION: x_recon,
            Output.DEFAULT_LATENT: x_embedded,
            Output.BATCH_LATENT: x_embedded,
            Output.ANCHORS_LATENT: anchors_embedded,
            Output.INV_LATENTS: attention_output[AttentionOutput.OUTPUT],
        }

    def loss_function(self, model_out, batch, *args, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = model_out[Output.RECONSTRUCTION]
        input = batch["image"]

        recons_loss = F.mse_loss(recons, input, reduction="mean")

        loss = recons_loss
        return {"loss": loss, "reconstruction": recons_loss.detach()}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples, self.latent_dim)

        z = z.to(current_device)
        ...  # call attention
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        ...  # call attention

        return self.forward(x)[0]
