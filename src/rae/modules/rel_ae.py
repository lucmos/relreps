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
        activation: str = "torch.nn.GELU",
        remove_encoder_last_activation: bool = False,
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
            width=metadata.width,
            height=metadata.height,
            n_channels=metadata.n_channels,
            hidden_dims=hidden_dims,
            activation=activation,
            remove_encoder_last_activation=remove_encoder_last_activation,
        )
        encoder_out_numel = math.prod(self.encoder_out_shape[1:])

        # self.encoder_out = nn.Sequential(
        #     nn.Linear(encoder_out_numel, latent_dim),
        # )

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
        self.decoder_in = nn.Sequential(
            nn.Linear(
                self.relative_attention.output_dim,
                encoder_out_numel,
            ),
            hydra.utils.instantiate({"_target_": activation}),
        )

        # TODO: these buffers are duplicated in the pl_gclassifier. Remove one of the two.
        self.register_buffer("anchors_images", metadata.anchors_images)
        self.register_buffer("anchors_latents", metadata.anchors_latents)
        self.register_buffer("anchors_targets", metadata.anchors_targets)

    def embed(self, input: Tensor) -> Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # result = self.encoder_out(result)
        return result

    def decode(self, **kwargs) -> Dict[str, Tensor]:

        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        attention_output = self.relative_attention.decode(**kwargs)

        result = self.decoder_in(attention_output[AttentionOutput.OUTPUT])
        result = result.view(-1, *self.encoder_out_shape[1:])
        result = self.decoder(result)
        return {
            Output.RECONSTRUCTION: result,
            Output.DEFAULT_LATENT: attention_output[Output.BATCH_LATENT],
            Output.BATCH_LATENT: attention_output[Output.BATCH_LATENT],
            Output.ANCHORS_LATENT: attention_output[Output.ANCHORS_LATENT],
            Output.INV_LATENTS: attention_output[AttentionOutput.OUTPUT],
            Output.SIMILARITIES: attention_output[AttentionOutput.SIMILARITIES],
            Output.NON_QUANTIZED_SIMILARITIES: attention_output[AttentionOutput.NON_QUANTIZED_SIMILARITIES],
        }

    def encode(
        self,
        x: Tensor,
        new_anchors_images: Optional[torch.Tensor] = None,
        new_anchors_targets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Tensor]:
        x_embedded = self.embed(x)

        with torch.no_grad():
            anchors_embedded = self.embed(self.anchors_images if new_anchors_images is None else new_anchors_images)

        attention_encoding = self.relative_attention.encode(
            x=x_embedded,
            anchors=anchors_embedded,
            anchors_targets=self.anchors_targets if new_anchors_targets is None else new_anchors_targets,
        )
        return {
            **attention_encoding,
            Output.BATCH_LATENT: x_embedded,
            Output.ANCHORS_LATENT: anchors_embedded,
        }

    def loss_function(self, model_out, batch, *args, **kwargs) -> dict:
        """https://stackoverflow.com/questions/64909658/what-could-cause-a-vaevariational-autoencoder-to-output-random-noise-even-afte

        Computes the VAE loss function.
        'KL(N(mu, sigma), N(0, 1)) = log frac{1}{sigma} + frac{sigma^2 + mu^2}{2} - frac{1}{2}'
        :param args:
        :param kwargs:
        :return:
        """
        predictions = model_out[Output.RECONSTRUCTION]
        targets = batch["image"]
        mse = F.mse_loss(predictions, targets, reduction="mean")
        log_sigma_opt = 0.5 * mse.log()
        r_loss = 0.5 * torch.pow((targets - predictions) / log_sigma_opt.exp(), 2) + log_sigma_opt
        r_loss = r_loss.sum()
        loss = r_loss
        return {
            "loss": loss,
            "reconstruction": r_loss.detach() / targets.shape[0],
        }

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
