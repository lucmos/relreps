import math
from typing import Dict, List

import hydra.utils
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rae.modules.blocks import build_dynamic_encoder_decoder
from rae.modules.enumerations import Output


class VanillaAE(nn.Module):
    def __init__(
        self,
        metadata,
        input_size,
        latent_dim: int,
        hidden_dims: List = None,
        latent_activation: str = "torch.nn.GELU",
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

        self.encoder_out = nn.Sequential(
            nn.Linear(encoder_out_numel, latent_dim),
            hydra.utils.instantiate({"_target_": latent_activation})
            if latent_activation is not None
            else nn.Identity(),
        )

        self.decoder_in = nn.Sequential(
            nn.Linear(
                self.latent_dim,
                encoder_out_numel,
            ),
            hydra.utils.instantiate({"_target_": latent_activation})
            if latent_activation is not None
            else nn.Identity(),
        )

    def encode(self, input: Tensor) -> Dict[str, Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.encoder_out(result)
        return {
            Output.BATCH_LATENT: result,
        }

    def decode(self, batch_latent: Tensor) -> Dict[Output, Tensor]:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_in(batch_latent)
        result = result.view(-1, *self.encoder_out_shape[1:])
        result = self.decoder(result)
        return {
            Output.RECONSTRUCTION: result,
            Output.DEFAULT_LATENT: batch_latent,
            Output.BATCH_LATENT: batch_latent,
        }

    def loss_function(self, model_out, batch, *args, **kwargs) -> dict:
        """https://stackoverflow.com/questions/64909658/what-could-cause-a-vaevariational-autoencoder-to-output-random-noise-even-afte

        Computes the VAE loss function.
        KL(N(mu, sigma), N(0, 1)) = log frac{1}{sigma} + frac{sigma^2 + mu^2}{2} - frac{1}{2}
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

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
