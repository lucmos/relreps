import math
from typing import Dict, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rae.modules.blocks import build_dynamic_encoder_decoder
from rae.modules.enumerations import Output


class VanillaVAE(nn.Module):
    def __init__(
        self,
        metadata,
        input_size,
        latent_dim: int,
        hidden_dims: List = None,
        calibrated_loss: bool = True,
        kld_weight: float = 1,
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
        self.calibrated_loss = calibrated_loss
        self.kld_weight = kld_weight

        self.encoder, self.encoder_out_shape, self.decoder = build_dynamic_encoder_decoder(
            width=metadata.width,
            height=metadata.height,
            n_channels=metadata.n_channels,
            hidden_dims=hidden_dims,
        )
        encoder_out_numel = math.prod(self.encoder_out_shape[1:])

        self.fc_mu = nn.Linear(encoder_out_numel, latent_dim)
        self.fc_var = nn.Linear(encoder_out_numel, latent_dim)

        self.decoder_in = nn.Sequential(
            nn.Linear(
                self.latent_dim,
                encoder_out_numel,
            ),
            nn.GELU(),
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

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        z = self.reparameterize(mu, log_var)
        return {
            Output.BATCH_LATENT: z,
            Output.LATENT_MU: mu,
            Output.LATENT_LOGVAR: log_var,
        }

    def decode(self, batch_latent: Tensor, **kwargs) -> Dict[Output, Tensor]:
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
            Output.LATENT_MU: kwargs[Output.LATENT_MU],
            Output.LATENT_LOGVAR: kwargs[Output.LATENT_LOGVAR],
        }

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        if self.training or not self.calibrated_loss:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def _compute_kl_loss(self, mean, log_variance):
        return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

    def _calibrated_loss_function(self, model_out, batch, *args, **kwargs) -> dict:
        """https://stackoverflow.com/questions/64909658/what-could-cause-a-vaevariational-autoencoder-to-output-random-noise-even-afte

        Computes the VAE loss function.
        KL(N(mu, sigma), N(0, 1)) = log frac{1}{sigma} + frac{sigma^2 + mu^2}{2} - frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        predictions = model_out[Output.RECONSTRUCTION]
        targets = batch["image"]
        mean = model_out[Output.LATENT_MU]
        log_variance = model_out[Output.LATENT_LOGVAR]
        mse = F.mse_loss(predictions, targets, reduction="mean")
        log_sigma_opt = 0.5 * mse.log()
        r_loss = 0.5 * torch.pow((targets - predictions) / log_sigma_opt.exp(), 2) + log_sigma_opt
        r_loss = r_loss.sum()
        kl_loss = self._compute_kl_loss(mean, log_variance)
        loss = r_loss + kl_loss
        return {
            "loss": loss,
            "reconstruction": r_loss.detach() / targets.shape[0],
            "kld": kl_loss.detach() / targets.shape[0],
        }

    def _uncalibrated_loss_function(self, model_out, batch, *args, **kwargs) -> dict:
        """https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/experiment.py#L15

        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        predictions = model_out[Output.RECONSTRUCTION]
        targets = batch["image"]
        mean = model_out[Output.LATENT_MU]
        log_variance = model_out[Output.LATENT_LOGVAR]

        kld_weight = self.kld_weight  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(predictions, targets)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_variance - mean**2 - log_variance.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

    def loss_function(self, model_out, batch, *args, **kwargs) -> dict:
        """https://stackoverflow.com/questions/64909658/what-could-cause-a-vaevariational-autoencoder-to-output-random-noise-even-afte

        Computes the VAE loss function.
        KL(N(mu, sigma), N(0, 1)) = log frac{1}{sigma} + frac{sigma^2 + mu^2}{2} - frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        if self.calibrated_loss:
            return self._calibrated_loss_function(model_out=model_out, batch=batch, *args, **kwargs)
        else:
            return self._uncalibrated_loss_function(model_out=model_out, batch=batch, *args, **kwargs)

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
