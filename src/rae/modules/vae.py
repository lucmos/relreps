import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rae.modules.enumerations import Output
from rae.utils.tensor_ops import build_transposed_convolution, infer_dimension


class VanillaVAE(nn.Module):
    def __init__(
        self,
        metadata,
        input_size,
        in_channels: int,
        latent_dim: int,
        kld_weight: float,
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
        self.kld_weight = kld_weight

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Build Encoder
        self.encoder_shape_sequence = [
            [metadata.width, metadata.height],
        ]
        running_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    (conv2d := nn.Conv2d(running_channels, out_channels=h_dim, kernel_size=3, stride=2, padding=1)),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
                )
            )
            conv2d_out = infer_dimension(
                self.encoder_shape_sequence[-1][0],
                self.encoder_shape_sequence[-1][1],
                running_channels,
                conv2d,
            )
            self.encoder_shape_sequence.append([conv2d_out.shape[2], conv2d_out.shape[3]])
            running_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.encoder_out_shape = infer_dimension(
            metadata.width, metadata.height, n_channels=in_channels, model=self.encoder, batch_size=2
        ).shape
        encoder_out_numel = math.prod(self.encoder_out_shape[1:])
        self.fc_mu = nn.Linear(encoder_out_numel, latent_dim)
        self.fc_var = nn.Linear(encoder_out_numel, latent_dim)

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, encoder_out_numel)

        hidden_dims.reverse()
        hidden_dims = hidden_dims + hidden_dims[-1:]

        running_input_width = self.encoder_out_shape[2]
        running_input_height = self.encoder_out_shape[3]
        modules = []
        for i, (target_output_width, target_output_height) in zip(
            range(len(hidden_dims) - 1), reversed(self.encoder_shape_sequence[:-1])
        ):
            modules.append(
                nn.Sequential(
                    build_transposed_convolution(
                        in_channels=hidden_dims[i],
                        out_channels=hidden_dims[i + 1],
                        target_output_width=target_output_width,
                        target_output_height=target_output_height,
                        input_width=running_input_width,
                        input_height=running_input_height,
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(),
                )
            )
            running_input_width = target_output_width
            running_input_height = target_output_height

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
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

        return [mu, log_var]

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
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: Tensor, **kwargs) -> Dict[Output, Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return {
            Output.RECONSTRUCTION: x_recon,
            Output.DEFAULT_LATENT: mu,
            Output.BATCH_LATENT: z,
            Output.LATENT_MU: mu,
            Output.LATENT_LOGVAR: log_var,
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
        mu = model_out[Output.LATENT_MU]
        log_var = model_out[Output.LATENT_LOGVAR]

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.kld_weight * kld_loss
        return {"loss": loss, "reconstruction": recons_loss.detach(), "kld": -kld_loss.detach()}

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
