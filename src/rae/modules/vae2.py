import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rae.modules.blocks import build_transposed_convolution
from rae.modules.enumerations import Output
from rae.utils.tensor_ops import infer_dimension


def build_dynamic_encoder_decoder2(
    width,
    height,
    n_channels,
    hidden_dims: Optional[Sequence[int]],
    last_activation: str = "tanh",
) -> Tuple[nn.Module, Sequence[int], nn.Module]:
    """Builds a dynamic convolutional encoder-decoder pair with parametrized hidden dimensions number and size.

    Args:
        width: the width of the images to work with
        height: the height of the images
        n_channels: the number of channels of the images
        hidden_dims: a sequence of ints to specify the number and size of the hidden layers in the encoder and decoder

    Returns:
        the encoder, the shape in the latent space, the decoder
    """
    modules = []

    if hidden_dims is None:
        hidden_dims = (32, 64, 128, 256)

    STRIDE = (2, 2)
    PADDING = (1, 1)
    # Build Encoder
    encoder_shape_sequence = [
        [width, height],
    ]
    running_channels = n_channels
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                (
                    conv2d := nn.Conv2d(
                        running_channels, out_channels=h_dim, kernel_size=3, stride=STRIDE, padding=PADDING
                    )
                ),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(),
            )
        )
        conv2d_out = infer_dimension(
            encoder_shape_sequence[-1][0],
            encoder_shape_sequence[-1][1],
            running_channels,
            conv2d,
        )
        encoder_shape_sequence.append([conv2d_out.shape[2], conv2d_out.shape[3]])
        running_channels = h_dim

    encoder = nn.Sequential(*modules)

    encoder_out_shape = infer_dimension(width, height, n_channels=n_channels, model=encoder, batch_size=1).shape

    # Build Decoder
    hidden_dims = list(reversed(hidden_dims))
    hidden_dims = hidden_dims + hidden_dims[-1:]

    running_input_width = encoder_out_shape[2]
    running_input_height = encoder_out_shape[3]
    modules = []
    for i, (target_output_width, target_output_height) in zip(
        range(len(hidden_dims) - 1), reversed(encoder_shape_sequence[:-1])
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
                    stride=STRIDE,
                    padding=PADDING,
                ),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU(),
            )
        )
        running_input_width = target_output_width
        running_input_height = target_output_height

    decoder = nn.Sequential(
        *modules,
        nn.Sequential(
            nn.Conv2d(hidden_dims[-1], out_channels=n_channels, kernel_size=3, padding=1),
            (nn.Tanh() if last_activation == "tanh" else nn.Sigmoid()),
        ),
    )
    return encoder, encoder_out_shape, decoder


class VanillaVAE(nn.Module):
    def __init__(
        self,
        metadata,
        input_size,
        latent_dim: int,
        kld_weight: float,
        hidden_dims: List = None,
        last_activation: str = "tanh",
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

        self.encoder, self.encoder_out_shape, self.decoder = build_dynamic_encoder_decoder2(
            width=metadata.width,
            height=metadata.height,
            n_channels=metadata.n_channels,
            hidden_dims=hidden_dims,
            last_activation=last_activation,
        )
        encoder_out_numel = math.prod(self.encoder_out_shape[1:])

        self.fc_mu = nn.Linear(encoder_out_numel, latent_dim)
        self.fc_var = nn.Linear(encoder_out_numel, latent_dim)

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, encoder_out_numel)

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

    def _compute_kl_loss(self, mean, log_variance):
        return -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

    def loss_function(self, model_out, batch, *args, **kwargs) -> dict:
        """https://stackoverflow.com/questions/64909658/what-could-cause-a-vaevariational-autoencoder-to-output-random-noise-even-afte

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

        # kld_weight: float = kwargs.get("kld_weight", self.kld_weight)
        # recons = model_out[Output.RECONSTRUCTION]
        # input = batch["image"]
        # mu = model_out[Output.LATENT_MU]
        # log_var = model_out[Output.LATENT_LOGVAR]
        #
        # recons_loss = F.mse_loss(recons, input, reduction="mean")
        #
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        #
        # loss = recons_loss + kld_weight * kld_loss
        # return {"loss": loss, "reconstruction": recons_loss.detach(), "kld": -kld_loss.detach()}

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
