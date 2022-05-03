import torch
import torch.nn.functional as F
from torch import nn

from rae.data.datamodule import MetaData
from rae.modules.enumerations import Output


class Encoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1
        )  # out: hidden_channels x 14 x 14

        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=4, stride=2, padding=1
        )  # out: (hidden_channels x 2) x 7 x 7

        self.fc_mu = nn.Linear(in_features=hidden_channels * 2 * 7 * 7, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=hidden_channels * 2 * 7 * 7, out_features=latent_dim)

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        x = x.view(x.shape[0], -1)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        return x_mu, x_logvar


class RaeDecoder(nn.Module):
    def __init__(self, hidden_channels: int, latent_dim: int, normalize_latents: bool) -> None:
        super().__init__()
        self.normalize_latents = normalize_latents
        self.hidden_channels = hidden_channels

        self.fc = nn.Linear(in_features=latent_dim, out_features=hidden_channels * 2 * 7 * 7)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1
        )
        self.conv1 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.activation = nn.ReLU()

    def forward(self, batch_latent: torch.Tensor, anchors_latents: torch.Tensor) -> torch.Tensor:
        if self.normalize_latents:
            batch_latent = F.normalize(batch_latent, p=2, dim=-1)
            anchors_latents = F.normalize(anchors_latents, p=2, dim=-1)

        latent = torch.einsum("bi, ji -> bj", (batch_latent, anchors_latents))

        x = self.fc(latent)
        x = x.view(x.size(0), self.hidden_channels * 2, 7, 7)
        x = self.activation(self.conv2(x))
        x = torch.sigmoid(
            self.conv1(x)
        )  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x, latent


class RAE(nn.Module):
    def __init__(self, metadata: MetaData, hidden_channels: int, latent_dim: int, normalize_latents: bool):
        super().__init__()
        self.metadata = metadata
        self.register_buffer("anchors_images", metadata.anchors_images)
        self.register_buffer("anchors_latents", metadata.anchors_latents)

        if self.anchors_images is None and self.anchors_latents is None:
            raise ValueError("The RAE model needs anchors!")

        self.normalize_latents = normalize_latents

        self.encoder = Encoder(hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.decoder = RaeDecoder(
            hidden_channels=hidden_channels,
            latent_dim=(self.anchors_images if self.anchors_images is not None else self.anchors_latents).shape[0],
            normalize_latents=normalize_latents,
        )

    def forward(self, x):
        if self.anchors_images is not None:
            with torch.no_grad():
                anchors_latent, _, _ = self.embed(self.anchors_images)
        else:
            anchors_latent = self.anchors_latents

        batch_latent, batch_latent_mu, batch_latent_logvar = self.embed(x)

        x_recon, latent = self.decoder(batch_latent, anchors_latent)

        return {
            Output.OUT: x_recon,
            Output.DEFAULT_LATENT: Output.LATENT_MU,
            Output.ANCHORS_LATENT: anchors_latent,
            Output.BATCH_LATENT: batch_latent,
            Output.LATENT_MU: batch_latent_mu,
            Output.LATENT_LOGVAR: batch_latent_logvar,
            Output.INV_LATENTS: latent,
        }

    def embed(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(
            latent_mu,
            latent_logvar,
        )
        return latent, latent_mu, latent_logvar

    def latent_sample(self, mu, logvar):
        if self.training:
            # Convert the logvar to std
            std = (logvar * 0.5).exp()

            # the reparameterization trick
            return torch.distributions.Normal(loc=mu, scale=std).rsample()

            # Or if you prefer to do it without a torch.distribution...
            # std = logvar.mul(0.5).exp_()
            # eps = torch.empty_like(std).normal_()
            # return eps.mul(std).add_(mu)
        else:
            return mu
