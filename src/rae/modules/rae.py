import functools
from enum import auto
from typing import Optional

import torch
import torch.nn.functional as F
from backports.strenum import StrEnum
from torch import nn

from rae.data.datamodule import MetaData
from rae.modules.enumerations import Output


class RelativeEmbeddingMethod(StrEnum):
    BASIS_CHANGE = auto()
    INNER = auto()


class NormalizationMode(StrEnum):
    L2 = auto()
    OFF = auto()
    BATCHNORM = auto()
    INSTANCENORM = auto()
    LAYERNORM = auto()
    INSTANCENORM_NOAFFINE = auto()
    LAYERNORM_NOAFFINE = auto()


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
    def __init__(
        self,
        hidden_channels: int,
        latent_dim: int,
        relative_embedding_method: str,
        normalize_relative_embedding: str,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels

        self.fc = nn.Linear(in_features=latent_dim, out_features=hidden_channels * 2 * 7 * 7)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=hidden_channels * 2, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1
        )
        self.conv1 = nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=1, kernel_size=4, stride=2, padding=1)

        self.activation = nn.ReLU()
        self.relative_embedding_method = relative_embedding_method
        self.normalize_relative_embedding = normalize_relative_embedding

        if self.normalize_relative_embedding == NormalizationMode.BATCHNORM:
            self.batch_norm = nn.BatchNorm1d(num_features=latent_dim)

    def forward(
        self,
        batch_latent: Optional[torch.Tensor] = None,
        anchors_latents: Optional[torch.Tensor] = None,
        relative_embedding: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, torch.Tensor):
        if relative_embedding is None:
            if self.relative_embedding_method == RelativeEmbeddingMethod.INNER:
                relative_embedding = torch.einsum("bi, ji -> bj", (batch_latent, anchors_latents))

            elif self.relative_embedding_method == RelativeEmbeddingMethod.BASIS_CHANGE:
                relative_embedding = torch.linalg.lstsq(anchors_latents.T, batch_latent.T)[0].T

        if self.normalize_relative_embedding == NormalizationMode.L2:
            relative_embedding = F.normalize(relative_embedding, p=2, dim=-1)
        elif self.normalize_relative_embedding == NormalizationMode.BATCHNORM:
            relative_embedding = self.batch_norm(relative_embedding)

        x = self.fc(relative_embedding)
        x = x.view(x.size(0), self.hidden_channels * 2, 7, 7)
        x = self.activation(self.conv2(x))
        x = torch.sigmoid(
            self.conv1(x)
        )  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x, relative_embedding


class RAE(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        hidden_channels: int,
        latent_dim: int,
        normalize_latents: str,
        normalize_only_anchors_latents: bool = False,
        relative_embedding_method: str = RelativeEmbeddingMethod.INNER,
        normalize_relative_embedding: str = NormalizationMode.OFF,
    ):
        super().__init__()

        if relative_embedding_method not in set(RelativeEmbeddingMethod):
            raise ValueError(f"Relative embedding method not valid: {relative_embedding_method}")

        if normalize_relative_embedding not in set(NormalizationMode):
            raise ValueError(f"Relative Embedding normalization not valid: {normalize_relative_embedding}")

        self.metadata = metadata
        self.register_buffer("anchors_images", metadata.anchors_images)
        self.register_buffer("anchors_latents", metadata.anchors_latents)

        if self.anchors_images is None and self.anchors_latents is None:
            raise ValueError("The RAE model needs anchors!")

        self.normalize_latents = normalize_latents
        self.normalize_only_anchors_latents = normalize_only_anchors_latents

        self.encoder = Encoder(hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.decoder = RaeDecoder(
            hidden_channels=hidden_channels,
            latent_dim=(self.anchors_images if self.anchors_images is not None else self.anchors_latents).shape[0],
            relative_embedding_method=relative_embedding_method,
            normalize_relative_embedding=normalize_relative_embedding,
        )

        if self.normalize_latents == NormalizationMode.BATCHNORM:
            self.latent_normalization = nn.BatchNorm1d(num_features=latent_dim, track_running_stats=True)
        elif self.normalize_latents == NormalizationMode.INSTANCENORM:
            self.latent_normalization = nn.InstanceNorm1d(
                num_features=latent_dim, affine=True, track_running_stats=True
            )
        elif self.normalize_latents == NormalizationMode.LAYERNORM:
            self.latent_normalization = nn.LayerNorm(latent_dim, elementwise_affine=True)
        elif self.normalize_latents == NormalizationMode.INSTANCENORM_NOAFFINE:
            self.latent_normalization = nn.InstanceNorm1d(
                num_features=latent_dim, affine=False, track_running_stats=True
            )
        elif self.normalize_latents == NormalizationMode.LAYERNORM_NOAFFINE:
            self.latent_normalization = nn.LayerNorm(latent_dim, elementwise_affine=False)
        elif (
            isinstance(self.normalize_latents, bool) and self.normalize_latents
        ) or self.normalize_latents == NormalizationMode.L2:
            self.latent_normalization = functools.partial(F.normalize, p=2, dim=-1)
        elif isinstance(self.normalize_latents, bool) and not self.normalize_latents:
            self.latent_normalization = None
        else:
            raise ValueError(f"Invalid latent normalization {self.latent_normalization}")

    def apply_latent_normalization(self, x: torch.Tensor) -> torch.Tensor:
        if self.latent_normalization is None:
            return x
        if (
            self.normalize_latents == NormalizationMode.INSTANCENORM
            or self.normalize_latents == NormalizationMode.INSTANCENORM_NOAFFINE
        ):
            x = torch.transpose(x, 1, 0)
            x = self.latent_normalization(x)
            x = torch.transpose(x, 1, 0)
            return x
        else:
            return self.latent_normalization(x)

    def forward(self, x):
        if self.anchors_images is not None:
            with torch.no_grad():
                anchors_latent, _, _ = self.embed(self.anchors_images)
        else:
            anchors_latent = self.anchors_latents

        batch_latent, batch_latent_mu, batch_latent_logvar = self.embed(x)

        if self.normalize_latents:
            if not self.normalize_only_anchors_latents:
                batch_latent = self.apply_latent_normalization(batch_latent)
            anchors_latent = self.apply_latent_normalization(anchors_latent)

        x_recon, latent = self.decoder(batch_latent, anchors_latent)

        return {
            Output.RECONSTRUCTION: x_recon,
            Output.DEFAULT_LATENT: batch_latent_mu,
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
