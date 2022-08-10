import functools
import logging
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from rae.data.vision.datamodule import MetaData
from rae.modules.attention import NormalizationMode, RelativeEmbeddingMethod
from rae.modules.enumerations import Output
from rae.utils.tensor_ops import infer_dimension

pylogger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self, metadata: MetaData, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        # self.conv1 = nn.Conv2d(
        #     in_channels=1, out_channels=hidden_channels, kernel_size=4, stride=2, padding=1
        # )  # out: hidden_channels x 14 x 14
        #
        # self.conv2 = nn.Conv2d(
        #     in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=4, stride=2, padding=1
        # )  # out: (hidden_channels x 2) x 7 x 7

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=metadata.n_channels, out_channels=hidden_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3),
            nn.ReLU(),
        )

        fake_out = infer_dimension(metadata.width, metadata.height, metadata.n_channels, model=self.sequential)
        out_dimension = fake_out[0].nelement()

        self.fc_mu = nn.Linear(in_features=out_dimension, out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=out_dimension, out_features=latent_dim)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        x = self.sequential(x)

        x = x.view(x.shape[0], -1)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar


class RaeDecoder(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
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
        fake_out = infer_dimension(7, 7, self.hidden_channels * 2, model=self.conv2)

        stride = (2, 2)
        padding = (1, 1)
        output_padding = (0, 0)
        dilation = (1, 1)
        # kernel_w = (metadata.width - (fake_out.width −1)×stride[0] + 2×padding[0] - output_padding[0]  - 1)/dilation[0] + 1
        # kernel_h = (metadata.height - (fake_out.height −1)×stride[1] + 2×padding[1] - output_padding[1]  - 1)/dilation[1] + 1
        kernel_w = (
            metadata.width - (fake_out.size(2) - 1) * stride[0] + 2 * padding[0] - output_padding[0] - 1
        ) / dilation[0] + 1
        kernel_h = (
            metadata.height - (fake_out.size(3) - 1) * stride[1] + 2 * padding[0] - output_padding[1] - 1
        ) / dilation[1] + 1

        self.conv1 = nn.ConvTranspose2d(
            in_channels=fake_out.shape[1],
            out_channels=metadata.n_channels,
            kernel_size=(int(kernel_w), int(kernel_h)),
            stride=stride,
            padding=padding,
        )

        assert ((out := self.conv1(fake_out)).shape[2], out.shape[3]) == (metadata.width, metadata.height), (
            out.shape[2],
            out.shape[3],
            metadata.width,
            metadata.height,
        )

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
        x = self.conv1(x)  # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x, relative_embedding


class RAE(nn.Module):
    def __init__(
        self,
        metadata: MetaData,
        hidden_channels: int,
        latent_dim: int,
        normalize_latents: str,
        normalize_means: str = False,
        normalize_only_anchors_means: bool = False,
        reparametrize_anchors: bool = True,
        normalize_only_anchors_latents: bool = False,
        relative_embedding_method: str = RelativeEmbeddingMethod.INNER,
        normalize_relative_embedding: str = NormalizationMode.OFF,
        **kwargs,
    ):
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        if relative_embedding_method not in set(RelativeEmbeddingMethod):
            raise ValueError(f"Relative embedding method not valid: {relative_embedding_method}")

        if normalize_relative_embedding not in set(NormalizationMode):
            raise ValueError(f"Relative Embedding normalization not valid: {normalize_relative_embedding}")

        self.metadata = metadata
        self.register_buffer("anchors_images", metadata.anchors_images)
        self.register_buffer("anchors_latents", metadata.anchors_latents)

        if self.anchors_images is None and self.anchors_latents is None:
            raise ValueError("The RAE model needs anchors!")

        self.reparametrize_anchors = reparametrize_anchors

        self.normalize_latents = normalize_latents
        self.normalize_only_anchors_latents = normalize_only_anchors_latents

        self.normalize_means = normalize_means
        self.normalize_only_anchors_means = normalize_only_anchors_means

        self.encoder = Encoder(metadata=metadata, hidden_channels=hidden_channels, latent_dim=latent_dim)
        self.decoder = RaeDecoder(
            metadata=metadata,
            hidden_channels=hidden_channels,
            latent_dim=(self.anchors_images if self.anchors_images is not None else self.anchors_latents).shape[0],
            relative_embedding_method=relative_embedding_method,
            normalize_relative_embedding=normalize_relative_embedding,
        )

        self.mean_normalization = self._instantiate_normalization(
            normalization_mode=self.normalize_means, latent_dim=latent_dim
        )
        self.latent_normalization = self._instantiate_normalization(
            normalization_mode=self.normalize_latents, latent_dim=latent_dim
        )

    @staticmethod
    def _instantiate_normalization(normalization_mode: Union[str, NormalizationMode], latent_dim: int):
        if normalization_mode == NormalizationMode.BATCHNORM:
            return nn.BatchNorm1d(num_features=latent_dim, track_running_stats=True)
        elif normalization_mode == NormalizationMode.INSTANCENORM:
            return nn.InstanceNorm1d(num_features=latent_dim, affine=True, track_running_stats=True)
        elif normalization_mode == NormalizationMode.LAYERNORM:
            return nn.LayerNorm(latent_dim, elementwise_affine=True)
        elif normalization_mode == NormalizationMode.INSTANCENORM_NOAFFINE:
            return nn.InstanceNorm1d(num_features=latent_dim, affine=False, track_running_stats=True)
        elif normalization_mode == NormalizationMode.LAYERNORM_NOAFFINE:
            return nn.LayerNorm(latent_dim, elementwise_affine=False)
        elif (
            isinstance(normalization_mode, bool) and normalization_mode
        ) or normalization_mode == NormalizationMode.L2:
            return functools.partial(F.normalize, p=2, dim=-1)
        elif (
            isinstance(normalization_mode, bool) and not normalization_mode
        ) or normalization_mode == NormalizationMode.OFF:
            return None
        else:
            raise ValueError(f"Invalid latent normalization {normalization_mode}")

    @staticmethod
    def _apply_normalization(
        normalization_mode: Union[str, NormalizationMode],
        normalization_fn: Callable[[torch.Tensor], torch.Tensor],
        x: torch.Tensor,
    ) -> torch.Tensor:
        if normalization_fn is None or normalization_mode == NormalizationMode.OFF or not normalization_mode:
            return x
        if isinstance(normalization_fn, nn.InstanceNorm1d):
            x = torch.transpose(x, 1, 0)
            x = normalization_fn(x)
            x = torch.transpose(x, 1, 0)
            return x
        else:
            return normalization_fn(x)

    def forward(self, x):
        if self.anchors_images is not None:
            with torch.no_grad():
                anchors_latent, _, _ = self.embed(
                    self.anchors_images, normalize_mean=self.normalize_means, reparametrize=self.reparametrize_anchors
                )
        else:
            anchors_latent = self.anchors_latents

        batch_latent, batch_latent_mu, batch_latent_logvar = self.embed(
            x,
            normalize_mean=self.normalize_means if not self.normalize_only_anchors_means else NormalizationMode.OFF,
            reparametrize=True,
        )

        if not self.normalize_only_anchors_latents:
            batch_latent = self._apply_normalization(
                normalization_mode=self.normalize_latents, normalization_fn=self.latent_normalization, x=batch_latent
            )
        anchors_latent = self._apply_normalization(
            normalization_mode=self.normalize_latents, normalization_fn=self.latent_normalization, x=anchors_latent
        )

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

    def embed(self, x, reparametrize: bool, normalize_mean: Union[NormalizationMode, str]):
        latent_mu, latent_logvar = self.encoder(x)

        latent_mu = self._apply_normalization(
            normalization_mode=normalize_mean, normalization_fn=self.mean_normalization, x=latent_mu
        )

        if reparametrize:
            latent = self.latent_sample(
                latent_mu,
                latent_logvar,
            )
        else:
            latent = latent_mu

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
