import abc
import logging
import math
from enum import auto
from typing import Dict, Iterable, Optional, Sequence, Set, Union

import hydra.utils
import torch
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from torch import nn
from torch.nn import ModuleList

from rae.utils.tensor_ops import stratified_gaussian_sampling, stratified_mean, stratified_sampling

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum

pylogger = logging.getLogger(__name__)


class AnchorsSamplingMode(StrEnum):
    NONE = auto()
    STRATIFIED = auto()
    GAUSSIAN = auto()


class AttentionElement(StrEnum):
    NONE = auto()
    ATTENTION_KEYS = auto()
    ATTENTION_QUERIES = auto()
    ATTENTION_VALUES = auto()


class NormalizationMode(StrEnum):
    NONE = auto()
    L2 = auto()
    # BATCHNORM = auto()
    # INSTANCENORM = auto()
    # LAYERNORM = auto()
    # INSTANCENORM_NOAFFINE = auto()
    # LAYERNORM_NOAFFINE = auto()


class RelativeEmbeddingMethod(StrEnum):
    BASIS_CHANGE = auto()
    INNER = auto()


class SimilaritiesAggregationMode(StrEnum):
    NONE = auto()
    STRATIFIED_AVG = auto()


class SimilaritiesQuantizationMode(StrEnum):
    CUSTOM_ROUND = auto()
    NONE = auto()
    DIFFERENTIABLE_ROUND = auto()
    SMOOTH_STEPS = auto()
    CLUSTER = auto()


class AbsoluteQuantizationMode(StrEnum):
    NONE = auto()
    DIFFERENTIABLE_ROUND = auto()
    CLUSTER = auto()
    # SMOOTH_STEPS = auto()


class ValuesMethod(StrEnum):
    RELATIVE_ANCHORS_ATTENTION = auto()
    SELF_ATTENTION = auto()
    SIMILARITIES = auto()
    TRAINABLE = auto()
    ANCHORS = auto()


class OutputNormalization(StrEnum):
    NONE = auto()
    L2 = auto()
    BATCHNORM = auto()
    INSTANCENORM = auto()
    LAYERNORM = auto()


class SubspacePooling(StrEnum):
    NONE = auto()
    SUM = auto()
    MEAN = auto()
    LINEAR = auto()
    MAX = auto()


class AttentionOutput(StrEnum):
    NON_QUANTIZED_SIMILARITIES = auto()
    SUBSPACE_OUTPUTS = auto()
    ANCHORS_TARGETS = auto()
    ORIGINAL_ANCHORS = auto()
    ANCHORS = auto()
    BATCH_LATENT = auto()
    ANCHORS_LATENT = auto()
    OUTPUT = auto()
    SIMILARITIES = auto()
    UNTRASFORMED_ATTENDED = auto()


class CustomRound(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, similarities_bin_size):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, torch.as_tensor(similarities_bin_size))
        return torch.round(input / similarities_bin_size) * similarities_bin_size

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        (input, similarities_bin_size) = ctx.saved_tensors
        return torch.round(grad_output / similarities_bin_size) * similarities_bin_size


cround = CustomRound.apply


class QKVTransforms(nn.Module):
    def __init__(
        self,
        transform_elements: Iterable[AttentionElement],
        in_features: int,
        hidden_features: int,
        values_mode: ValuesMethod,
    ):
        """Block that gathers the transformation of the queries, keys and values in the attention.

        If values_mode = TRAINABLE we are invariant to batch-anchors rotations, if it is false we are equivariant
        to batch-anchors rotations

        Args:
            in_features: hidden dimension of the input batch and anchors
            hidden_features: hidden dimension of the transformed tensor
            transform_elements: the attention elements to independently transform with a linear layer
            values_mode: if True use trainable parameters as queries otherwise use the anchors
        """
        super().__init__()

        self.transform_elements = set(transform_elements)
        pylogger.info(f"Transforming: {self.transform_elements}")

        self.module_dict = nn.ModuleDict(
            {
                element: nn.Linear(
                    in_features=in_features,
                    out_features=hidden_features,
                    bias=False,
                )
                for element in self.transform_elements
            }
        )

        if AttentionElement.ATTENTION_VALUES in self.transform_elements and not values_mode == ValuesMethod.ANCHORS:
            raise ValueError(f"Impossible to transform values if the values mode is {values_mode}")

    def forward(self, x: torch.Tensor, element: AttentionElement) -> torch.Tensor:
        if element in self.module_dict:
            return self.module_dict[element](x)
        else:
            return x


class AbstractRelativeAttention(nn.Module, abc.ABC):
    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError


class RelativeAttention(AbstractRelativeAttention):
    def __init__(
        self,
        n_anchors: int,
        n_classes: int,
        similarity_mode: RelativeEmbeddingMethod,
        values_mode: ValuesMethod,
        normalization_mode: Optional[NormalizationMode] = None,
        in_features: Optional[int] = None,
        hidden_features: Optional[int] = None,
        transform_elements: Optional[Set[AttentionElement]] = None,
        self_attention_hidden_dim: Optional[int] = None,
        values_self_attention_nhead: Optional[int] = None,
        absolute_quantization_mode: Optional[AbsoluteQuantizationMode] = None,
        absolute_num_clusters: Optional[int] = None,
        absolute_bin_size: Optional[float] = None,
        similarities_quantization_mode: Optional[SimilaritiesQuantizationMode] = None,
        similarities_num_clusters: Optional[int] = None,
        similarities_bin_size: Optional[float] = None,
        similarities_aggregation_mode: Optional[SimilaritiesAggregationMode] = None,
        similarities_aggregation_n_groups: Optional[int] = None,
        anchors_sampling_mode: Optional[AnchorsSamplingMode] = None,
        n_anchors_sampling_per_class: Optional[int] = None,
        output_normalization_mode: Optional[OutputNormalization] = None,
    ):
        """Relative attention block.

        If values_mode = TRAINABLE we are invariant to batch-anchors rotations, if it is false we are equivariant
        to batch-anchors rotations

        Args:
            in_features: hidden dimension of the input batch and anchors
            hidden_features: hidden dimension of the output (the queries, if trainable)
            n_anchors: number of anchors
            n_classes: number of classes
            transform_elements: the attention elements to independently transform with a linear layer
            normalization_mode: normalization to apply to the anchors and batch before computing the attention
            similarity_mode: how to compute similarities: inner, basis_change
            values_mode: if True use trainable parameters as queries otherwise use the anchors
            values_self_attention_nhead: number of head if the values mode is self attention
            similarities_quantization_mode: the quantization modality to quantize the similarities
            similarities_bin_size: the size of the bins in the quantized similarities
            similarities_aggregation_mode: how the similarities should be aggregated
            similarities_aggregation_n_groups: number of groups when aggregating the similarities
            anchors_sampling_mode: how to sample the anchors from the available ones at each step
            n_anchors_sampling_per_class: how many anchors must be sampled for each class
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.absolute_num_clusters = absolute_num_clusters
        self.similarities_num_clusters = similarities_num_clusters
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.n_anchors = n_anchors
        self.n_classes = n_classes
        self.similarity_mode = similarity_mode

        # Parameter validation
        self.transform_elements = set(transform_elements) if transform_elements is not None else set()

        self.normalization_mode = normalization_mode if normalization_mode is not None else NormalizationMode.NONE

        self.values_mode = values_mode
        self.self_attention_hidden_dim = self_attention_hidden_dim
        self.values_self_attention_nhead = values_self_attention_nhead
        if self.values_self_attention_nhead and self.values_mode != ValuesMethod.SELF_ATTENTION:
            raise ValueError(
                f"values_self_attention_nhead={self.values_self_attention_nhead} "
                f"provided but unused if values_mode={self.values_mode}"
            )

        self.absolute_quantization_mode = (
            absolute_quantization_mode if absolute_quantization_mode is not None else AbsoluteQuantizationMode.NONE
        )
        self.absolute_bin_size = absolute_bin_size
        if self.absolute_bin_size and self.absolute_quantization_mode == AbsoluteQuantizationMode.NONE:
            raise ValueError(
                f"absolute_bin_size={self.absolute_bin_size} "
                f"provided but unused if similarities_quantization_mode={self.absolute_quantization_mode}"
            )

        self.similarities_quantization_mode = (
            similarities_quantization_mode
            if similarities_quantization_mode is not None
            else SimilaritiesQuantizationMode.NONE
        )
        self.similarities_bin_size = similarities_bin_size
        if self.similarities_bin_size and self.similarities_quantization_mode == SimilaritiesQuantizationMode.NONE:
            raise ValueError(
                f"similarities_bin_size={self.similarities_bin_size} "
                f"provided but unused if similarities_quantization_mode={self.similarities_quantization_mode}"
            )

        self.similarities_aggregation_mode = (
            similarities_aggregation_mode
            if similarities_aggregation_mode is not None
            else SimilaritiesAggregationMode.NONE
        )
        self.similarities_aggregation_n_groups = similarities_aggregation_n_groups
        if (
            self.similarities_aggregation_n_groups
            and self.similarities_aggregation_mode == SimilaritiesAggregationMode.NONE
        ):
            raise ValueError(
                f"similarities_aggregation_n_groups={self.similarities_aggregation_n_groups} "
                f"provided but unused if similarities_aggregation_mode={self.similarities_aggregation_mode}"
            )

        self.anchors_sampling_mode = (
            anchors_sampling_mode if anchors_sampling_mode is not None else AnchorsSamplingMode.NONE
        )
        self.n_anchors_sampling_per_class = n_anchors_sampling_per_class
        if (self.n_anchors_sampling_per_class and self.n_anchors_sampling_per_class > 1) and (
            self.anchors_sampling_mode == AnchorsSamplingMode.NONE
        ):
            raise ValueError(
                f"n_anchors_sampling_per_class={self.n_anchors_sampling_per_class} "
                f"provided but unused if anchors_sampling_mode={self.anchors_sampling_mode}"
            )

        self.output_normalization_mode = (
            output_normalization_mode if output_normalization_mode is not None else OutputNormalization.NONE
        )

        if self.output_normalization_mode not in set(OutputNormalization):
            raise ValueError(f"Unknown output normalization mode {self.output_normalization_mode}")

        if (similarities_quantization_mode in set(SimilaritiesQuantizationMode)) != (similarities_bin_size is not None):
            raise ValueError(
                f"Quantization '{similarities_quantization_mode}' not supported with bin size '{similarities_bin_size}'"
            )

        if similarities_aggregation_mode is not None and similarities_aggregation_mode not in set(
            SimilaritiesAggregationMode
        ):
            raise ValueError(f"Similarity aggregation mode not supported: {similarities_aggregation_mode}")

        if anchors_sampling_mode is not None and anchors_sampling_mode not in set(AnchorsSamplingMode):
            raise ValueError(f"Anchors sampling mode not supported: {anchors_sampling_mode}")
        if anchors_sampling_mode is not None and n_anchors_sampling_per_class is None:
            raise ValueError(
                f"Impossible to sample with mode: {anchors_sampling_mode} without specifying the number of anchors per class"
            )

        if self.similarity_mode == RelativeEmbeddingMethod.BASIS_CHANGE and (
            self.n_anchors_sampling_per_class is not None and self.n_anchors_sampling_per_class > 1
        ):
            raise ValueError("The basis change is not deterministic with possibly repeated basis vectors")

        # End Parameter validation

        self.pre_attention_transforms = QKVTransforms(
            in_features=in_features,
            hidden_features=hidden_features,
            transform_elements=self.transform_elements,
            values_mode=values_mode,
        )

        if values_mode == ValuesMethod.TRAINABLE:
            if self.similarities_aggregation_mode == SimilaritiesAggregationMode.STRATIFIED_AVG:
                self.values = nn.Parameter(
                    torch.randn(self.n_classes * self.similarities_aggregation_n_groups, self.hidden_features)
                )
            elif self.anchors_sampling_mode == AnchorsSamplingMode.STRATIFIED:
                self.values = nn.Parameter(
                    torch.randn(self.n_classes * self.n_anchors_sampling_per_class, self.hidden_features)
                )
            else:
                self.values = nn.Parameter(torch.randn(self.n_anchors, self.hidden_features))
        elif values_mode == ValuesMethod.SELF_ATTENTION:
            self.sim_norm = nn.LayerNorm(normalized_shape=self.output_dim)
            # self.sim_to_queries = nn.Linear(1, self.self_attention_hidden_dim, bias=False)
            # self.sim_to_keys = nn.Linear(1, self.self_attention_hidden_dim, bias=False)
            # self.sim_to_values = nn.Linear(1, self.self_attention_hidden_dim, bias=False)

            self.sim_to_queries = nn.Parameter(torch.randn(self.n_anchors, self.self_attention_hidden_dim))
            self.sim_to_keys = nn.Parameter(torch.randn(self.n_anchors, self.self_attention_hidden_dim))
            self.sim_to_values = nn.Parameter(torch.randn(self.n_anchors, self.self_attention_hidden_dim))

            self.self_attention = nn.MultiheadAttention(
                embed_dim=self.self_attention_hidden_dim,
                num_heads=values_self_attention_nhead,
                batch_first=True,
            )

            self.self_attention_aggregation = nn.Linear(
                in_features=self.self_attention_hidden_dim, out_features=1, bias=False
            )
        elif values_mode == ValuesMethod.RELATIVE_ANCHORS_ATTENTION:
            self.sim_norm = nn.LayerNorm(normalized_shape=self.n_anchors)
            self.anchors_relative_attention = RelativeAttention(
                n_anchors=n_anchors,
                n_classes=n_classes,
                similarity_mode=similarity_mode,
                values_mode=ValuesMethod.SIMILARITIES,
                normalization_mode=normalization_mode,
                in_features=in_features,
                hidden_features=hidden_features,
                transform_elements=transform_elements,
                self_attention_hidden_dim=self_attention_hidden_dim,
                values_self_attention_nhead=values_self_attention_nhead,
                similarities_quantization_mode=similarities_quantization_mode,
                similarities_bin_size=similarities_bin_size,
                similarities_aggregation_mode=similarities_aggregation_mode,
                similarities_aggregation_n_groups=similarities_aggregation_n_groups,
                anchors_sampling_mode=anchors_sampling_mode,
                n_anchors_sampling_per_class=n_anchors_sampling_per_class,
                output_normalization_mode=output_normalization_mode,
            )
        elif values_mode not in set(ValuesMethod):
            raise ValueError(f"Values mode not supported: {self.values_mode}")

        if self.output_normalization_mode == OutputNormalization.BATCHNORM:
            self.outnorm = nn.BatchNorm1d(num_features=self.output_dim)
        elif self.output_normalization_mode == OutputNormalization.LAYERNORM:
            self.outnorm = nn.LayerNorm(normalized_shape=self.output_dim)
        elif self.output_normalization_mode == OutputNormalization.INSTANCENORM:
            self.outnorm = nn.InstanceNorm1d(num_features=self.output_dim, affine=True)

    def encode(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
        anchors_targets: Optional[torch.Tensor] = None,
    ):
        original_anchors = anchors
        if x.shape[-1] != anchors.shape[-1]:
            raise ValueError(f"Inconsistent dimensions between batch and anchors: {x.shape}, {anchors.shape}")

        # Sample anchors
        if self.anchors_sampling_mode == AnchorsSamplingMode.NONE:
            pass
        elif self.anchors_sampling_mode == AnchorsSamplingMode.STRATIFIED:
            sampling_idxs = stratified_sampling(
                targets=anchors_targets, samples_per_class=self.n_anchors_sampling_per_class
            )
            anchors_targets = anchors_targets[sampling_idxs]
            anchors = anchors[sampling_idxs, :]
        elif self.anchors_sampling_mode == AnchorsSamplingMode.GAUSSIAN:
            anchors = stratified_gaussian_sampling(
                values=anchors, targets=anchors_targets, samples_per_class=self.n_anchors_sampling_per_class
            )
        else:
            raise ValueError(f"Sampling mode not supported: {self.anchors_sampling_mode}")

        # Transform into keys and queries
        x = x_latents = self.pre_attention_transforms(x, element=AttentionElement.ATTENTION_QUERIES)
        anchors = anchors_latents = self.pre_attention_transforms(anchors, element=AttentionElement.ATTENTION_KEYS)

        # Normalize latents
        if self.normalization_mode == NormalizationMode.NONE:
            pass
        elif self.normalization_mode == NormalizationMode.L2:
            x = F.normalize(x, p=2, dim=-1)
            anchors = F.normalize(anchors, p=2, dim=-1)
        else:
            raise ValueError(f"Normalization mode not supported: {self.normalization_mode}")

        # Quantize absolute
        if self.absolute_quantization_mode == AbsoluteQuantizationMode.NONE:
            pass
        elif self.absolute_quantization_mode == AbsoluteQuantizationMode.DIFFERENTIABLE_ROUND:
            anchors = torch.round(anchors / self.absolute_bin_size) * self.absolute_bin_size
            anchors = anchors + (anchors - anchors).detach()

            x = torch.round(x / self.absolute_bin_size) * self.absolute_bin_size
            x = x + (x - x).detach()
        elif self.absolute_quantization_mode == AbsoluteQuantizationMode.CLUSTER:

            # kmeans = KMeans(n_clusters=self.absolute_num_clusters, random_state=0).fit(
            #     torch.cat([x, anchors]).detach().cpu().numpy()
            # )
            #
            # x = torch.as_tensor(kmeans.cluster_centers_[kmeans.predict(x.detach().cpu().numpy())])
            # anchors = torch.as_tensor(kmeans.cluster_centers_[kmeans.predict(anchors.detach().cpu().numpy())])
            union = torch.cat([x, anchors])
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=self.absolute_num_clusters
            ).fit_predict(union.detach().cpu().numpy())
            cluster_centers = torch.zeros((labels.max() + 1, x.shape[1]))

            for i in range(labels.max() + 1):
                cluster_centers[i] = union[labels == i].mean(0)

            x = torch.as_tensor(cluster_centers[labels[: x.shape[0]]])
            anchors = torch.as_tensor(cluster_centers[labels[x.shape[0] :]])

        elif self.absolute_quantization_mode == AbsoluteQuantizationMode.SMOOTH_STEPS:
            raise NotImplementedError()

        # Compute queries-keys similarities
        if self.similarity_mode == RelativeEmbeddingMethod.INNER:
            similarities = torch.einsum("bm, am -> ba", x, anchors)
            if self.normalization_mode == NormalizationMode.NONE:
                similarities = similarities / math.sqrt(x.shape[-1])
        elif self.similarity_mode == RelativeEmbeddingMethod.BASIS_CHANGE:
            similarities = torch.linalg.lstsq(anchors.T, x.T)[0].T
        else:
            raise ValueError(f"Similarity mode not supported: {self.similarity_mode}")

        # Aggregate similarities
        # TODO; add possibility to perform partial aggregation to have more features than num_classes
        if self.similarities_aggregation_mode == SimilaritiesAggregationMode.STRATIFIED_AVG:
            # Compute the mean of the similarities grouped by anchor targets
            if anchors_targets is None:
                raise ValueError("Impossible to stratify the similarities without anchors targets")
            similarities = stratified_mean(
                samples=similarities,
                labels=anchors_targets,
                n_groups=self.similarities_aggregation_n_groups,
                num_classes=self.n_classes,
            )

        # Quantize similarities
        quantized_similarities = similarities
        if self.similarities_quantization_mode == SimilaritiesQuantizationMode.NONE:
            pass
        elif self.similarities_quantization_mode == SimilaritiesQuantizationMode.DIFFERENTIABLE_ROUND:
            quantized_similarities = torch.round(similarities / self.similarities_bin_size) * self.similarities_bin_size
            quantized_similarities = similarities + (quantized_similarities - similarities).detach()
        elif self.similarities_quantization_mode == SimilaritiesQuantizationMode.SMOOTH_STEPS:
            quantized_similarities = quantized_similarities - torch.sin(2 * torch.pi * quantized_similarities) / (
                2 * torch.pi
            )
        elif self.similarities_quantization_mode == SimilaritiesQuantizationMode.CUSTOM_ROUND:
            quantized_similarities = cround(quantized_similarities, self.similarities_bin_size)
        elif self.similarities_quantization_mode == SimilaritiesQuantizationMode.CLUSTER:
            # kmeans = KMeans(n_clusters=self.similarities_num_clusters, random_state=0).fit(
            #     quantized_similarities.detach().cpu().numpy()
            # )
            # quantized_similarities = torch.as_tensor(
            #     kmeans.cluster_centers_[kmeans.predict(quantized_similarities.detach().cpu().numpy())]
            # )

            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=self.similarities_num_clusters
            ).fit_predict(quantized_similarities.detach().cpu().numpy())
            cluster_centers = torch.zeros((labels.max() + 1, x.shape[1]))

            for i in range(labels.max() + 1):
                cluster_centers[i] = quantized_similarities[labels == i].mean(0)

            quantized_similarities = torch.as_tensor(cluster_centers[labels])
        else:
            raise NotImplementedError()

        return {
            AttentionOutput.SIMILARITIES: quantized_similarities,
            AttentionOutput.NON_QUANTIZED_SIMILARITIES: similarities,
            AttentionOutput.ANCHORS: anchors,
            AttentionOutput.ORIGINAL_ANCHORS: original_anchors,
            AttentionOutput.ANCHORS_TARGETS: anchors_targets,
            AttentionOutput.ANCHORS_LATENT: anchors_latents,
            AttentionOutput.BATCH_LATENT: x_latents,
        }

    def decode(
        self,
        similarities,
        anchors: torch.Tensor,
        anchors_targets: Optional[torch.Tensor] = None,
        original_anchors: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Compute the weighted average of the values
        if self.values_mode == ValuesMethod.TRAINABLE:
            weights = F.softmax(similarities, dim=-1)
            output = torch.einsum("bw, wh -> bh", weights, self.values)
        elif self.values_mode == ValuesMethod.ANCHORS:
            weights = F.softmax(similarities, dim=-1)
            output = torch.einsum(
                "bw, wh -> bh",
                weights,
                self.pre_attention_transforms(anchors, element=AttentionElement.ATTENTION_VALUES),
            )
        elif self.values_mode == ValuesMethod.SIMILARITIES:
            output = similarities
        elif self.values_mode == ValuesMethod.SELF_ATTENTION:
            relative_output = self.sim_norm(similarities)

            queries = torch.einsum("ba, af -> baf", relative_output, self.sim_to_queries)
            keys = torch.einsum("ba, af -> baf", relative_output, self.sim_to_keys)
            values = torch.einsum("ba, af -> baf", relative_output, self.sim_to_values)

            output, _ = self.self_attention(query=queries, key=keys, value=values)
            output = self.self_attention_aggregation(output)

            output = output.squeeze(-1)
        elif self.values_mode == ValuesMethod.RELATIVE_ANCHORS_ATTENTION:
            weights = self.sim_norm(similarities)
            # weights = F.softmax(similarities, dim=-1)
            # weights = similarities

            relative_anchors = self.anchors_relative_attention(
                x=original_anchors,
                anchors=original_anchors,
                anchors_targets=anchors_targets,
            )[AttentionOutput.OUTPUT]

            output = torch.einsum("bw, wh -> bh", weights, relative_anchors)
        else:
            assert False

        # Normalize the output
        if self.output_normalization_mode == OutputNormalization.NONE:
            pass
        elif self.output_normalization_mode == OutputNormalization.L2:
            output = F.normalize(output, p=2, dim=-1)
        elif self.output_normalization_mode == OutputNormalization.BATCHNORM:
            output = self.outnorm(output)
        elif self.output_normalization_mode == OutputNormalization.LAYERNORM:
            output = self.outnorm(output)
        elif self.output_normalization_mode == OutputNormalization.INSTANCENORM:
            output = torch.einsum("lc -> cl", output)
            output = self.outnorm(output)
            output = torch.einsum("cl -> lc", output)
        else:
            assert False

        # TODO: This should also return the Anchors Targets tensor, because it could change depending on the parameters
        return {
            AttentionOutput.OUTPUT: output,
            AttentionOutput.UNTRASFORMED_ATTENDED: output,
            AttentionOutput.SIMILARITIES: similarities,
            AttentionOutput.NON_QUANTIZED_SIMILARITIES: kwargs[AttentionOutput.NON_QUANTIZED_SIMILARITIES],
            AttentionOutput.ANCHORS_LATENT: kwargs[AttentionOutput.ANCHORS_LATENT],
            AttentionOutput.BATCH_LATENT: kwargs[AttentionOutput.BATCH_LATENT],
        }

    def forward(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
        anchors_targets: Optional[torch.Tensor] = None,
    ) -> Dict[AttentionOutput, torch.Tensor]:
        """Forward pass.

        Args:
            x: [batch_size, hidden_dim]
            anchors: [num_anchors, hidden_dim]
            anchors_targets: [num_anchors]
        """
        encoding = self.encode(x=x, anchors=anchors, anchors_targets=anchors_targets)
        return self.decode(**encoding)

    @property
    def output_dim(self) -> int:
        if self.values_mode == ValuesMethod.ANCHORS:
            return self.in_features
        elif self.values_mode == ValuesMethod.TRAINABLE:
            return self.hidden_features
        elif (
            self.values_mode == ValuesMethod.SIMILARITIES
            or self.values_mode == ValuesMethod.SELF_ATTENTION
            or self.values_mode == ValuesMethod.RELATIVE_ANCHORS_ATTENTION
        ):
            if self.similarities_aggregation_mode == SimilaritiesAggregationMode.STRATIFIED_AVG:
                return self.n_classes * self.similarities_aggregation_n_groups
            elif (
                self.anchors_sampling_mode == AnchorsSamplingMode.STRATIFIED
                or self.anchors_sampling_mode == AnchorsSamplingMode.GAUSSIAN
            ):
                return self.n_classes * self.n_anchors_sampling_per_class
            else:
                return self.n_anchors
        else:
            raise ValueError(f"Values mode not supported: {self.values_mode}")


class RelativeTransformerBlock(AbstractRelativeAttention):
    def __init__(
        self,
        relative_attention: AbstractRelativeAttention,
        learning_block: nn.Module,
    ):
        """Relative Transformer block.

        Args:
            relative_attention: an instance of the RelativeAttention module
            learning_block: a learning block employed after the attention
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.relative_attention: AbstractRelativeAttention = (
            hydra.utils.instantiate(relative_attention)
            if not isinstance(relative_attention, AbstractRelativeAttention)
            else relative_attention
        )
        self.block: nn.Module = (
            hydra.utils.instantiate(learning_block, in_features=self.relative_attention.output_dim)
            if not isinstance(learning_block, nn.Module)
            else learning_block
        )

    def output_dim(self) -> int:
        return getattr(self.block, "out_features", self.relative_attention.output_dim)

    def encode(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
        anchors_targets: Optional[torch.Tensor] = None,
    ) -> Dict[AttentionOutput, torch.Tensor]:
        return self.relative_attention.encode(x=x, anchors=anchors, anchors_targets=anchors_targets)

    def decode(self, **kwargs):
        attention_output = self.relative_attention.decode(**kwargs)
        output = self.block(attention_output[AttentionOutput.OUTPUT])
        return {
            AttentionOutput.OUTPUT: output,
            AttentionOutput.UNTRASFORMED_ATTENDED: attention_output[AttentionOutput.UNTRASFORMED_ATTENDED],
            AttentionOutput.SIMILARITIES: kwargs[AttentionOutput.SIMILARITIES],
            AttentionOutput.NON_QUANTIZED_SIMILARITIES: kwargs[AttentionOutput.NON_QUANTIZED_SIMILARITIES],
            AttentionOutput.ANCHORS_LATENT: kwargs[AttentionOutput.ANCHORS_LATENT],
            AttentionOutput.BATCH_LATENT: kwargs[AttentionOutput.BATCH_LATENT],
        }

    def forward(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
        anchors_targets: Optional[torch.Tensor] = None,
    ) -> Dict[AttentionOutput, torch.Tensor]:
        encoding = self.encode(x=x, anchors=anchors, anchors_targets=anchors_targets)
        return self.decode(**encoding)


class MultiheadRelativeAttention(AbstractRelativeAttention):
    def __init__(
        self,
        in_features: int,
        relative_attentions: Sequence[AbstractRelativeAttention],
        subspace_pooling: SubspacePooling,
        n_anchors: Optional[int] = None,
        n_classes: Optional[int] = None,
        **kwargs,
    ):
        """MultiHead Relative Attention, apply the relative attention to embedding subspace.

        Args:
            relative_attentions: a sequence RelativeAttentions, one for each subspace
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.in_features = in_features
        self.num_subspaces = len(relative_attentions)

        if not self.num_subspaces:
            raise ValueError("Impossible to subdivide the embedding in zero subspaces!")

        self.subspace_in_features = in_features // self.num_subspaces
        assert (self.in_features / self.num_subspaces) == (self.in_features // self.num_subspaces)

        self.relative_attentions: ModuleList = nn.ModuleList(
            (
                hydra.utils.instantiate(
                    relative_attention,
                    in_features=self.subspace_in_features,
                    n_anchors=n_anchors,
                    n_classes=n_classes,
                )
                if not isinstance(relative_attention, AbstractRelativeAttention)
                else relative_attention
            )
            for relative_attention in relative_attentions
        )

        assert len(set(x.output_dim for x in self.relative_attentions)) == 1
        self.subspace_output_dim: int = list(self.relative_attentions)[0].output_dim

        self.subspace_pooling: SubspacePooling = (
            subspace_pooling if subspace_pooling is not None else SubspacePooling.NONE
        )

        if self.subspace_pooling not in set(SubspacePooling):
            raise ValueError(f"Representation Subspace Pooling method not supported: {subspace_pooling}")

        if self.subspace_pooling == SubspacePooling.LINEAR:
            self.head_pooling = nn.Linear(
                in_features=sum(x.output_dim for x in self.relative_attentions),
                out_features=self.subspace_output_dim,
            )

    @property
    def output_dim(self) -> int:
        if self.subspace_pooling == SubspacePooling.NONE:
            return sum(x.output_dim for x in self.relative_attentions)
        else:
            return self.subspace_output_dim

    def encode(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
        anchors_targets: Optional[torch.Tensor] = None,
    ) -> Dict[AttentionOutput, Union[torch.Tensor, Sequence[torch.Tensor]]]:
        subspace_outputs = []

        for i, relative_attention in enumerate(self.relative_attentions):
            x_i_subspace = x[:, i * self.subspace_in_features : (i + 1) * self.subspace_in_features]
            anchors_i_subspace = anchors[:, i * self.subspace_in_features : (i + 1) * self.subspace_in_features]
            subspace_output = relative_attention.encode(
                x=x_i_subspace,
                anchors=anchors_i_subspace,
                anchors_targets=anchors_targets,
            )
            subspace_outputs.append(subspace_output)

        return {AttentionOutput.SUBSPACE_OUTPUTS: subspace_outputs}

    def decode(
        self,
        subspace_outputs,
        **kwargs,
    ):
        decoded_attention = []
        for subspace_output, relative_attention in zip(subspace_outputs, self.relative_attentions):
            decoded_subspace = relative_attention.decode(**subspace_output)
            decoded_attention.append(decoded_subspace)

        attention_encodings = {
            key: [subspace[key] for subspace in subspace_outputs] for key in subspace_outputs[0].keys()
        }
        attention_output = {
            key: [subspace[key] for subspace in decoded_attention] for key in decoded_attention[0].keys()
        }

        for to_merge in [AttentionOutput.OUTPUT]:
            attention_output[to_merge] = torch.stack(attention_output[to_merge], dim=1)

        for to_merge in [
            AttentionOutput.SIMILARITIES,
            AttentionOutput.NON_QUANTIZED_SIMILARITIES,
            AttentionOutput.ANCHORS_LATENT,
            AttentionOutput.BATCH_LATENT,
        ]:
            attention_output[to_merge] = torch.stack(attention_encodings[to_merge], dim=1)

        if self.subspace_pooling == SubspacePooling.LINEAR:
            attention_output[AttentionOutput.OUTPUT] = torch.flatten(attention_output[AttentionOutput.OUTPUT], 1, 2)
            attention_output[AttentionOutput.OUTPUT] = self.head_pooling(attention_output[AttentionOutput.OUTPUT])
        elif self.subspace_pooling == SubspacePooling.MAX:
            attention_output[AttentionOutput.OUTPUT] = attention_output[AttentionOutput.OUTPUT].max(dim=1)[0]
        elif self.subspace_pooling == SubspacePooling.SUM:
            attention_output[AttentionOutput.OUTPUT] = attention_output[AttentionOutput.OUTPUT].sum(dim=1)
        elif self.subspace_pooling == SubspacePooling.MEAN:
            attention_output[AttentionOutput.OUTPUT] = attention_output[AttentionOutput.OUTPUT].mean(dim=1)
        elif self.subspace_pooling == SubspacePooling.NONE:
            attention_output[AttentionOutput.OUTPUT] = torch.flatten(attention_output[AttentionOutput.OUTPUT], 1, 2)
        else:
            raise NotImplementedError

        return {
            AttentionOutput.OUTPUT: attention_output[AttentionOutput.OUTPUT],
            AttentionOutput.UNTRASFORMED_ATTENDED: attention_output[AttentionOutput.UNTRASFORMED_ATTENDED],
            AttentionOutput.SIMILARITIES: attention_output[AttentionOutput.SIMILARITIES],
            AttentionOutput.NON_QUANTIZED_SIMILARITIES: attention_output[AttentionOutput.NON_QUANTIZED_SIMILARITIES],
            AttentionOutput.ANCHORS_LATENT: attention_output[AttentionOutput.ANCHORS_LATENT].squeeze(1),
            AttentionOutput.BATCH_LATENT: attention_output[AttentionOutput.BATCH_LATENT].squeeze(1),
        }

    def forward(
        self,
        x: torch.Tensor,
        anchors: torch.Tensor,
        anchors_targets: Optional[torch.Tensor] = None,
    ) -> Dict[AttentionOutput, torch.Tensor]:
        encoding = self.encode(x=x, anchors=anchors, anchors_targets=anchors_targets)
        return self.decode(**encoding)
