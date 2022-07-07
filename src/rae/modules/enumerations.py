from enum import auto

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class SimilaritiesAggregationMode(StrEnum):
    STRATIFIED_AVG = auto()


class SimilaritiesQuantizationMode(StrEnum):
    DIFFERENTIABLE_ROUND = auto()
    # SMOOTH_STEPS = auto()


class NormalizationMode(StrEnum):
    L2 = auto()
    OFF = auto()
    # BATCHNORM = auto()
    # INSTANCENORM = auto()
    # LAYERNORM = auto()
    # INSTANCENORM_NOAFFINE = auto()
    # LAYERNORM_NOAFFINE = auto()


class RelativeEmbeddingMethod(StrEnum):
    BASIS_CHANGE = auto()
    INNER = auto()


class ValuesMethod(StrEnum):
    SIMILARITIES = auto()
    TRAINABLE = auto()
    ANCHORS = auto()


class Output(StrEnum):
    LOGITS = auto()
    RECONSTRUCTION = auto()
    DEFAULT_LATENT = auto()
    DEFAULT_LATENT_NORMALIZED = auto()
    BATCH_LATENT = auto()
    LATENT_MU = auto()
    LATENT_LOGVAR = auto()
    ANCHORS_LATENT = auto()
    INV_LATENTS = auto()
    LOSS = auto()
    BATCH = auto()


class AttentionOutput(StrEnum):
    OUTPUT = auto()
    SIMILARITIES = auto()
    UNTRASFORMED_ATTENDED = auto()


class SupportedViz(StrEnum):
    ANCHORS_SOURCE = auto()
    ANCHORS_RECONSTRUCTED = auto()
    LATENT_EVOLUTION_PLOTLY_ANIMATION = auto()
    LATENT_SPACE = auto()
    LATENT_SPACE_PCA = auto()
    LATENT_SPACE_NORMALIZED = auto()
    VALIDATION_IMAGES_SOURCE = auto()
    VALIDATION_IMAGES_RECONSTRUCTED = auto()
    ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT = auto()
    ANCHORS_SELF_INNER_PRODUCT = auto()
    ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT_NORMALIZED = auto()
    ANCHORS_SELF_INNER_PRODUCT_NORMALIZED = auto()
    INVARIANT_LATENT_DISTRIBUTION = auto()


class Stage(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()
