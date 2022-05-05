from enum import auto

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class Output(StrEnum):
    RECONSTRUCTION = auto()
    DEFAULT_LATENT = auto()
    BATCH_LATENT = auto()
    LATENT_MU = auto()
    LATENT_LOGVAR = auto()
    ANCHORS_LATENT = auto()
    INV_LATENTS = auto()
    LOSS = auto()
    BATCH = auto()


class SupportedViz(StrEnum):
    ANCHORS_SOURCE = auto()
    ANCHORS_RECONSTRUCTED = auto()
    LATENT_EVOLUTION = auto()
    VALIDATION_IMAGES_SOURCE = auto()
    VALIDATION_IMAGES_RECONSTRUCTED = auto()
    ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT = auto()
    ANCHORS_SELF_INNER_PRODUCT = auto()
    ANCHORS_VALIDATION_IMAGES_INNER_PRODUCT_NORMALIZED = auto()
    ANCHORS_SELF_INNER_PRODUCT_NORMALIZED = auto()
    INVARIANT_LATENT_DISTRIBUTION = auto()
