from enum import auto

try:
    # be ready for 3.10 when it drops
    from enum import StrEnum
except ImportError:
    from backports.strenum import StrEnum


class Output(StrEnum):
    OUT = auto()
    DEFAULT_LATENT = auto()
    LATENT = auto()
    LATENT_MU = auto()
    LATENT_LOGVAR = auto()
