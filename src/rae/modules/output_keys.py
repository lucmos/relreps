from enum import Enum, auto


class Output(Enum):
    OUT = auto()
    DEFAULT_LATENT = auto()
    LATENT = auto()
    LATENT_MU = auto()
    LATENT_LOGVAR = auto()
