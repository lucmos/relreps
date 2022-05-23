from pathlib import Path

import streamlit as st
import torchvision
import wandb
from matplotlib import pyplot as plt

from nn_core.serialization import load_model

from rae.pl_modules.pl_gae import LightningGAE


def plot_image(img):
    fig, ax = plt.subplots(1, 1)
    ax.axis("off")
    ax.imshow(torchvision.utils.make_grid(img.cpu(), 1, 1).permute(1, 2, 0))
    return fig


@st.cache(allow_output_mutation=True)
def get_model(checkpoint_path: Path):
    model = load_model(module_class=LightningGAE, checkpoint_path=checkpoint_path, map_location="cpu")
    model.eval()
    return model


def check_wandb_login():
    if wandb.api.api_key is None:
        st.error("You are not logged in on `Weights and Biases`: https://docs.wandb.ai/ref/cli/wandb-login")
        st.stop()
