from pathlib import Path

import streamlit as st
import wandb

from nn_core.serialization import load_model
from nn_core.ui import select_checkpoint

from rae.pl_modules.vision.pl_gautoencoder import LightningAutoencoder


@st.cache(allow_output_mutation=True)
def get_model(checkpoint_path: Path):
    return load_model(module_class=LightningAutoencoder, checkpoint_path=checkpoint_path)


if wandb.api.api_key is None:
    st.error("You are not logged in on `Weights and Biases`: https://docs.wandb.ai/ref/cli/wandb-login")
    st.stop()

st.sidebar.subheader(f"Logged in W&B as: {wandb.api.viewer()['entity']}")

checkpoint_path = select_checkpoint()
model: LightningAutoencoder = get_model(checkpoint_path=checkpoint_path)
model
