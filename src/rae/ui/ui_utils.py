from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import torchvision
import wandb
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from nn_core.serialization import NNCheckpointIO, load_model

from rae.modules.enumerations import Output
from rae.pl_modules.pl_gae import LightningGAE
from rae.utils.plotting import plot_violin


def show_code_version(code_version: str):
    st.sidebar.markdown(f"Demo compatibility up to version: `{code_version}`")
    st.sidebar.markdown("---")


def compute_weights_difference(m1, m2):
    wdif = sum((x - y).abs().sum().detach().item() for x, y in zip(m1.parameters(), m2.parameters())) / sum(
        p.numel() for p in m1.parameters()
    )
    st.metric("D1 - D2 weights", f"{wdif:.4f}")


def plot_image(img):
    fig, ax = plt.subplots(1, 1)
    ax.axis("off")
    ax.imshow(torchvision.utils.make_grid(img.cpu(), 1, 1).permute(1, 2, 0))
    return fig


@st.cache(allow_output_mutation=True)
def get_model(checkpoint_path: Path, supported_code_version: str):
    try:
        model = load_model(
            module_class=LightningGAE,
            checkpoint_path=checkpoint_path,
            map_location="cpu",
            substitute_values={"rae.modules.rae.RAE": "rae.modules.rae_model.RAE"},
        )
        model.eval()
        return model
    except:  # noqa
        ckpt_code_version = NNCheckpointIO.load(path=checkpoint_path, map_location="cpu")["cfg"]["core"]["version"]
        st.error(
            f"Codebase version mismatch. Checkpoint trained with version `{ckpt_code_version}` support version `{supported_code_version}"
        )
        st.stop()
        return None


def check_wandb_login():
    if wandb.api.api_key is None:
        st.error("You are not logged in on `Weights and Biases`: https://docs.wandb.ai/ref/cli/wandb-login")
        st.stop()


def display_latent(st_container, metadata, model_out, pca: Optional[PCA] = None):
    color_discrete_map = {
        class_name: color
        for class_name, color in zip(metadata.class_to_idx, px.colors.qualitative.Plotly[: len(metadata.class_to_idx)])
    }
    if pca is None:
        pca = PCA(n_components=2)
        pca.fit(model_out[Output.DEFAULT_LATENT])

    latents = pca.transform(model_out[Output.DEFAULT_LATENT])

    fig = px.scatter(
        pd.DataFrame(
            {
                "image_index": metadata.fixed_images_idxs,
                "class": metadata.fixed_images_classes,
                "target": metadata.fixed_images_targets,
                "latent_0": latents[:, 0],
                "latent_1": latents[:, 1],
            }
        ),
        x="latent_0",
        y="latent_1",
        category_orders={"class": metadata.class_to_idx.keys()},
        color="class",
        hover_name="image_index",
        hover_data=["image_index"],
        color_discrete_map=color_discrete_map,
        size_max=40,
        color_continuous_scale=None,
        labels={"latent_1": "", "latent_0": ""},
    )
    fig.layout.showlegend = False
    st_container.plotly_chart(
        fig,
        use_container_width=True,
    )
    return pca


def display_distributions(st_container, model_out):
    st_container.pyplot(
        plot_violin(
            model_out[Output.INV_LATENTS],
            title="Relative Latent Space distribution",
            y_label="validation distribution",
            x_label="anchors",
        )
    )
