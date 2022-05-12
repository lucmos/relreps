from pathlib import Path

import streamlit as st
import torchvision
import wandb
from matplotlib import pyplot as plt

from nn_core.serialization import load_model
from nn_core.ui import select_checkpoint

from rae.modules.enumerations import Output
from rae.pl_modules.pl_gae import LightningGAE

plt.style.use("ggplot")


@st.cache(allow_output_mutation=True)
def get_model(checkpoint_path: Path):
    model = load_model(module_class=LightningGAE, checkpoint_path=checkpoint_path, map_location="cpu")
    model.eval()
    return model


def plot_image(img):
    fig, ax = plt.subplots(1, 1)
    ax.axis("off")
    ax.imshow(torchvision.utils.make_grid(img.cpu(), 1, 1).permute(1, 2, 0))
    return fig


if wandb.api.api_key is None:
    st.error("You are not logged in on `Weights and Biases`: https://docs.wandb.ai/ref/cli/wandb-login")
    st.stop()

st.sidebar.subheader(f"Logged in W&B as: {wandb.api.viewer()['entity']}")

checkpoint_path = select_checkpoint(default_run_path="gladia/rae/3la7i1zj")
model: LightningGAE = get_model(checkpoint_path=checkpoint_path)


st.sidebar.markdown("---")
images = model.metadata.fixed_images.cpu().detach()
fixed_image_idx = st.sidebar.slider("Select sample image:", 0, max_value=images.shape[0])
image = images[fixed_image_idx]


set_zero = st.sidebar.checkbox("Set to zero")

# Compute the relative embeddings...
relative_embedding = model(image[None, ...])[Output.INV_LATENTS]

# Only for visualization limits
min_value, max_value = relative_embedding.detach().min(), relative_embedding.detach().max()


# Visualize the sample and the reconstruction
_, col1, col2, _ = st.columns(4)
with col1:
    st.subheader("Sample")
    st.pyplot(
        plot_image(image),
        clear_figure=True,
    )
with col2:
    st.subheader("Reconstruction")
    reconstruction_plot = st.empty()

# Visualize for each anchor: image, relative value, widget to modify the value
for i_col, relative_value in enumerate(relative_embedding.squeeze()):
    value = relative_value.detach().item()

    if set_zero:
        value = 0.0

    col1, col2, col3 = st.columns([1.5, 7, 3])
    with col1:
        anchor_image = model.metadata.anchors_images[i_col]
        st.pyplot(plot_image(anchor_image), clear_figure=True)

    with col3:
        value = st.number_input("", min_value=-100.0, max_value=100.0, step=0.1, value=value, key=f"{i_col}")

    with col2:
        fig, ax = plt.subplots(1, 1, figsize=(5, 1), dpi=120)
        fig.set_tight_layout(tight=True)
        ax.barh([0], [value], height=1, align="center")
        ax.set_xlim(min_value, max_value)
        st.pyplot(fig)

    # Update the relative embedding
    relative_embedding[..., i_col] = value


# Reconstruct the image with the updated embedding
reconstruction, _ = model.autoencoder.decoder(relative_embedding=relative_embedding)
reconstruction_plot.pyplot(
    plot_image(reconstruction),
    clear_figure=True,
)
