import numpy as np
import streamlit as st
import torch
import wandb
from matplotlib import pyplot as plt

from nn_core.ui import select_checkpoint

from rae.modules.enumerations import Output
from rae.pl_modules.vision.pl_gautoencoder import LightningAutoencoder
from rae.ui.ui_utils import check_wandb_login, get_model, plot_image, show_code_version

plt.style.use("ggplot")


check_wandb_login()


CODE_VERSION = "0.0.1"
show_code_version(code_version=CODE_VERSION)

# Load model
st.sidebar.subheader(f"Logged in W&B as: {wandb.api.viewer()['entity']}")
checkpoint_path = select_checkpoint(default_run_path="gladia/rae/356rslt5")
model: LightningAutoencoder = get_model(checkpoint_path=checkpoint_path, supported_code_version=CODE_VERSION)

# Select a sample
st.sidebar.markdown("---")
images = model.metadata.fixed_images.cpu().detach()
fixed_image_idx = st.sidebar.slider("Select sample image:", 0, max_value=images.shape[0])
image = images[fixed_image_idx]

# Compute the relative embeddings...
relative_embedding = model(image[None, ...])[Output.INV_LATENTS].detach().numpy()

# Bar plots limits
lim = max(1, max(abs(relative_embedding.min()), abs(relative_embedding.max())) + 0.5)

# Maintain user changes between page refreshes
st.sidebar.markdown("---")
col1, col2 = st.sidebar.columns(2)
reset_latents = col1.button("Reset latents")
zero_latents = col2.button("Zero latents")

sample_key = f"sample{fixed_image_idx}"
if sample_key not in st.session_state:
    st.session_state[sample_key] = relative_embedding

if reset_latents:
    st.session_state[sample_key] = relative_embedding

if zero_latents:
    st.session_state[sample_key] = np.zeros_like(relative_embedding)


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


def on_change(sample_key: str, col: int) -> None:
    # Update the relative embedding in col with the value of the widget wih key col
    st.session_state[sample_key][..., col] = st.session_state[f"{col}"]


# Visualize for each anchor
for i_col in range(st.session_state[sample_key].shape[-1]):
    # Get the relative value
    value = st.session_state[sample_key][..., i_col].item()

    col1, col2, col3 = st.columns([1.5, 7, 3])
    # The image
    with col1:
        anchor_image = model.metadata.anchors_images[i_col]
        st.pyplot(plot_image(anchor_image), clear_figure=True)

    # A widget to modify the measure w.r.t this anchor
    with col3:

        value = st.number_input(
            "",
            min_value=-100.0,
            max_value=100.0,
            step=0.1,
            value=value,
            key=f"{i_col}",
            on_change=on_change,
            args=(sample_key, i_col),
        )

    # The measure w.r.t this anchor, compute from the sample or modified by the user
    with col2:
        fig, ax = plt.subplots(1, 1, figsize=(5, 1), dpi=120)
        fig.set_tight_layout(tight=True)
        ax.barh([0], [value], height=1, align="center")
        ax.set_xlim(-lim, lim)
        st.pyplot(fig)


# Reconstruct the image with the updated embedding
st.markdown("---")
reconstruction, _ = model.autoencoder.decoder(relative_embedding=torch.as_tensor(st.session_state[sample_key]))
reconstruction_plot.pyplot(
    plot_image(reconstruction),
    clear_figure=True,
)
