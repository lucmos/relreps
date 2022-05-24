from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from nn_core.ui import select_checkpoint

from rae.modules.enumerations import Output
from rae.pl_modules.pl_gae import LightningGAE
from rae.ui.ui_utils import check_wandb_login, get_model, plot_image

plt.style.use("ggplot")
st.set_page_config(layout="wide")

slider_placeholder = st.sidebar.empty()
visualize_latent_space = st.sidebar.checkbox("Visualize latent space")
visualize_decoder_weights_diff = st.sidebar.checkbox("Visualize decoder weights diff")

check_wandb_login()
st.sidebar.subheader(f"Logged in W&B as: {wandb.api.viewer()['entity']}")


def display_latent(st_container, metadata, model, pca: Optional[PCA] = None):
    model_out = model(images)
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


def compute_weights_difference(m1, m2):
    wdif = sum((x - y).abs().sum().detach().item() for x, y in zip(m1.parameters(), m2.parameters())) / sum(
        p.numel() for p in rae_encoder.autoencoder.decoder.parameters()
    )
    st.metric("D1 - D2 weights", f"{wdif:.4f}")


with torch.no_grad():
    st.sidebar.header("RAE checkpoints")
    rae_encoder_ckpt = select_checkpoint(st_key="rae_encoder_ckpt", default_run_path="gladia/rae/2c3w6plr")
    rae_encoder: LightningGAE = get_model(checkpoint_path=rae_encoder_ckpt)
    st.sidebar.markdown("---")
    rae_decoder_ckpt = select_checkpoint(st_key="rae_decoder_ckpt", default_run_path="gladia/rae/l1rnvm2u")
    rae_decoder: LightningGAE = get_model(checkpoint_path=rae_decoder_ckpt)
    st.sidebar.markdown("---")

    metadata = rae_encoder.metadata
    color_discrete_map = {
        class_name: color
        for class_name, color in zip(metadata.class_to_idx, px.colors.qualitative.Plotly[: len(metadata.class_to_idx)])
    }
    images = metadata.fixed_images.cpu().detach()
    fixed_image_idx = slider_placeholder.number_input("Select sample image:", 0, max_value=images.shape[0], value=2)
    image = images[fixed_image_idx]

    st.subheader("RAE")
    model_out = rae_encoder(image[None])
    batch_latent = model_out[Output.BATCH_LATENT]
    anchors_latent = model_out[Output.ANCHORS_LATENT]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("Source (E1)")
        source_plot = plot_image(image)
        st.pyplot(source_plot)

    with col2:
        st.markdown("Decoder (D1)")
        reconstruction, _ = rae_encoder.autoencoder.decoder(batch_latent, anchors_latent)
        st.pyplot(plot_image(reconstruction))

        if visualize_latent_space:
            pca = display_latent(st_container=col2, metadata=metadata, model=rae_encoder, pca=None)

    with col3:
        st.markdown("Decoder (D2)")
        reconstruction, _ = rae_decoder.autoencoder.decoder(batch_latent, anchors_latent)
        st.pyplot(plot_image(reconstruction))

        if visualize_latent_space:
            pca = display_latent(st_container=col3, metadata=metadata, model=rae_decoder, pca=None)

    st.sidebar.subheader("VAE checkpoints")
    vae_encoder_ckpt = select_checkpoint(st_key="vae_encoder_ckpt", default_run_path="gladia/rae/3ufahj5a")
    vae_encoder: LightningGAE = get_model(checkpoint_path=vae_encoder_ckpt)
    st.sidebar.markdown("---")
    vae_decoder_ckpt = select_checkpoint(st_key="vae_decoder_ckpt", default_run_path="gladia/rae/24d608t3")
    vae_decoder: LightningGAE = get_model(checkpoint_path=vae_decoder_ckpt)
    st.sidebar.markdown("---")

    if visualize_decoder_weights_diff:
        compute_weights_difference(rae_encoder.autoencoder.decoder, rae_decoder.autoencoder.decoder)

    st.subheader("VAE")
    model_out = vae_encoder(image[None])
    batch_latent_mu = model_out[Output.LATENT_MU]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(source_plot)
    with col2:
        reconstruction = vae_encoder.autoencoder.decoder(batch_latent_mu)
        st.pyplot(plot_image(reconstruction))

        if visualize_latent_space:
            pca = display_latent(st_container=col2, metadata=metadata, model=vae_encoder, pca=None)

    with col3:
        reconstruction = vae_decoder.autoencoder.decoder(batch_latent_mu)
        st.pyplot(plot_image(reconstruction))

        if visualize_latent_space:
            pca = display_latent(st_container=col3, metadata=metadata, model=vae_decoder, pca=None)

    if visualize_decoder_weights_diff:
        compute_weights_difference(vae_encoder.autoencoder.decoder, vae_decoder.autoencoder.decoder)

    st.sidebar.subheader("AE checkpoints")
    ae_encoder_ckpt = select_checkpoint(st_key="ae_encoder_ckpt", default_run_path="gladia/rae/3a9iwpmo")
    ae_encoder: LightningGAE = get_model(checkpoint_path=ae_encoder_ckpt)
    st.sidebar.markdown("---")
    ae_decoder_ckpt = select_checkpoint(st_key="ae_decoder_ckpt", default_run_path="gladia/rae/16tamf2p")
    ae_decoder: LightningGAE = get_model(checkpoint_path=ae_decoder_ckpt)
    st.sidebar.markdown("---")

    st.subheader("AE")
    model_out = ae_encoder(image[None])
    batch_latent = model_out[Output.BATCH_LATENT]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.pyplot(source_plot)
    with col2:
        reconstruction = ae_encoder.autoencoder.decoder(batch_latent)
        st.pyplot(plot_image(reconstruction))

        if visualize_latent_space:
            pca = display_latent(st_container=col2, metadata=metadata, model=ae_encoder, pca=None)

    with col3:
        reconstruction = ae_decoder.autoencoder.decoder(batch_latent)
        st.pyplot(plot_image(reconstruction))

        if visualize_latent_space:
            pca = display_latent(st_container=col3, metadata=metadata, model=rae_decoder, pca=None)

    if visualize_decoder_weights_diff:
        compute_weights_difference(ae_encoder.autoencoder.decoder, ae_decoder.autoencoder.decoder)
