import streamlit as st
import wandb
from matplotlib import pyplot as plt

from nn_core.ui import select_checkpoint

from rae.modules.enumerations import Output
from rae.pl_modules.pl_gae import LightningGAE
from rae.ui.ui_utils import check_wandb_login, get_model, plot_image

plt.style.use("ggplot")
st.set_page_config(layout="centered")

slider_placeholder = st.sidebar.empty()
check_wandb_login()
st.sidebar.subheader(f"Logged in W&B as: {wandb.api.viewer()['entity']}")

st.sidebar.header("RAE checkpoints")
rae_encoder_ckpt = select_checkpoint(default_run_path="gladia/rae/356rslt5")
rae_encoder: LightningGAE = get_model(checkpoint_path=rae_encoder_ckpt)
st.sidebar.markdown("---")
rae_decoder_ckpt = select_checkpoint(default_run_path="gladia/rae/1f1u0s7r")
rae_decoder: LightningGAE = get_model(checkpoint_path=rae_decoder_ckpt)
st.sidebar.markdown("---")

images = rae_encoder.metadata.fixed_images.cpu().detach()
fixed_image_idx = slider_placeholder.slider("Select sample image:", 0, max_value=images.shape[0], value=2)
image = images[fixed_image_idx]

st.subheader("RAE")
model_out = rae_encoder(image[None])
batch_latent = model_out[Output.BATCH_LATENT]
anchors_latent = model_out[Output.ANCHORS_LATENT]

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("Source")
    source_plot = plot_image(image)
    st.pyplot(source_plot)

with col2:
    st.markdown("First Decoder")
    reconstruction, _ = rae_encoder.autoencoder.decoder(batch_latent, anchors_latent)
    st.pyplot(plot_image(reconstruction))

with col3:
    st.markdown("Second Decoder")
    reconstruction, _ = rae_decoder.autoencoder.decoder(batch_latent, anchors_latent)
    st.pyplot(plot_image(reconstruction))

st.sidebar.subheader("VAE checkpoints")
vae_encoder_ckpt = select_checkpoint(default_run_path="gladia/rae/3ufahj5a")
vae_encoder: LightningGAE = get_model(checkpoint_path=vae_encoder_ckpt)
st.sidebar.markdown("---")
vae_decoder_ckpt = select_checkpoint(default_run_path="gladia/rae/9ciiajda")
vae_decoder: LightningGAE = get_model(checkpoint_path=vae_decoder_ckpt)
st.sidebar.markdown("---")


st.subheader("VAE")
model_out = vae_encoder(image[None])
batch_latent_mu = model_out[Output.LATENT_MU]

col1, col2, col3 = st.columns(3)
with col1:
    st.pyplot(source_plot)
with col2:
    reconstruction = vae_encoder.autoencoder.decoder(batch_latent_mu)
    st.pyplot(plot_image(reconstruction))
with col3:
    reconstruction = vae_decoder.autoencoder.decoder(batch_latent_mu)
    st.pyplot(plot_image(reconstruction))


st.sidebar.subheader("AE checkpoints")
ae_encoder_ckpt = select_checkpoint(default_run_path="gladia/rae/3a9iwpmo")
ae_encoder: LightningGAE = get_model(checkpoint_path=ae_encoder_ckpt)
st.sidebar.markdown("---")
ae_decoder_ckpt = select_checkpoint(default_run_path="gladia/rae/16tamf2p")
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
with col3:
    reconstruction = ae_decoder.autoencoder.decoder(batch_latent)
    st.pyplot(plot_image(reconstruction))
