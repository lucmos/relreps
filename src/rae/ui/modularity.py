import streamlit as st
import torch
import wandb
from matplotlib import pyplot as plt
from numerize.numerize import numerize
from torch.nn.functional import mse_loss

from nn_core.ui import select_checkpoint

from rae.modules.enumerations import Output
from rae.pl_modules.vision.pl_gautoencoder import LightningAutoencoder
from rae.ui.ui_utils import (
    check_wandb_login,
    compute_weights_difference,
    display_distributions,
    display_latent,
    get_model,
    plot_image,
    show_code_version,
)

plt.style.use("ggplot")
st.set_page_config(layout="wide")

CODE_VERSION = "0.0.1"
show_code_version(code_version=CODE_VERSION)

visualize_latent_space = st.sidebar.checkbox("Visualize latent space")
visualize_relative_distribution = st.sidebar.checkbox("Visualize relative distribution")
visualize_stats = st.sidebar.checkbox("Visualize stats", value=True)

vae_enabled = st.sidebar.checkbox("Enable VAE", value=True)
ae_enabled = st.sidebar.checkbox("Enable AE", value=True)

check_wandb_login()
st.sidebar.subheader(f"Logged in W&B as: {wandb.api.viewer()['entity']}")


with torch.no_grad():
    st.sidebar.header("RAE checkpoints")
    rae_1_ckpt = select_checkpoint(st_key="rae_1_ckpt", default_run_path="gladia/rae/2c3w6plr")
    rae_1: LightningAutoencoder = get_model(checkpoint_path=rae_1_ckpt, supported_code_version=CODE_VERSION)
    st.sidebar.markdown("---")
    rae_2_ckpt = select_checkpoint(st_key="rae_2_ckpt", default_run_path="gladia/rae/l1rnvm2u")
    rae_2: LightningAutoencoder = get_model(checkpoint_path=rae_2_ckpt, supported_code_version=CODE_VERSION)
    st.sidebar.markdown("---")

    metadata = rae_1.metadata

    images = metadata.fixed_images.cpu().detach()
    fixed_image_idx = int(st.number_input("Select sample image:", 0, max_value=images.shape[0] - 1, value=2))
    image = images[fixed_image_idx]

    st.subheader("RAE")
    model_out = rae_1(image[None])
    batch_latent = model_out[Output.BATCH_LATENT]
    anchors_latent = model_out[Output.ANCHORS_LATENT]

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown("Source (E1)")
        source_plot = plot_image(image)
        st.pyplot(source_plot)

    with col2:
        st.markdown("Decoder (D1)")
        reconstruction_d1, _ = rae_1.autoencoder.decoder(batch_latent, anchors_latent)
        st.pyplot(plot_image(reconstruction_d1))

        if visualize_relative_distribution or visualize_latent_space or visualize_stats:
            model_out_d1 = rae_1(images)

        if visualize_relative_distribution:
            display_distributions(st_container=col2, model_out=model_out_d1)

        if visualize_latent_space:
            pca = display_latent(st_container=col2, metadata=metadata, model_out=model_out_d1, pca=None)

    with col3:
        st.markdown("Decoder (D2)")
        reconstruction_d2, _ = rae_2.autoencoder.decoder(batch_latent, anchors_latent)
        st.pyplot(plot_image(reconstruction_d2))

        if visualize_relative_distribution or visualize_latent_space or visualize_stats:
            model_out_d2 = rae_2(images)

        if visualize_relative_distribution:
            display_distributions(st_container=col3, model_out=model_out_d2)

        if visualize_latent_space:
            pca = display_latent(st_container=col3, metadata=metadata, model_out=model_out_d2, pca=None)

    if visualize_stats:
        with col4:
            st.markdown("**Ground truth**")
            st.metric(
                "D1 gt: mse(D1(E1(50samples)), 50samples)",
                numerize(mse_loss(model_out_d1[Output.RECONSTRUCTION], images).item(), decimals=5),
            )
            st.metric(
                "D2 gt: mse(D2(E2(50samples)), 50samples)",
                numerize(mse_loss(model_out_d2[Output.RECONSTRUCTION], images).item(), decimals=5),
            )

        with col5:
            st.markdown("**Invariance**")
            st.metric(
                "Invariance Decoder: mse(D1(E1(source)), D2(E1(source)))",
                numerize(mse_loss(reconstruction_d1, reconstruction_d2).item(), decimals=5),
            )
            st.metric(
                "Invariance Decoder: mse(D1(E1(50samples)), D2(E1(50samples)))",
                numerize(
                    mse_loss(
                        model_out_d1[Output.RECONSTRUCTION],
                        rae_2.autoencoder.decoder(
                            model_out_d1[Output.BATCH_LATENT], model_out_d1[Output.ANCHORS_LATENT]
                        )[0],
                    ).item(),
                    decimals=5,
                ),
            )
            st.metric(
                "Invariance Decoder: mse(D1(E2(50samples)), D2(E2(50samples)))",
                numerize(
                    mse_loss(
                        rae_1.autoencoder.decoder(
                            model_out_d2[Output.BATCH_LATENT], model_out_d2[Output.ANCHORS_LATENT]
                        )[0],
                        model_out_d2[Output.RECONSTRUCTION],
                    ).item(),
                    decimals=5,
                ),
            )
            st.metric(
                "Invariance Encoder: mse(D1(E1(50samples)), D1(E2(50samples)))",
                numerize(
                    mse_loss(
                        model_out_d1[Output.RECONSTRUCTION],
                        rae_1.autoencoder.decoder(
                            model_out_d2[Output.BATCH_LATENT], model_out_d2[Output.ANCHORS_LATENT]
                        )[0],
                    ).item(),
                    decimals=5,
                ),
            )
            st.metric(
                "Invariance Encoder: mse(D2(E1(50samples)), D2(E2(50samples)))",
                numerize(
                    mse_loss(
                        rae_2.autoencoder.decoder(
                            model_out_d1[Output.BATCH_LATENT], model_out_d1[Output.ANCHORS_LATENT]
                        )[0],
                        model_out_d2[Output.RECONSTRUCTION],
                    ).item(),
                    decimals=5,
                ),
            )
        with col6:
            st.markdown("**Model1 vs Model2**")

            st.metric(
                "RAE1 vs RAE2: mse(D1(E1(50samples)), D2(E2(50samples)))",
                numerize(
                    mse_loss(model_out_d1[Output.RECONSTRUCTION], model_out_d2[Output.RECONSTRUCTION]).item(),
                    decimals=5,
                ),
            )

            st.metric(
                "Distances errors: mse(distances_E1, distances_E2)",
                numerize(
                    mse_loss(model_out_d1[Output.INV_LATENTS], model_out_d2[Output.INV_LATENTS]).item(), decimals=5
                ),
            )
            compute_weights_difference(rae_1.autoencoder.decoder, rae_2.autoencoder.decoder)

    if vae_enabled:

        st.sidebar.subheader("VAE checkpoints")
        vae_1_ckpt = select_checkpoint(st_key="vae_1_ckpt", default_run_path="gladia/rae/3ufahj5a")
        vae_1: LightningAutoencoder = get_model(checkpoint_path=vae_1_ckpt, supported_code_version=CODE_VERSION)
        st.sidebar.markdown("---")
        vae_2_ckpt = select_checkpoint(st_key="vae_2_ckpt", default_run_path="gladia/rae/24d608t3")
        vae_2: LightningAutoencoder = get_model(checkpoint_path=vae_2_ckpt, supported_code_version=CODE_VERSION)
        st.sidebar.markdown("---")

        st.subheader("VAE")
        model_out = vae_1(image[None])
        batch_latent_mu = model_out[Output.LATENT_MU]

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.pyplot(source_plot)
        with col2:
            reconstruction = vae_1.autoencoder.decoder(batch_latent_mu)
            st.pyplot(plot_image(reconstruction))

            model_out_d1 = vae_1(images)

            if visualize_latent_space:
                pca = display_latent(st_container=col2, metadata=metadata, model_out=model_out_d1, pca=None)

        with col3:
            reconstruction = vae_2.autoencoder.decoder(batch_latent_mu)
            st.pyplot(plot_image(reconstruction))

            model_out_d2 = vae_2(images)

            if visualize_latent_space:
                pca = display_latent(st_container=col3, metadata=metadata, model_out=model_out_d2, pca=None)

        if visualize_stats:
            with col4:
                st.markdown("**Ground truth**")
                st.metric(
                    "D1 gt: mse(D1(E1(50samples)), 50samples)",
                    numerize(mse_loss(model_out_d1[Output.RECONSTRUCTION], images).item(), decimals=5),
                )
                st.metric(
                    "D2 gt: mse(D2(E2(50samples)), 50samples)",
                    numerize(mse_loss(model_out_d2[Output.RECONSTRUCTION], images).item(), decimals=5),
                )

            with col5:
                st.markdown("**Invariance**")
                st.metric(
                    "Invariance Decoder: mse(D1(E1(source)), D2(E1(source)))",
                    numerize(mse_loss(reconstruction_d1, reconstruction_d2).item(), decimals=5),
                )
                st.metric(
                    "Invariance Decoder: mse(D1(E1(50samples)), D2(E1(50samples)))",
                    numerize(
                        mse_loss(
                            model_out_d1[Output.RECONSTRUCTION],
                            vae_2.autoencoder.decoder(model_out_d1[Output.BATCH_LATENT]),
                        ).item(),
                        decimals=5,
                    ),
                )
                st.metric(
                    "Invariance Decoder: mse(D1(E2(50samples)), D2(E2(50samples)))",
                    numerize(
                        mse_loss(
                            model_out_d2[Output.RECONSTRUCTION],
                            vae_1.autoencoder.decoder(model_out_d2[Output.BATCH_LATENT]),
                        ).item(),
                        decimals=5,
                    ),
                )
                st.metric(
                    "Invariance Encoder: mse(D1(E1(50samples)), D1(E2(50samples)))",
                    numerize(
                        mse_loss(
                            model_out_d1[Output.RECONSTRUCTION],
                            vae_1.autoencoder.decoder(model_out_d2[Output.BATCH_LATENT]),
                        ).item(),
                        decimals=5,
                    ),
                )
                st.metric(
                    "Invariance Encoder: mse(D2(E1(50samples)), D2(E2(50samples)))",
                    numerize(
                        mse_loss(
                            vae_2.autoencoder.decoder(model_out_d1[Output.BATCH_LATENT]),
                            model_out_d2[Output.RECONSTRUCTION],
                        ).item(),
                        decimals=5,
                    ),
                )
            with col6:
                st.markdown("**Model1 vs Model2**")

                st.metric(
                    "RAE1 vs RAE2: mse(D1(E1(50samples)), D2(E2(50samples)))",
                    numerize(
                        mse_loss(model_out_d1[Output.RECONSTRUCTION], model_out_d2[Output.RECONSTRUCTION]).item(),
                        decimals=5,
                    ),
                )

                st.metric(
                    "Distances errors: mse(distances_E1, distances_E2)",
                    numerize(
                        mse_loss(model_out_d1[Output.DEFAULT_LATENT], model_out_d2[Output.DEFAULT_LATENT]).item(),
                        decimals=5,
                    ),
                )
                compute_weights_difference(vae_1.autoencoder.decoder, vae_2.autoencoder.decoder)

    if ae_enabled:

        st.sidebar.subheader("AE checkpoints")
        ae_1_ckpt = select_checkpoint(st_key="ae_1_ckpt", default_run_path="gladia/rae/3a9iwpmo")
        ae_1: LightningAutoencoder = get_model(checkpoint_path=ae_1_ckpt, supported_code_version=CODE_VERSION)
        st.sidebar.markdown("---")
        ae_2_ckpt = select_checkpoint(st_key="ae_2_ckpt", default_run_path="gladia/rae/16tamf2p")
        ae_2: LightningAutoencoder = get_model(checkpoint_path=ae_2_ckpt, supported_code_version=CODE_VERSION)
        st.sidebar.markdown("---")

        st.subheader("AE")
        model_out = ae_1(image[None])
        batch_latent = model_out[Output.BATCH_LATENT]

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            st.pyplot(source_plot)
        with col2:
            reconstruction = ae_1.autoencoder.decoder(batch_latent)
            st.pyplot(plot_image(reconstruction))

            model_out_d1 = ae_1(images)

            if visualize_latent_space:
                pca = display_latent(st_container=col2, metadata=metadata, model_out=model_out_d1, pca=None)

        with col3:
            reconstruction = ae_2.autoencoder.decoder(batch_latent)
            st.pyplot(plot_image(reconstruction))

            model_out_d2 = ae_2(images)

            if visualize_latent_space:
                pca = display_latent(st_container=col3, metadata=metadata, model_out=model_out_d2, pca=None)

        if visualize_stats:
            with col4:
                st.markdown("**Ground truth**")
                st.metric(
                    "D1 gt: mse(D1(E1(50samples)), 50samples)",
                    numerize(mse_loss(model_out_d1[Output.RECONSTRUCTION], images).item(), decimals=5),
                )
                st.metric(
                    "D2 gt: mse(D2(E2(50samples)), 50samples)",
                    numerize(mse_loss(model_out_d2[Output.RECONSTRUCTION], images).item(), decimals=5),
                )

            with col5:
                st.markdown("**Invariance**")
                st.metric(
                    "Invariance Decoder: mse(D1(E1(source)), D2(E1(source)))",
                    numerize(mse_loss(reconstruction_d1, reconstruction_d2).item(), decimals=5),
                )
                st.metric(
                    "Invariance Decoder: mse(D1(E1(50samples)), D2(E1(50samples)))",
                    numerize(
                        mse_loss(
                            model_out_d1[Output.RECONSTRUCTION],
                            ae_2.autoencoder.decoder(model_out_d1[Output.BATCH_LATENT]),
                        ).item(),
                        decimals=5,
                    ),
                )
                st.metric(
                    "Invariance Decoder: mse(D1(E2(50samples)), D2(E2(50samples)))",
                    numerize(
                        mse_loss(
                            model_out_d2[Output.RECONSTRUCTION],
                            ae_1.autoencoder.decoder(model_out_d2[Output.BATCH_LATENT]),
                        ).item(),
                        decimals=5,
                    ),
                )
                st.metric(
                    "Invariance Encoder: mse(D1(E1(50samples)), D1(E2(50samples)))",
                    numerize(
                        mse_loss(
                            model_out_d1[Output.RECONSTRUCTION],
                            ae_1.autoencoder.decoder(model_out_d2[Output.BATCH_LATENT]),
                        ).item(),
                        decimals=5,
                    ),
                )
                st.metric(
                    "Invariance Encoder: mse(D2(E1(50samples)), D2(E2(50samples)))",
                    numerize(
                        mse_loss(
                            ae_2.autoencoder.decoder(model_out_d1[Output.BATCH_LATENT]),
                            model_out_d2[Output.RECONSTRUCTION],
                        ).item(),
                        decimals=5,
                    ),
                )
            with col6:
                st.markdown("**Model1 vs Model2**")

                st.metric(
                    "RAE1 vs RAE2: mse(D1(E1(50samples)), D2(E2(50samples)))",
                    numerize(
                        mse_loss(model_out_d1[Output.RECONSTRUCTION], model_out_d2[Output.RECONSTRUCTION]).item(),
                        decimals=5,
                    ),
                )

                st.metric(
                    "Distances errors: mse(distances_E1, distances_E2)",
                    numerize(
                        mse_loss(model_out_d1[Output.DEFAULT_LATENT], model_out_d2[Output.DEFAULT_LATENT]).item(),
                        decimals=5,
                    ),
                )
                compute_weights_difference(ae_1.autoencoder.decoder, ae_2.autoencoder.decoder)
