import copy
from pathlib import Path
from typing import Dict, Optional, Tuple, Type

import hydra
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torchvision
import torchvision.transforms.functional as transformf
import wandb
from matplotlib import pyplot as plt
from plotly.graph_objs import Figure
from pytorch_lightning import LightningModule
from sklearn.decomposition import PCA
from stqdm import stqdm
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import NNCheckpointIO, load_model

from rae.data.vision.cifar100 import CIFAR100Dataset
from rae.data.vision.datamodule import MetaData
from rae.modules.enumerations import Output
from rae.pl_modules.vision.pl_gautoencoder import LightningAutoencoder
from rae.pl_modules.vision.pl_gclassifier import LightningClassifier
from rae.utils.plotting import plot_violin

AVAILABLE_TRANSFORMS = {
    "brighter": lambda x: transformf.adjust_brightness(x, brightness_factor=1.5),
    "darker": lambda x: transformf.adjust_brightness(x, brightness_factor=0.5),
    "lower_contrast": lambda x: transformf.adjust_contrast(x, contrast_factor=0.5),
    "higher_contrast": lambda x: transformf.adjust_contrast(x, contrast_factor=1.5),
    "lower_gamma": lambda x: transformf.adjust_gamma(x, gamma=0.5),
    "higher_gamma": lambda x: transformf.adjust_gamma(x, gamma=1.5),
    "lower_hue": lambda x: transformf.adjust_hue(x, hue_factor=-0.5),
    "higher_hue": lambda x: transformf.adjust_hue(x, hue_factor=0.5),
    "lower_saturation": lambda x: transformf.adjust_saturation(x, saturation_factor=0.25),
    "higher_saturation": lambda x: transformf.adjust_saturation(x, saturation_factor=0.75),
    "lower_sharpness": lambda x: transformf.adjust_sharpness(x, sharpness_factor=0),
    "higher_sharpness": lambda x: transformf.adjust_sharpness(x, sharpness_factor=4),
    "autocontrast": lambda x: transformf.autocontrast(x),
    "gaussian_blur": lambda x: transformf.gaussian_blur(x, kernel_size=5),
    "hflip": lambda x: transformf.hflip(x),
    "invert": lambda x: transformf.invert(x),
    "rgb_to_grayscale": lambda x: transformf.rgb_to_grayscale(x, 3),
    "solarize": lambda x: transformf.solarize(x, threshold=0.5),
    "rotate": lambda x: transformf.rotate(x, 45),
    "vflip": lambda x: transformf.vflip(x),
}


def show_code_version(code_version: str):
    st.sidebar.markdown(f"Demo compatibility with version: `{code_version}`")
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
def _get_model(
    checkpoint_path: Path, supported_code_version: str, module_class: Type[LightningModule] = LightningAutoencoder
):
    try:
        model = load_model(
            module_class=module_class,
            checkpoint_path=checkpoint_path,
            map_location="cpu",
            substitute_values={"rae.modules.rae.RAE": "rae.modules.rae_model.RAE"},
        )
        model.eval()
        return model
    except Exception as e:  # noqa
        ckpt_code_version = NNCheckpointIO.load(path=checkpoint_path, map_location="cpu")["cfg"]["core"]["version"]
        st.error(
            f"Codebase version mismatch. Checkpoint trained with version `{ckpt_code_version}` support version `{supported_code_version}"
        )
        st.write(e)
        st.stop()
        return None


def get_model(
    checkpoint_path: Path, supported_code_version: str, module_class: Type[LightningModule] = LightningAutoencoder
):
    model = _get_model(checkpoint_path, supported_code_version, module_class)
    return copy.deepcopy(model)


@st.cache
def get_model_cfg(ckpt_path: str):
    cfg = NNCheckpointIO.load(path=ckpt_path, map_location="cpu")["cfg"]
    return cfg


@st.cache
def get_model_transforms(cfg: Dict):
    used_transforms = cfg["nn"]["data"]["transforms"]
    return hydra.utils.instantiate(used_transforms)


@st.cache
def get_val_dataset(original_transforms):
    dataset = CIFAR100Dataset(
        split="test",
        transform=original_transforms,
        path=PROJECT_ROOT / "data",
    )
    return dataset


@st.cache
def get_val_dataloader(dataset, batch_size=10):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


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


def plot_latent_space_comparison(metadata: MetaData, original_latents, novel_latents) -> Tuple[Figure, torch.Tensor]:
    pca = PCA(n_components=2)
    latents = torch.cat((original_latents, novel_latents), dim=0)
    latents = pca.fit_transform(latents)
    df = pd.DataFrame(
        {
            "latent0": latents[:, 0],
            "latent1": latents[:, 1],
            "is_novel_anchor": [False] * novel_latents.shape[0] + [True] * novel_latents.shape[0],
            "target": metadata.anchors_targets.tolist() + metadata.anchors_targets.tolist(),
            "image_index": metadata.anchors_idxs + metadata.anchors_idxs,
            "class_name": metadata.anchors_classes + metadata.anchors_classes,
            "anchor_index": list(range(novel_latents.shape[0])) + list(range(novel_latents.shape[0])),
        }
    )
    color_discrete_map = {
        class_name: color
        for class_name, color in zip(metadata.class_to_idx, px.colors.qualitative.Plotly[: len(metadata.class_to_idx)])
    }
    latent_val_fig = px.scatter(
        df,
        x="latent0",
        y="latent1",
        category_orders={"class_name": metadata.class_to_idx.keys()},
        color="class_name",
        hover_name="image_index",
        hover_data=["image_index", "anchor_index"],
        facet_col="is_novel_anchor",
        color_discrete_map=color_discrete_map,
        # symbol="is_anchor",
        # symbol_map={False: "circle", True: "star"},
        size_max=40,
        # range_x=[-5, 5],
        color_continuous_scale=None,
        # range_y=[-5, 5],
    )
    return latent_val_fig, latents


def compute_accuracy(
    model: LightningClassifier,
    dataloader,
    compute_device,
    new_anchors_images=None,
):
    accuracy: Accuracy = Accuracy(num_classes=len(model.metadata.class_to_idx))
    model.eval()
    model = model.to(compute_device)
    if new_anchors_images is not None:
        new_anchors_images = new_anchors_images.to(compute_device)

    with torch.no_grad():
        inv_latents = []
        batch_latents = []
        anchors_latents = []
        for batch in stqdm(dataloader):
            images = batch["image"].to(compute_device)
            targets = batch["target"].to(compute_device)
            if new_anchors_images is None:
                output = model(images)
            else:
                output = model(images, new_anchors_images=new_anchors_images)
                inv_latents.append(output[Output.INV_LATENTS])
                anchors_latents.append(output[Output.ANCHORS_LATENT])
            batch_latents.append(output[Output.BATCH_LATENT])
            accuracy(output[Output.LOGITS].cpu(), targets.cpu())
        if inv_latents:
            inv_latents = torch.cat(inv_latents, dim=0).cpu()
        batch_latents = torch.cat(batch_latents, dim=0).cpu()
        if anchors_latents:
            anchors_latents = torch.cat(anchors_latents, dim=0).cpu()

    return accuracy.compute().item(), inv_latents, batch_latents, anchors_latents


def plot_bar(data, **kwargs) -> Figure:
    fig = px.bar(data, **kwargs)
    fig.update_layout(showlegend=False)
    return fig
