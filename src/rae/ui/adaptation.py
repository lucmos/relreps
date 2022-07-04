from typing import Dict

import hydra.utils
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as transformf
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from sklearn.decomposition import PCA
from stqdm import stqdm
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.transforms import transforms

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import NNCheckpointIO
from nn_core.ui import select_checkpoint

from rae.data.cifar100 import CIFAR100Dataset
from rae.data.datamodule import MetaData
from rae.modules.enumerations import Output
from rae.pl_modules.pl_gclassifier import LightningClassifier
from rae.ui.ui_utils import check_wandb_login, get_model, plot_image, show_code_version
from rae.utils.plotting import plot_images

seed_everything(0)

plt.style.use("ggplot")


st.set_page_config(layout="wide")
st.title("Domain Adaptation")

check_wandb_login()


CODE_VERSION = "0.1.0"
show_code_version(code_version=CODE_VERSION)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown(f"Device: `{device}`\n\n---")

st.sidebar.header("Absolute Model")
abs_ckpt = select_checkpoint(st_key="relative_resnet", default_run_path="gladia/rae/1526hguc")
abs_model: LightningClassifier = get_model(
    module_class=LightningClassifier, checkpoint_path=abs_ckpt, supported_code_version=CODE_VERSION
)

st.sidebar.header("Relative Model")
rel_ckpt = select_checkpoint(st_key="absolute_resnet", default_run_path="gladia/rae/235uwwsh")
rel_model = get_model(
    module_class=LightningClassifier,
    checkpoint_path=rel_ckpt,
    supported_code_version=CODE_VERSION,
)


@st.cache
def get_model_cfg(ckpt_path: str):
    cfg = NNCheckpointIO.load(path=ckpt_path, map_location="cpu")["cfg"]
    return cfg


@st.cache
def get_model_transforms(cfg: Dict):
    used_transforms = cfg["nn"]["data"]["transforms"]
    return hydra.utils.instantiate(used_transforms)


def compute_accuracy(model: LightningClassifier, dataloader, new_anchors_images=None):
    accuracy: Accuracy = Accuracy(num_classes=len(model.metadata.class_to_idx))
    model.eval()
    model = model.to(device)
    if new_anchors_images is not None:
        new_anchors_images = new_anchors_images.to(device)

    with torch.no_grad():
        inv_latents = []
        batch_latents = []
        anchors_latents = []
        for batch in stqdm(dataloader):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
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


possible_adaptation_transforms = {
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


cfg = get_model_cfg(rel_ckpt)

# Original samples
original_transforms = get_model_transforms(cfg)
original_val_dataset = get_val_dataset(original_transforms)
original_val_dataloader = get_val_dataloader(original_val_dataset)
original_anchors_images = rel_model.anchors_images.to(rel_model.device)

sample_idx = st.number_input("Select an image", min_value=0, max_value=len(original_val_dataset), value=0)
original_sample = original_val_dataset[int(sample_idx)]

# New transformed samples
chosen_adaptation_transforms = st.multiselect("Select novel transforms", possible_adaptation_transforms.keys())
adaptation_transforms = transforms.Compose([possible_adaptation_transforms[x] for x in chosen_adaptation_transforms])
novel_val_dataset = get_val_dataset(transforms.Compose([original_transforms, adaptation_transforms]))
novel_val_dataloader = get_val_dataloader(novel_val_dataset)
novel_anchors_images = torch.stack([adaptation_transforms(x) for x in rel_model.anchors_images]).to(rel_model.device)

novel_sample = novel_val_dataset[int(sample_idx)]


st.markdown(f'Class `{original_sample["class"]}`')
col1, col2 = st.columns(2)
with col1:
    st.markdown("Original domain")
    fig = plot_image(original_sample["image"])
    st.pyplot(fig)

with col2:
    st.markdown("Novel domain")
    fig = plot_image(novel_sample["image"])
    st.pyplot(fig)


st.sidebar.markdown("---")
if st.sidebar.checkbox("Show anchors"):
    with col1:
        f = plot_images(original_anchors_images, title="anchors")
        st.pyplot(f)
    with col2:
        f = plot_images(novel_anchors_images, title="novel anchors")
        st.pyplot(f)


if st.checkbox("Evaluate on the original/novel sample"):
    _, original_inv_latents, original_batch_latents, original_anchors_latents = compute_accuracy(
        rel_model,
        # dataloader=[{"image": novel_sample["image"][None], "target": torch.as_tensor(novel_sample["target"])[None]}],
        dataloader=[
            {"image": original_sample["image"][None], "target": torch.as_tensor(original_sample["target"])[None]}
        ],
        new_anchors_images=original_anchors_images,
    )

    _, novel_inv_latents, novel_batch_latents, novel_anchors_latents = compute_accuracy(
        rel_model,
        dataloader=[{"image": novel_sample["image"][None], "target": torch.as_tensor(novel_sample["target"])[None]}],
        new_anchors_images=novel_anchors_images,
    )

    dists = F.pairwise_distance(original_anchors_latents, novel_anchors_latents)
    st.markdown("L2 distance old-novel anchors latents")
    fig = px.bar(dists, labels={"index": "anchors", "value": "L2 distance"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    coss = F.cosine_similarity(original_anchors_latents, novel_anchors_latents)
    st.markdown("Cosine similarity old-novel anchors latents")
    fig = px.bar(coss, labels={"index": "anchors", "value": "Cosine Similarity"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Latent space old-novel anchors")
    pca = PCA(n_components=2)
    latents = torch.cat((original_anchors_latents, novel_anchors_latents), dim=0)
    latents = pca.fit_transform(latents)
    metadata: MetaData = rel_model.metadata
    df = pd.DataFrame(
        {
            "latent0": latents[:, 0],
            "latent1": latents[:, 1],
            "is_novel_anchor": [False] * novel_anchors_latents.shape[0] + [True] * novel_anchors_latents.shape[0],
            "target": metadata.anchors_targets.tolist() + metadata.anchors_targets.tolist(),
            "image_index": metadata.anchors_idxs + metadata.anchors_idxs,
            "class_name": metadata.anchors_classes + metadata.anchors_classes,
            "anchor_index": list(range(novel_anchors_latents.shape[0])) + list(range(novel_anchors_latents.shape[0])),
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
    st.plotly_chart(latent_val_fig, use_container_width=True)


    st.markdown("Anchors latent movements")
    import plotly.figure_factory as ff
    original_latents = latents[:original_anchors_latents.shape[0]]
    novel_latents = latents[original_anchors_latents.shape[0]:]

    scale = st.number_input("Arrows scale", 0., 1., 1.)
    st.plotly_chart(ff.create_quiver(
        original_latents[:, 0],
        original_latents[:, 1],
        novel_latents[:, 0] - original_latents[:, 0],
        novel_latents[:, 1] - original_latents[:, 1],
        name='quiver', line_width=1, scaleratio=1, scale=scale, arrow_scale=.25 ), use_container_width=True, )


    st.markdown("Original invariant latents for selected original sample")
    fig = px.bar(original_inv_latents.T, labels={"index": "anchors", "value": "Similarity"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Novel invariant latents for selected original sample")
    fig = px.bar(novel_inv_latents.T, labels={"index": "anchors", "value": "Similarity"})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Difference between old-new invariant latents for selected sample")
    fig = px.bar(
        (original_inv_latents - novel_inv_latents).T, labels={"index": "anchors", "value": "Difference Original-New"}
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.metric("L2 distance old-novel invariant latents", F.pairwise_distance(original_inv_latents, novel_inv_latents))
    st.metric(
        "Cosine similarity old-novel invariant latents", F.cosine_similarity(original_inv_latents, novel_inv_latents)
    )

if st.sidebar.checkbox("Evaluate on the CIFAR100 validation set"):
    novel_val_dataloader = get_val_dataloader(novel_val_dataset, batch_size=512)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Relative model performance")
        acc, inv_latents, batch_latents, anchors_latents = compute_accuracy(
            rel_model, dataloader=novel_val_dataloader, new_anchors_images=novel_anchors_images
        )
        st.metric("Accuracy", acc)
    with col2:
        st.header("Absolute model performance")
        acc, _, batch_latents, anchors_latents = compute_accuracy(abs_model, dataloader=novel_val_dataloader)
        st.metric("Accuracy", acc)
