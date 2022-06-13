from typing import Dict

import hydra.utils
import streamlit as st
import torch
import torchvision.transforms.functional as transformf
from matplotlib import pyplot as plt
from stqdm import stqdm
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision.transforms import transforms

from nn_core.common import PROJECT_ROOT
from nn_core.serialization import NNCheckpointIO
from nn_core.ui import select_checkpoint

from rae.data.cifar100 import CIFAR100Dataset
from rae.modules.enumerations import Output
from rae.pl_modules.pl_gclassifier import LightningClassifier
from rae.ui.ui_utils import check_wandb_login, get_model, plot_image, show_code_version
from rae.utils.plotting import plot_images

plt.style.use("ggplot")


st.set_page_config(layout="wide")
st.title("Domain Adaptation")

check_wandb_login()


CODE_VERSION = "0.1.0"
show_code_version(code_version=CODE_VERSION)

device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown(f"Device: `{device}`\n\n---")

st.sidebar.header("Absolute Model")
abs_ckpt = select_checkpoint(st_key="relative_resnet", default_run_path="gladia/rae/2pi13o2j")
abs_model: LightningClassifier = get_model(
    module_class=LightningClassifier, checkpoint_path=abs_ckpt, supported_code_version=CODE_VERSION
)

st.sidebar.header("Relative Model")
rel_ckpt = select_checkpoint(st_key="absolute_resnet", default_run_path="gladia/rae/rkntehs9")
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
        for batch in stqdm(dataloader):
            images = batch["image"].to(device)
            targets = batch["target"].to(device)
            if new_anchors_images is None:
                output = model(images)
            else:
                output = model(images, new_anchors_images=new_anchors_images)
            accuracy(output[Output.LOGITS].cpu(), targets.cpu())
    st.metric("Accuracy", accuracy.compute().item())


@st.cache
def get_val_dataset(original_transforms):
    dataset = CIFAR100Dataset(
        split="test",
        transform=original_transforms,
        path=PROJECT_ROOT / "data",
    )
    return dataset


cfg = get_model_cfg(rel_ckpt)
original_transforms = get_model_transforms(cfg)
original_val_dataset = get_val_dataset(original_transforms)

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


sample_idx = st.number_input("Select an image", min_value=0, max_value=len(original_val_dataset), value=0)
sample = original_val_dataset[int(sample_idx)]

chosen_adaptation_transforms = st.multiselect("Select novel transforms", possible_adaptation_transforms.keys())
adaptation_transforms = transforms.Compose([possible_adaptation_transforms[x] for x in chosen_adaptation_transforms])

st.markdown(f'Class `{sample["class"]}`')
col1, col2 = st.columns(2)
with col1:
    st.markdown("Train domain")
    fig = plot_image(sample["image"])
    st.pyplot(fig)

with col2:
    st.markdown("Novel domain")
    val_dataset = get_val_dataset(transforms.Compose([original_transforms, adaptation_transforms]))
    sample = val_dataset[int(sample_idx)]
    fig = plot_image(sample["image"])
    st.pyplot(fig)


@st.cache
def get_val_dataloader(dataset):
    return DataLoader(dataset, batch_size=512, shuffle=False)


new_anchors_images = torch.stack([adaptation_transforms(x) for x in rel_model.anchors_images])
new_anchors_images.to(rel_model.device)
val_dataloader = get_val_dataloader(val_dataset)

st.sidebar.markdown("---")
if st.sidebar.checkbox("Show anchors"):
    with col1:
        f = plot_images(rel_model.anchors_images, title="anchors")
        st.pyplot(f)
    with col2:
        f = plot_images(new_anchors_images, title="new anchors")
        st.pyplot(f)

if st.checkbox("Evaluate on the CIFAR100 validation set"):
    col1, col2 = st.columns(2)
    with col1:
        st.header("Relative model performance")
        compute_accuracy(rel_model, dataloader=val_dataloader, new_anchors_images=new_anchors_images)
    with col2:
        st.header("Absolute model performance")
        compute_accuracy(abs_model, dataloader=val_dataloader)
