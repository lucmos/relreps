import plotly.figure_factory as ff
import streamlit as st
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from torchvision.transforms import transforms

from nn_core.ui import select_checkpoint

from rae.pl_modules.pl_gclassifier import LightningClassifier
from rae.ui.ui_utils import (
    AVAILABLE_TRANSFORMS,
    check_wandb_login,
    compute_accuracy,
    get_model,
    get_model_cfg,
    get_model_transforms,
    get_val_dataloader,
    get_val_dataset,
    plot_bar,
    plot_image,
    plot_latent_space_comparison,
    show_code_version,
)
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


cfg = get_model_cfg(rel_ckpt)

# Original samples
original_transforms = get_model_transforms(cfg)
original_val_dataset = get_val_dataset(original_transforms)
original_val_dataloader = get_val_dataloader(original_val_dataset)
original_anchors_images = rel_model.anchors_images.to(rel_model.device)

sample_idx = st.number_input("Select an image", min_value=0, max_value=len(original_val_dataset), value=0)
original_sample = original_val_dataset[int(sample_idx)]

# New transformed samples
chosen_adaptation_transforms = st.multiselect("Select novel transforms", AVAILABLE_TRANSFORMS.keys())
adaptation_transforms = transforms.Compose([AVAILABLE_TRANSFORMS[x] for x in chosen_adaptation_transforms])
novel_val_dataset = get_val_dataset(transforms.Compose([original_transforms, adaptation_transforms]))
novel_val_dataloader = get_val_dataloader(novel_val_dataset)
novel_anchors_images = torch.stack([adaptation_transforms(x) for x in rel_model.anchors_images]).to(rel_model.device)

novel_sample = novel_val_dataset[int(sample_idx)]


st.markdown(f'Class `{original_sample["class"]}`')
col1, col2 = st.columns(2)
with col1:
    st.markdown("Original domain")
    st.pyplot(plot_image(original_sample["image"]))

with col2:
    st.markdown("Novel domain")
    st.pyplot(plot_image(novel_sample["image"]))


st.sidebar.markdown("---")
if st.sidebar.checkbox("Show anchors"):
    with col1:
        st.pyplot(plot_images(original_anchors_images, title="anchors"))
    with col2:
        st.pyplot(plot_images(novel_anchors_images, title="novel anchors"))


if st.checkbox("Evaluate on the original/novel sample"):
    _, original_inv_latents, original_batch_latents, original_anchors_latents = compute_accuracy(
        rel_model,
        # dataloader=[{"image": novel_sample["image"][None], "target": torch.as_tensor(novel_sample["target"])[None]}],
        dataloader=[
            {"image": original_sample["image"][None], "target": torch.as_tensor(original_sample["target"])[None]}
        ],
        device=device,
        new_anchors_images=original_anchors_images,
    )

    _, novel_inv_latents, novel_batch_latents, novel_anchors_latents = compute_accuracy(
        rel_model,
        dataloader=[{"image": novel_sample["image"][None], "target": torch.as_tensor(novel_sample["target"])[None]}],
        device=device,
        new_anchors_images=novel_anchors_images,
    )

    st.markdown("L2 distance old-novel anchors latents")
    st.plotly_chart(
        plot_bar(
            F.pairwise_distance(original_anchors_latents, novel_anchors_latents),
            labels={"index": "anchors", "value": "L2 distance"},
        ),
        use_container_width=True,
    )

    st.markdown("Cosine similarity old-novel anchors latents")
    st.plotly_chart(
        plot_bar(
            F.cosine_similarity(original_anchors_latents, novel_anchors_latents),
            labels={"index": "anchors", "value": "Cosine Similarity"},
        ),
        use_container_width=True,
    )

    st.markdown("Latent space old-novel anchors")
    latent_val_fig, latents = plot_latent_space_comparison(
        metadata=rel_model.metadata,
        original_latents=original_anchors_latents,
        novel_latents=novel_anchors_latents,
    )
    st.plotly_chart(latent_val_fig, use_container_width=True)

    st.markdown("Anchors latent movements")

    original_latents = latents[: original_anchors_latents.shape[0]]
    novel_latents = latents[original_anchors_latents.shape[0] :]

    scale = st.number_input("Arrows scale", 0.0, 1.0, 1.0)
    st.plotly_chart(
        ff.create_quiver(
            original_latents[:, 0],
            original_latents[:, 1],
            novel_latents[:, 0] - original_latents[:, 0],
            novel_latents[:, 1] - original_latents[:, 1],
            name="quiver",
            line_width=1,
            scaleratio=1,
            scale=scale,
            arrow_scale=0.25,
        ),
        use_container_width=True,
    )

    st.markdown("Original invariant latents for selected original sample")
    st.plotly_chart(
        plot_bar(original_inv_latents.T, labels={"index": "anchors", "value": "Similarity"}),
        use_container_width=True,
    )

    st.markdown("Novel invariant latents for selected original sample")
    st.plotly_chart(
        plot_bar(novel_inv_latents.T, labels={"index": "anchors", "value": "Similarity"}),
        use_container_width=True,
    )

    st.markdown("Difference between old-new invariant latents for selected sample")
    st.plotly_chart(
        plot_bar(
            (original_inv_latents - novel_inv_latents).T,
            labels={"index": "anchors", "value": "Difference Original-New"},
        ),
        use_container_width=True,
    )

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
            rel_model,
            dataloader=novel_val_dataloader,
            new_anchors_images=novel_anchors_images,
            device=device,
        )
        st.metric("Accuracy", acc)

    with col2:
        st.header("Absolute model performance")
        acc, _, batch_latents, anchors_latents = compute_accuracy(
            abs_model,
            dataloader=novel_val_dataloader,
            device=device,
        )
        st.metric("Accuracy", acc)
