import logging

import plotly.figure_factory as ff
import streamlit as st
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from stqdm import stqdm
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from nn_core.ui import select_checkpoint

from rae.modules.enumerations import Output
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

pylogger = logging.getLogger(__name__)


class DatasetFromTensor(Dataset):
    def __init__(self, images, targets, class_to_idx):
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.images = images
        self.targets = targets
        self.class_to_idx = class_to_idx
        self.idx_to_class = {y: x for x, y in class_to_idx.items()}

    @property
    def class_vocab(self):
        return self.class_to_idx

    def __len__(self) -> int:
        # example
        return len(self.targets)

    def __getitem__(self, index: int):
        # example
        image, target = self.images[index], self.targets[index]
        return {"index": index, "image": image, "target": target, "class": self.idx_to_class[target.item()]}

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}(n_instances={len(self)})"


seed_everything(0)

plt.style.use("ggplot")


st.set_page_config(layout="wide")
st.title("Domain Adaptation")

check_wandb_login()

execute_all = st.checkbox("Execute all")


CODE_VERSION = "0.0.2"
show_code_version(code_version=CODE_VERSION)

device = "cpu"
best_device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.markdown(f"Device: `{best_device}`\n\n---")

st.sidebar.header("Absolute Model")
abs_ckpt = select_checkpoint(st_key="relative_resnet", default_run_path="gladia/rae/1526hguc")
abs_model: LightningClassifier = get_model(
    module_class=LightningClassifier, checkpoint_path=abs_ckpt, supported_code_version=CODE_VERSION
).to(device)

st.sidebar.header("Relative Model")
rel_ckpt = select_checkpoint(st_key="absolute_resnet", default_run_path="gladia/rae/u0uhfmny")
rel_model = get_model(
    module_class=LightningClassifier,
    checkpoint_path=rel_ckpt,
    supported_code_version=CODE_VERSION,
).to(device)


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


def finetune(model, parameters_to_tune, dataloader, lr, epochs, new_anchors_images, compute_device):
    model = model.to(compute_device)
    if new_anchors_images is not None:
        new_anchors_images = new_anchors_images.to(compute_device)

    st.write(f"tot: {sum(x.sum().detach() for x in parameters_to_tune)}")
    model.train()
    opt = optim.Adam(parameters_to_tune, lr=lr, weight_decay=1e-5)
    for epoch in (bar := stqdm(range(epochs))):
        epoch_loss = 0
        num_batches = 0

        for batch in dataloader:
            image = batch["image"].to(compute_device)
            target = batch["target"].to(compute_device)

            opt.zero_grad()
            if new_anchors_images is not None:
                out = model(image, new_anchors_images=new_anchors_images)
            else:
                out = model(image)

            loss = model.loss(out[Output.LOGITS], target)

            loss.backward()
            opt.step()

            num_batches += 1
            epoch_loss += loss.detach().cpu().item()

        bar.set_description(
            f"Loss: {epoch_loss / num_batches:.3f}, tot: {sum(x.sum().detach() for x in parameters_to_tune)}"
        )
    st.write(epoch_loss / num_batches)
    model.eval()
    return model.to("cpu")


if sample_eval := st.checkbox("Evaluate on the original/novel sample", value=execute_all):
    _, original_inv_latents, original_batch_latents, original_anchors_latents = compute_accuracy(
        rel_model,
        # dataloader=[{"image": novel_sample["image"][None], "target": torch.as_tensor(novel_sample["target"])[None]}],
        dataloader=[
            {"image": original_sample["image"][None], "target": torch.as_tensor(original_sample["target"])[None]}
        ],
        compute_device=best_device,
        new_anchors_images=original_anchors_images,
    )

if st.checkbox("Fine tune on transformed anchors", value=execute_all):
    finetune_dataset = DatasetFromTensor(
        images=novel_anchors_images,
        targets=rel_model.metadata.anchors_targets,
        class_to_idx=rel_model.metadata.class_to_idx,
    )
    finetune_dataloader = DataLoader(finetune_dataset, batch_size=256, shuffle=True)

    novel_val_dataloader = get_val_dataloader(novel_val_dataset, batch_size=256)

    rel_model = finetune(
        model=rel_model,
        parameters_to_tune=(
            *rel_model.model.resnet.parameters(),
            *rel_model.model.resnet_post_fc.parameters(),
            # rel_model.model.relative_attention_block.attention.attention.values,
            # *rel_model.model.relative_attention_block.attention.linear.parameters(),
        ),
        dataloader=finetune_dataloader,
        lr=0.002,
        epochs=10,
        new_anchors_images=novel_anchors_images,
        compute_device=best_device,
    )

    abs_model = finetune(
        model=abs_model,
        parameters_to_tune=(
            *abs_model.model.resnet.parameters(),
            *abs_model.model.resnet_post_fc.parameters(),
            # abs_model.model.relative_attention_block.attention.attention.values,
            # *abs_model.model.relative_attention_block.attention.linear.parameters(),
        ),
        dataloader=finetune_dataloader,
        lr=0.002,
        epochs=10,
        new_anchors_images=None,
        compute_device=best_device,
    )

if sample_eval:
    _, novel_inv_latents, novel_batch_latents, novel_anchors_latents = compute_accuracy(
        rel_model,
        dataloader=[{"image": novel_sample["image"][None], "target": torch.as_tensor(novel_sample["target"])[None]}],
        compute_device=best_device,
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

if st.sidebar.checkbox("Evaluate on the CIFAR100 validation set", value=execute_all):
    novel_val_dataloader = get_val_dataloader(novel_val_dataset, batch_size=512)

    col1, col2 = st.columns(2)
    with col1:
        st.header("Relative model performance")
        acc, inv_latents, batch_latents, anchors_latents = compute_accuracy(
            rel_model,
            dataloader=novel_val_dataloader,
            new_anchors_images=novel_anchors_images,
            compute_device=best_device,
        )
        st.metric("Accuracy", acc)

    with col2:
        st.header("Absolute model performance")
        acc, _, batch_latents, anchors_latents = compute_accuracy(
            abs_model,
            dataloader=novel_val_dataloader,
            compute_device=best_device,
        )
        st.metric("Accuracy", acc)
