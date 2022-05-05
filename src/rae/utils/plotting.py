from typing import Optional, Tuple

import plotly.express as px
import torch
import torchvision
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def plot_latent_evolution(metadata, validation_stats_df, n_samples):
    color_discrete_map = {
        class_name: color
        for class_name, color in zip(metadata.class_to_idx, px.colors.qualitative.Plotly[: len(metadata.class_to_idx)])
    }

    latent_val_fig = px.scatter(
        validation_stats_df.loc[validation_stats_df["image_index"] < n_samples],
        x="latent_0",
        y="latent_1",
        animation_frame="epoch",
        animation_group="image_index",
        category_orders={"class": metadata.class_to_idx.keys()},
        #             # size='std_0',  # TODO: fixme, plotly crashes with any column name to set the anchor size
        color="class",
        hover_name="image_index",
        hover_data=["image_index", "anchor"],
        facet_col="is_anchor",
        color_discrete_map=color_discrete_map,
        # symbol="is_anchor",
        # symbol_map={False: "circle", True: "star"},
        size_max=40,
        range_x=[-5, 5],
        color_continuous_scale=None,
        range_y=[-5, 5],
    )
    return latent_val_fig


def plot_images(images: torch.Tensor, title: str, figsize: Optional[Tuple[int, int]] = None) -> Figure:
    images = images.cpu().detach()
    fig, ax = plt.subplots(1, 1, figsize=(17, 9) if figsize is None else figsize)
    ax.set_title(title)
    ax.axis("off")
    fig.set_tight_layout(tight=True)
    ax.imshow(torchvision.utils.make_grid(images.cpu(), 10, 5).permute(1, 2, 0))

    # Plotly version
    # fig = px.imshow(torchvision.utils.make_grid(images.cpu(), 10, 5).permute(1, 2, 0), title=title)
    return fig


def plot_matrix(matrix, **kwargs):
    matrix = matrix.cpu().detach()
    fig = px.imshow(
        matrix,
        color_continuous_midpoint=0,
        # range_color=[-5, 5],
        color_continuous_scale="RdBu",
        aspect="equal",
        **kwargs
    )
    return fig


def plot_violin(batched_tensors, **kwargs):
    batched_tensors = batched_tensors.cpu().detach()
    fig = px.violin(batched_tensors, points="outliers", box=True, **kwargs)
    return fig
