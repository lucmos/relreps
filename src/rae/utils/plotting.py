from typing import Optional, Tuple

import plotly.express as px
import seaborn as sns
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
        validation_stats_df.loc[
            ((validation_stats_df["image_index"] < n_samples) | (validation_stats_df["is_anchor"]))
        ],
        x="latent_0",
        y="latent_1",
        animation_frame="epoch",
        animation_group="image_index",
        category_orders={"class_name": metadata.class_to_idx.keys()},
        #             # size='std_0',  # TODO: fixme, plotly crashes with any column name to set the anchor size
        color="class_name",
        hover_name="image_index",
        hover_data=["image_index", "anchor_index"],
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


def plot_latent_space(metadata, validation_stats_df, epoch: int, x_data: str, y_data: str, n_samples: int):
    color_discrete_map = {
        class_name: color
        for class_name, color in zip(metadata.class_to_idx, px.colors.qualitative.Plotly[: len(metadata.class_to_idx)])
    }

    latent_val_fig = px.scatter(
        validation_stats_df.loc[
            ((validation_stats_df["image_index"] < n_samples) | (validation_stats_df["is_anchor"]))
            & (validation_stats_df["epoch"] == epoch)
        ],
        x=x_data,
        y=y_data,
        category_orders={"class_name": metadata.class_to_idx.keys()},
        #             # size='std_0',  # TODO: fixme, plotly crashes with any column name to set the anchor size
        color="class_name",
        hover_name="image_index",
        hover_data=["image_index", "anchor_index"],
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


def plot_violin(batched_tensors, title, x_label, y_label, **kwargs):
    batched_tensors = batched_tensors.cpu().detach()
    # plotly
    # fig = px.violin(batched_tensors, points="outliers", box=True, **kwargs)

    sns.set_theme()
    fig, ax = plt.subplots(1, 1, figsize=(21, 7), dpi=120)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # for item in [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
    #     item.set_fontsize(40)
    fig.set_tight_layout(tight=True)
    sns.violinplot(ax=ax, data=batched_tensors, linewidth=1)
    sns.set_theme()
    return fig
