import pandas as pd

from rae.modules.enumerations import Output


def cat_output_to_dataframe(validation_stats_df, output, current_epoch):
    return pd.concat(
        [
            validation_stats_df,
            pd.DataFrame(
                {
                    "image_index": output["batch"]["index"].cpu(),
                    "class": output["batch"]["class"],
                    "target": output["batch"]["target"].cpu(),
                    "latent_0": output[output[Output.DEFAULT_LATENT]][:, 0].cpu(),
                    "latent_1": output[output[Output.DEFAULT_LATENT]][:, 1].cpu(),
                    # "std_0": output["latent_logvar"][:, 0],
                    # "std_1": output["latent_logvar"][:, 1],
                    "epoch": [current_epoch] * len(output["batch"]["index"]),
                    "is_anchor": [False] * len(output["batch"]["index"]),
                    "anchor": [None] * len(output["batch"]["index"]),
                }
            ),
        ],
        ignore_index=False,
    )


def cat_anchors_stats_to_dataframe(
    validation_stats_df,
    anchors_images,
    anchors_out,
    anchors_latents,
    metadata,
    current_epoch,
):

    if anchors_images is not None:
        anchors_num = anchors_images.shape[0]
        non_elements = ["none"] * anchors_num

        validation_stats_df = pd.concat(
            [
                validation_stats_df,
                pd.DataFrame(
                    {
                        "image_index": metadata.anchors_idxs
                        if metadata.anchors_idxs is not None
                        else list(range(anchors_num)),
                        "class": metadata.anchors_classes if metadata.anchors_classes is not None else non_elements,
                        "target": metadata.anchors_targets.cpu()
                        if metadata.anchors_targets is not None
                        else non_elements,
                        "latent_0": anchors_out[anchors_out[Output.DEFAULT_LATENT]][:, 0].cpu(),
                        "latent_1": anchors_out[anchors_out[Output.DEFAULT_LATENT]][:, 1].cpu(),
                        # "std_0": anchors_latent_std[:, 0],
                        # "std_1": anchors_latent_std[:, 1],
                        "epoch": [current_epoch] * anchors_num,
                        "is_anchor": [True] * anchors_num,
                        "anchor": list(range(anchors_num)),
                    }
                ),
            ],
            ignore_index=False,
        )
    else:
        assert anchors_latents is not None
        anchors_num = anchors_latents.shape[0]
        non_elements = ["none"] * anchors_num
        validation_stats_df = pd.concat(
            [
                validation_stats_df,
                pd.DataFrame(
                    {
                        "image_index": metadata.anchors_idxs
                        if metadata.anchors_idxs is not None
                        else list(range(anchors_num)),
                        "class": non_elements,
                        "target": non_elements,
                        "latent_0": anchors_latents[:, 0].cpu(),
                        "latent_1": anchors_latents[:, 1].cpu(),
                        # "std_0": anchors_latent_std[:, 0],
                        # "std_1": anchors_latent_std[:, 1],
                        "epoch": [current_epoch] * anchors_num,
                        "is_anchor": [True] * anchors_num,
                        "anchor": list(range(anchors_num)),
                    }
                ),
            ],
            ignore_index=False,
        )
    return validation_stats_df
