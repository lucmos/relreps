import pandas as pd
import torch.nn.functional as F
from sklearn.decomposition import PCA

from rae.modules.enumerations import Output


def cat_output_to_dataframe(validation_stats_df, output, current_epoch, pca: PCA):
    latents = output[Output.DEFAULT_LATENT].cpu()
    latents_normalized = F.normalize(latents, p=2, dim=-1)
    latents_pca = pca.transform(latents)

    return pd.concat(
        [
            validation_stats_df,
            pd.DataFrame(
                {
                    "image_index": output["batch"]["index"].cpu(),
                    "class": output["batch"]["class"],
                    "target": output["batch"]["target"].cpu(),
                    "latent_0": latents[:, 0],
                    "latent_1": latents[:, 1],
                    "latent_0_normalized": latents_normalized[:, 0],
                    "latent_1_normalized": latents_normalized[:, 1],
                    "latent_0_pca": latents_pca[:, 0],
                    "latent_1_pca": latents_pca[:, 1],
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
    validation_stats_df, anchors_num, anchors_latents, metadata, current_epoch, pca: PCA
):
    non_elements = ["none"] * anchors_num
    latents = anchors_latents.cpu()
    latents_normalized = F.normalize(latents, p=2, dim=-1)
    latents_pca = pca.transform(latents)

    validation_stats_df = pd.concat(
        [
            validation_stats_df,
            pd.DataFrame(
                {
                    "image_index": metadata.anchors_idxs
                    if metadata.anchors_idxs is not None
                    else list(range(anchors_num)),
                    "class": metadata.anchors_classes if metadata.anchors_classes is not None else non_elements,
                    "target": metadata.anchors_targets.cpu() if metadata.anchors_targets is not None else non_elements,
                    "latent_0": latents[:, 0],
                    "latent_1": latents[:, 1],
                    "latent_0_normalized": latents_normalized[:, 0],
                    "latent_1_normalized": latents_normalized[:, 1],
                    "latent_0_pca": latents_pca[:, 0],
                    "latent_1_pca": latents_pca[:, 1],
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
