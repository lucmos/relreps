import logging
import random
from enum import auto
from pathlib import Path
from typing import Mapping, Sequence, Set

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from torch.utils.data import Dataset
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

import rae  # noqa
from rae.modules.text.encoder import GensimEncoder
from rae.utils.utils import StrEnum

pylogger = logging.getLogger(__name__)


class AnchorSamplingMethod(StrEnum):
    TF_IDF = auto()
    K_MEANS_RANDOM = auto()
    K_MEANS_MOST_FREQUENT = auto()
    K_MEANS_CENTROID = auto()
    ALL_SIMILAR = auto()
    RANDOM = auto()
    MOST_FREQUENT = auto()
    BEST_ALIGN = auto()


class AnchorPolicy:
    pass


def get_best_aligned_indices(A, B, chunk_size=500, p=2):
    num_chunks = A.shape[0] // chunk_size
    A = F.normalize(A, p=p, dim=-1)
    B = F.normalize(B, p=p, dim=-1)
    all_dists_diff = []
    for chunk_a, chunk_b in tqdm(
        zip(torch.chunk(A, num_chunks, dim=0), torch.chunk(B, num_chunks, dim=0)), total=num_chunks
    ):
        dists_a = torch.cdist(chunk_a, A, p=p)
        dists_b = torch.cdist(chunk_b, B, p=p)

        dists_diff = (dists_a - dists_b).norm(dim=-1)
        all_dists_diff.append(dists_diff)
    return torch.cat(all_dists_diff).sort().indices


class EmbeddingAnchorDataset(Dataset):
    @classmethod
    def build_kmeans(cls, k: Sequence[int]) -> Sequence[str]:
        pass

    @classmethod
    def build_anchors(cls, method: AnchorSamplingMethod, text_encoders: Sequence[GensimEncoder]) -> Sequence[str]:
        seed_everything(42)
        assert len({frozenset(encoder.model.key_to_index.keys()) for encoder in text_encoders}) == 1

        target_dir: Path = PROJECT_ROOT / "data" / "anchor_dataset"
        target_dir.mkdir(exist_ok=True, parents=True)

        target_encoder = [encoder for encoder in text_encoders if encoder.model_name == "local_fasttext"]
        if len(target_encoder) == 0:
            target_encoder = text_encoders[0]
        else:
            target_encoder = target_encoder[0]

        stopwords = target_encoder.stopwords

        if method == AnchorSamplingMethod.MOST_FREQUENT:
            anchors = list(target_encoder.model.key_to_index.keys())
        elif method == AnchorSamplingMethod.RANDOM:
            anchors = list(target_encoder.model.key_to_index.keys())
            random.shuffle(anchors)
        elif method == AnchorSamplingMethod.K_MEANS_RANDOM:
            vectors = normalize(target_encoder.model.vectors, norm="l2")
            clustered = KMeans(n_clusters=42).fit_predict(vectors)

            shuffled_idxs, shuffled_targets = shuffle(
                np.asarray(list(range(len(target_encoder.model.vectors)))),
                np.asarray(clustered),
                random_state=0,
            )
            all_targets = sorted(set(shuffled_targets))
            cluster2idxs = {target: shuffled_idxs[shuffled_targets == target] for target in all_targets}

            anchor_indices = []
            i = 0
            while len(anchor_indices) < len(clustered):
                for target, target_idxs in cluster2idxs.items():
                    if len(target_idxs) <= i:
                        continue
                    anchor_indices.append(target_idxs[i])
                i += 1

            anchors = [target_encoder.model.index_to_key[index] for index in anchor_indices]
        elif method == AnchorSamplingMethod.K_MEANS_MOST_FREQUENT:
            vectors = normalize(target_encoder.model.vectors, norm="l2")
            clustered = KMeans(n_clusters=42).fit_predict(vectors)

            idxs = np.asarray(list(range(len(target_encoder.model.vectors))))
            all_targets = sorted(set(clustered))
            cluster2idxs = {target: idxs[clustered == target] for target in all_targets}

            anchor_indices = []
            i = 0
            while len(anchor_indices) < len(clustered):
                for target, target_idxs in cluster2idxs.items():
                    if len(target_idxs) <= i:
                        continue
                    anchor_indices.append(target_idxs[i])
                i += 1

            anchors = [target_encoder.model.index_to_key[index] for index in anchor_indices]
        elif method == AnchorSamplingMethod.K_MEANS_CENTROID:
            vectors = normalize(target_encoder.model.vectors, norm="l2")
            clustered = KMeans(n_clusters=42).fit_predict(vectors)

            idxs = np.asarray(list(range(len(target_encoder.model.vectors))))
            all_targets = sorted(set(clustered))
            cluster2idxs = {target: idxs[clustered == target] for target in all_targets}

            for cluster_id, idxs in cluster2idxs.items():
                cluster_vectors = vectors[idxs]
                cluster_centroid: np.ndarray = np.average(cluster_vectors, axis=0)
                distances = cdist(cluster_centroid[None, :], cluster_vectors)[0]
                sorted_idxs = np.argsort(distances)
                cluster2idxs[cluster_id] = sorted_idxs

            anchor_indices = []
            i = 0
            while len(anchor_indices) < len(clustered):
                for target, target_idxs in cluster2idxs.items():
                    if len(target_idxs) <= i:
                        continue
                    anchor_indices.append(target_idxs[i])
                i += 1

            anchors = [target_encoder.model.index_to_key[index] for index in anchor_indices]
        elif method in {
            AnchorSamplingMethod.K_MEANS_10,
            AnchorSamplingMethod.K_MEANS_100,
            AnchorSamplingMethod.K_MEANS_500,
            AnchorSamplingMethod.K_MEANS_1000,
        }:
            # k = int(method.name.split("_")[-1])
            pass
        else:
            raise NotImplementedError

        assert all("\n" not in x for x in anchors)

        anchors = [anchor for anchor in anchors if anchor not in stopwords and len(anchor) > 3 and anchor.isalpha()]
        (target_dir / f"{method}.txt").write_text("\n".join(anchors), encoding="utf-8")

        return anchors

    @classmethod
    def load_anchors(cls, method: AnchorSamplingMethod, num_anchors: int) -> Sequence[str]:
        return (
            (PROJECT_ROOT / "data" / "anchor_dataset" / f"{method}.txt")
            .read_text(encoding="utf-8")
            .splitlines()[:num_anchors]
        )

    def __init__(self, split: Split, method: AnchorSamplingMethod, num_anchors: int, **kwargs):
        super().__init__()
        split = split if split == "train" else "test"
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}> ('{split}')")

        self.split: Split = split
        self.anchors = EmbeddingAnchorDataset.load_anchors(method=method, num_anchors=num_anchors)

    @property
    def classes(self) -> Sequence[str]:
        return None

    @property
    def targets(self) -> Sequence[int]:
        return None

    @property
    def class_vocab(self) -> Mapping[str, int]:
        return None

    def get_stopwords(self) -> Set[str]:
        return set()

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int):
        index: int = int(index)
        sample = self.anchors[index]
        return {
            "index": index,
            "data": sample,
            "target": None,
            "class": None,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the Dataset.

    Args:
        cfg: the hydra configuration
    """
    dataset: Dataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train,
        split="train",
        path=PROJECT_ROOT / "data",
        _recursive_=False,
    )
    _ = dataset[0]


if __name__ == "__main__":
    ENCODERS = [
        GensimEncoder(language="en", lemmatize=False, model_name=model_name)
        for model_name in (
            "local_fasttext",
            "word2vec-google-news-300",
            "glove-wiki-gigaword-300",
        )
    ]

    for method in (
        # AnchorSamplingMethod.K_MEANS_MOST_FREQUENT,
        # AnchorSamplingMethod.K_MEANS_RANDOM,
        # AnchorSamplingMethod.K_MEANS_CENTROID,
        # AnchorSamplingMethod.RANDOM,
        # AnchorSamplingMethod.MOST_FREQUENT,
    ):
        EmbeddingAnchorDataset.build_anchors(method=method, text_encoders=ENCODERS)
