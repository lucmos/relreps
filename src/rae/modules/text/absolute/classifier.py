import functools
import itertools
import logging
from typing import Any, Collection, Dict, Sequence, Set

import torch
from hydra.utils import instantiate
from torch import nn
from torch.types import Device
from transformers import AutoModel, PreTrainedModel

from rae.data.text.datamodule import EncodingLevel, MetaData
from rae.modules.blocks import DeepProjection
from rae.modules.enumerations import Output
from rae.modules.text.encoder import TextEncoder
from rae.utils.utils import to_device

pylogger = logging.getLogger(__name__)


class HFTextClassifier(nn.Module):
    def collate_fn(self, batch: Sequence[Any]):
        return self.text_encoder.collate_fn(batch=batch)

    def __init__(
        self,
        metadata: MetaData,
        text_encoder: TextEncoder,
        transformer_name: str,
        batch_pre_reduce: Collection[EncodingLevel] = None,
        batch_post_reduce: Collection[EncodingLevel] = None,
        finetune: bool = False,
        **kwargs,
    ) -> None:
        """Simple model that uses convolutions.

        Args:
            metadata: the metadata object
            relative_projection: the relative projection module (attention/transformer...)
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata
        self.text_encoder: TextEncoder = instantiate(text_encoder)

        self.text_encoder.add_stopwords(stopwords=metadata.stopwords)

        n_classes: int = len(self.metadata.class_to_idx)

        self.transformer: PreTrainedModel = AutoModel.from_pretrained(
            transformer_name, output_hidden_states=True, return_dict=True
        ).eval()

        if not finetune:
            self.transformer.requires_grad_(False)
            self.transformer.eval()

        self.finetune: bool = finetune

        self.pre_reduce: Set[EncodingLevel] = (
            set(batch_pre_reduce) if batch_pre_reduce is not None and len(batch_pre_reduce) > 0 else {}
        )

        self.post_reduce: Set[EncodingLevel] = (
            set(batch_post_reduce) if batch_post_reduce is not None and len(batch_post_reduce) > 0 else {}
        )

        # assert len(set.intersection(self.pre_reduce, self.post_reduce)) == 0
        assert all(x < y for x, y in itertools.product(self.pre_reduce, self.post_reduce))

        transformer_config = self.transformer.config.to_dict()
        # TODO: DistilBert doesn't have the "hidden_size" parameter :@
        transformer_encoding_dim = transformer_config["hidden_size" if "hidden_size" in transformer_config else "dim"]

        self.sequential = nn.Sequential(
            nn.LayerNorm(normalized_shape=transformer_encoding_dim),
            nn.Linear(in_features=transformer_encoding_dim, out_features=transformer_encoding_dim),
            DeepProjection(
                in_features=transformer_encoding_dim,
                out_features=n_classes,
                dropout=0.1,
                num_layers=3,
                activation=nn.SiLU(),
            ),
            nn.ReLU(),
        )

        self._cache: Dict[str, torch.Tensor] = {}

    def set_finetune_mode(self):
        if not self.finetune:
            self.transformer.requires_grad_(False)
            self.transformer.eval()

    def call_transformer(self, encodings, mask: torch.Tensor, sample_ids: Sequence[str]):
        if any(sample_id not in self._cache for sample_id in sample_ids):
            sample_encodings = self.transformer(**encodings)["hidden_states"][-1]
            # TODO: aggregation mode
            result = []
            for sample_encoding, sample_mask, sample_id in zip(sample_encodings, mask, sample_ids):
                sample_encoding: torch.Tensor = sample_encoding[sample_mask].mean(dim=0)
                result.append(sample_encoding)
                self._cache[sample_id] = sample_encoding.cpu()
        else:
            result = [self._cache[sample_id] for sample_id in sample_ids]

        return torch.stack(result, dim=0)

    def encode(self, batch, device: Device):
        if "encodings" not in batch:
            assert False, "Call the tokenizer, you lazy bastard"

        # x, reduced_to_sentence = EncodingLevel.reduce(
        #     encodings=x, **batch["sections"], reduced_to_sentence=False, reduce_transformations=self.pre_reduce
        # )

        with torch.no_grad():
            x = self.call_transformer(encodings=batch["encodings"], mask=batch["mask"], sample_ids=batch["index"]).to(
                device
            )

        return {
            Output.BATCH_LATENT: x,
            Output.DEFAULT_LATENT: x,
        }

    def decode(self, **encoding):
        out = self.sequential(encoding[Output.BATCH_LATENT])

        # out, _ = EncodingLevel.reduce(
        #     encodings=out,
        #     **encoding["sections"],
        #     reduced_to_sentence=encoding["reduced_to_sentence"],
        #     reduce_transformations=self.post_reduce,
        # )

        return {
            Output.LOGITS: out,
            Output.BATCH_LATENT: encoding[Output.BATCH_LATENT],
            Output.INT_PREDICTIONS: torch.argmax(out, dim=-1),
        }


class WETextClassifier(nn.Module):
    def collate_fn(self, batch: Sequence[Any]):
        return functools.partial(self.text_encoder.text_encoder.collate_fn, batch=batch)

    def __init__(
        self,
        metadata: MetaData,
        text_encoder: TextEncoder,
        batch_pre_reduce: Collection[EncodingLevel] = None,
        anchors_reduce: Collection[EncodingLevel] = None,
        batch_post_reduce: Collection[EncodingLevel] = None,
        **kwargs,
    ) -> None:
        """Simple model that uses convolutions.

        Args:
            metadata: the metadata object
        """
        super().__init__()
        pylogger.info(f"Instantiating <{self.__class__.__qualname__}>")

        self.metadata = metadata
        self.text_encoder: TextEncoder = instantiate(text_encoder)

        self.text_encoder.add_stopwords(stopwords=metadata.stopwords)

        n_classes: int = len(self.metadata.class_to_idx)

        self.pre_reduce: Set[EncodingLevel] = (
            set(batch_pre_reduce) if batch_pre_reduce is not None and len(batch_pre_reduce) > 0 else {}
        )
        self.anchors_reduce: Set[EncodingLevel] = (
            set(anchors_reduce) if anchors_reduce is not None and len(anchors_reduce) > 0 else {}
        )
        self.post_reduce: Set[EncodingLevel] = (
            set(batch_post_reduce) if batch_post_reduce is not None and len(batch_post_reduce) > 0 else {}
        )

        anchors = self.text_encoder.collate_fn(batch=metadata.anchor_samples)

        if not self.text_encoder.trainable:
            self.register_buffer("anchor_encodings", anchors["encodings"])
            self.register_buffer("anchor_words_per_text", anchors["sections"]["words_per_text"])
            self.register_buffer("anchor_words_per_sentence", anchors["sections"]["words_per_sentence"])
            self.register_buffer("anchor_sentences_per_text", anchors["sections"]["sentences_per_text"])

        # assert len(set.intersection(self.pre_reduce, self.post_reduce)) == 0
        assert all(x < y for x, y in itertools.product(self.pre_reduce, self.post_reduce))

        self.sequential = nn.Sequential(
            nn.Linear(in_features=300, out_features=300),
            DeepProjection(
                in_features=300,
                out_features=n_classes,
                dropout=0.1,
                num_layers=1,
                activation=nn.Tanh(),
            ),
            nn.ReLU(),
        )

    def set_finetune_mode(self):
        pass

    def encode(self, batch, device: Device):
        if "encodings" not in batch:
            batch = to_device(self.text_encoder.collate_fn(batch=batch), device=device)
        x = batch["encodings"]

        x, reduced_to_sentence = EncodingLevel.reduce(
            encodings=x, **batch["sections"], reduced_to_sentence=False, reduce_transformations=self.pre_reduce
        )

        return {"x": x, "reduced_to_sentence": reduced_to_sentence, "sections": batch["sections"]}

    def decode(self, **encoding):
        out = self.sequential(encoding["x"])

        out, _ = EncodingLevel.reduce(
            encodings=out,
            **encoding["sections"],
            reduced_to_sentence=encoding["reduced_to_sentence"],
            reduce_transformations=self.post_reduce,
        )

        return {
            Output.LOGITS: out,
            Output.DEFAULT_LATENT: EncodingLevel.reduce(
                encodings=encoding["x"],
                **encoding["sections"],
                reduced_to_sentence=encoding["reduced_to_sentence"],
                reduce_transformations=self.post_reduce,
            )[0],
            Output.BATCH_LATENT: encoding["x"],
            Output.INT_PREDICTIONS: torch.argmax(out, dim=-1),
        }


if __name__ == "__main__":
    for transformer_name in (
        "bert-base-cased",
        "bert-base-uncased",
        "google/electra-base-discriminator",
        "roberta-base",
        "albert-base-v2",
        "distilbert-base-uncased",
        "distilbert-base-cased",
        "xlm-roberta-base",
    ):
        transformer_config = AutoModel.from_pretrained(transformer_name).config.to_dict()
        # TODO: DistilBert doesn't have the "hidden_size" parameter :@
        transformer_encoding_dim = transformer_config["hidden_size" if "hidden_size" in transformer_config else "dim"]
        print(transformer_name, transformer_encoding_dim)
