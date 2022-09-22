import itertools
import logging
from typing import Any, Collection, Dict, Sequence, Set

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.types import Device
from transformers import AutoModel, PreTrainedModel

from rae.data.text.datamodule import EncodingLevel, MetaData
from rae.modules.attention import AttentionOutput
from rae.modules.blocks import DeepProjection
from rae.modules.enumerations import Output
from rae.modules.text.encoder import TextEncoder
from rae.utils.utils import chunk_iterable, to_device

pylogger = logging.getLogger(__name__)


class TextClassifier(nn.Module):
    def collate_fn(self, batch: Sequence[Any]):
        return self.self_encoder.text_encoder.collate_fn(batch=batch)

    def __init__(
        self,
        metadata: MetaData,
        relative_projection: DictConfig,
        text_encoder: TextEncoder,
        batch_pre_reduce: Collection[EncodingLevel] = None,
        anchors_reduce: Collection[EncodingLevel] = None,
        batch_post_reduce: Collection[EncodingLevel] = None,
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

        n_anchors = (
            len(anchors["sections"]["words_per_text"])
            if EncodingLevel.TEXT in self.anchors_reduce
            else (
                len(anchors["sections"]["words_per_sentence"])
                if EncodingLevel.SENTENCE in self.anchors_reduce
                else anchors["sections"]["words_per_sentence"].sum()
            )
        )
        self.relative_projection = instantiate(relative_projection, n_anchors=n_anchors, n_classes=n_classes)

        # assert len(set.intersection(self.pre_reduce, self.post_reduce)) == 0
        assert all(x < y for x, y in itertools.product(self.pre_reduce, self.post_reduce))

        self.sequential = nn.Sequential(
            nn.Linear(
                in_features=self.relative_projection.output_dim, out_features=self.relative_projection.output_dim
            ),
            DeepProjection(
                in_features=self.relative_projection.output_dim,
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

        with torch.no_grad():
            anchor_encodings, sections = (
                (
                    self.anchor_encodings,
                    dict(
                        words_per_text=self.anchor_words_per_text,
                        words_per_sentence=self.anchor_words_per_sentence,
                        sentences_per_text=self.anchor_sentences_per_text,
                    ),
                )
                if self.anchor_encodings is not None
                else (
                    (x := to_device(self.text_encoder.collate_fn(batch=self.anchor_batch), device=device)),
                    x["sections"],
                )
            )

            anchors, _ = EncodingLevel.reduce(
                encodings=anchor_encodings,
                **sections,
                reduced_to_sentence=False,
                reduce_transformations=self.anchors_reduce,
            )
        #
        attention_output = self.relative_projection.encode(x=x, anchors=anchors)

        return {
            **attention_output,
            "sections": batch["sections"],
            Output.ANCHORS_LATENT: anchors,
            Output.BATCH_LATENT: x,
            "reduced_to_sentence": reduced_to_sentence,
        }

    def decode(self, **encoding):
        x = self.relative_projection.decode(**encoding)
        out = self.sequential(x[AttentionOutput.OUTPUT])

        out, _ = EncodingLevel.reduce(
            encodings=out,
            **encoding["sections"],
            reduced_to_sentence=encoding["reduced_to_sentence"],
            reduce_transformations=self.post_reduce,
        )

        return {
            Output.LOGITS: out,
            Output.DEFAULT_LATENT: EncodingLevel.reduce(
                encodings=x,
                **encoding["sections"],
                reduced_to_sentence=encoding["reduced_to_sentence"],
                reduce_transformations=self.post_reduce,
            )[0],
            Output.BATCH_LATENT: encoding[Output.BATCH_LATENT],
            Output.ANCHORS_LATENT: encoding[Output.ANCHORS_LATENT],
            Output.INV_LATENTS: x[AttentionOutput.SIMILARITIES],
            Output.INT_PREDICTIONS: torch.argmax(out, dim=-1),
        }


class HFTextClassifier(nn.Module):
    def collate_fn(self, batch: Sequence[Any]):
        return self.text_encoder.collate_fn(batch=batch)

    def __init__(
        self,
        metadata: MetaData,
        relative_projection: DictConfig,
        text_encoder: TextEncoder,
        transformer_name: str,
        batch_pre_reduce: Collection[EncodingLevel] = None,
        anchors_reduce: Collection[EncodingLevel] = None,
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
        )

        self.finetune: bool = finetune

        if not finetune:
            self.transformer.requires_grad_(False)
            self.transformer.eval()

        self.pre_reduce: Set[EncodingLevel] = (
            set(batch_pre_reduce) if batch_pre_reduce is not None and len(batch_pre_reduce) > 0 else {}
        )
        self.anchors_reduce: Set[EncodingLevel] = (
            set(anchors_reduce) if anchors_reduce is not None and len(anchors_reduce) > 0 else {}
        )
        self.post_reduce: Set[EncodingLevel] = (
            set(batch_post_reduce) if batch_post_reduce is not None and len(batch_post_reduce) > 0 else {}
        )
        self.anchor_encodings = None

        self.relative_projection = instantiate(
            relative_projection, n_anchors=len(metadata.anchor_samples), n_classes=n_classes
        )

        # assert len(set.intersection(self.pre_reduce, self.post_reduce)) == 0
        assert all(x < y for x, y in itertools.product(self.pre_reduce, self.post_reduce))

        self.sequential = nn.Sequential(
            nn.LayerNorm(self.relative_projection.output_dim),
            nn.Linear(
                in_features=self.relative_projection.output_dim, out_features=self.relative_projection.output_dim
            ),
            DeepProjection(
                in_features=self.relative_projection.output_dim,
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
            if self.anchor_encodings is None:
                assert not self.text_encoder.trainable
                self.anchor_encodings = []
                for anchors_batch in chunk_iterable(self.metadata.anchor_samples, 128):
                    anchors_batch = to_device(self.text_encoder.collate_fn(batch=anchors_batch), device=device)
                    anchor_encodings = self.call_transformer(
                        encodings=anchors_batch["encodings"],
                        mask=anchors_batch["mask"],
                        sample_ids=anchors_batch["index"],
                    )
                    self.anchor_encodings.append(anchor_encodings)

                self.anchor_encodings = torch.cat(self.anchor_encodings, dim=0)
            anchors = self.anchor_encodings
            # anchors, _ = EncodingLevel.reduce(
            #     encodings=anchor_encodings,
            #     **sections,
            #     reduced_to_sentence=False,
            #     reduce_transformations=self.anchors_reduce,
            # )
            x = self.call_transformer(encodings=batch["encodings"], mask=batch["mask"], sample_ids=batch["index"]).to(
                device
            )

        attention_output = self.relative_projection.encode(x=x, anchors=anchors)

        return {
            **attention_output,
            # "sections": batch["sections"],
            Output.ANCHORS_LATENT: anchors,
            Output.BATCH_LATENT: x,
            Output.DEFAULT_LATENT: attention_output[AttentionOutput.SIMILARITIES],
            # "reduced_to_sentence": reduced_to_sentence,
        }

    def decode(self, **encoding):
        x = self.relative_projection.decode(**encoding)
        out = self.sequential(x[AttentionOutput.OUTPUT])

        # out, _ = EncodingLevel.reduce(
        #     encodings=out,
        #     **encoding["sections"],
        #     reduced_to_sentence=encoding["reduced_to_sentence"],
        #     reduce_transformations=self.post_reduce,
        # )

        return {
            Output.LOGITS: out,
            Output.BATCH_LATENT: encoding[Output.BATCH_LATENT],
            Output.ANCHORS_LATENT: encoding[Output.ANCHORS_LATENT],
            Output.INV_LATENTS: x[AttentionOutput.SIMILARITIES],
            Output.INT_PREDICTIONS: torch.argmax(out, dim=-1),
        }
