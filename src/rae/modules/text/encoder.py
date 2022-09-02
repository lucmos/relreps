import logging
from abc import abstractmethod
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

import fasttext
import gensim.downloader
import torch
from gensim.models import KeyedVectors
from spacy.tokens import Doc, Span, Token
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BatchEncoding, PreTrainedModel, PreTrainedTokenizer

from nn_core.common import PROJECT_ROOT

import rae  # noqa
from rae.data.text.datamodule import SpacyManager

pylogger = logging.getLogger(__name__)


class TextEncoder(nn.Module):
    def __init__(self, trainable: bool):
        super().__init__()
        self.trainable: bool = trainable

    @abstractmethod
    def add_stopwords(self, stopwords: Set[str]):
        raise NotImplementedError

    @abstractmethod
    def encoding_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str) -> Optional[Sequence[torch.Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def save(self, dst_dir: Path):
        raise NotImplementedError

    @classmethod
    def load(cls, src_dir: Path):
        data = torch.load(src_dir / "text_encoder.pt")
        encoder_class = data["encoder_class"]
        del data["encoder_class"]

        return encoder_class(**data)

    def collate_fn(self, batch: Sequence[Mapping[str, Any]]):
        """Custom collate function for dataloaders with access to split and metadata.

        Args:
            samples: A list of samples coming from the Dataset to be merged into a batch
            device: The Device to transfer the batch to

        Returns:
            A batch generated from the given samples
        """
        text_encodings: Sequence[Sequence[torch.Tensor]] = [self.encode(text=sample["data"]) for sample in batch]
        skipped_batch = [sample for sample, text_encoding in zip(batch, text_encodings) if text_encoding is None]
        batch = [sample for sample, text_encoding in zip(batch, text_encodings) if text_encoding is not None]
        if len(skipped_batch) != 0:
            pylogger.warning(
                f"Skipping {len(skipped_batch)} samples: {Counter(sample['class'] for sample in skipped_batch)}"
            )
        text_encodings = [text_encoding for text_encoding in text_encodings if text_encoding is not None]

        batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}

        # encodings ~ (sample_index, sentence_index, word_index)

        encodings = torch.cat(
            [sentence_encoding for text_encoding in text_encodings for sentence_encoding in text_encoding], dim=0
        )

        words_per_sentence = torch.tensor([sentence.size(0) for text in text_encodings for sentence in text])
        words_per_text = torch.tensor([sum(sentence.size(0) for sentence in text) for text in text_encodings])
        sentences_per_text = torch.tensor([len(text) for text in text_encodings])

        classes = batch["class"]
        targets = torch.as_tensor(batch["target"])

        sections = dict(
            words_per_sentence=words_per_sentence, words_per_text=words_per_text, sentences_per_text=sentences_per_text
        )

        other_params = {
            k: v
            for k, v in batch.items()
            if k
            not in {
                "class",
                "target",
            }
        }
        return dict(encodings=encodings, classes=classes, targets=targets, sections=sections, **other_params)


class FastTextEncoder(TextEncoder):
    def add_stopwords(self, stopwords: Set[str]):
        self.stopwords = set.union(self.stopwords, stopwords)

    def encoding_dim(self) -> int:
        return 300

    def save(self, dst_dir: Path):
        dst_path: Path = dst_dir / "text_encoder.pt"
        assert not dst_path.exists()
        torch.save(
            dict(
                encoder_class=FastTextEncoder,
                language=self.language,
                lemmatize=self.lemmatize,
            ),
            dst_path,
        )

    def __init__(self, language: str, lemmatize: bool):
        super().__init__(trainable=False)
        self.language: str = language
        self.lemmatize: bool = lemmatize

        self.model = fasttext.load_model(str(PROJECT_ROOT / "data" / "fasttext" / f"cc.{language}.300.bin"))
        self.pipeline = SpacyManager.instantiate(language)
        self.stopwords = self.pipeline.Defaults.stop_words

    @lru_cache(maxsize=50_000)
    def _encode_token(self, token: str) -> torch.Tensor:
        return torch.tensor(self.model[token])

    def encode(self, text: str) -> Optional[Sequence[torch.Tensor]]:
        document: Doc = self.pipeline(text=text)
        sentences: List[Span] = list(document.sents)
        sentences: List[List[Token]] = [list(sentence) for sentence in sentences]
        # Go to string representation and lemmatize (if needed)
        encoding: Sequence[Sequence[str]] = [
            [token.lemma_ if self.lemmatize else token.text for token in sentence] for sentence in sentences
        ]
        # Skip stopwords
        encoding: Sequence[Sequence[str]] = [
            [token for token in sentence if token.lower() not in self.stopwords] for sentence in encoding
        ]
        encoding = [sentence for sentence in encoding if len(sentence) > 0]
        assert len(encoding) > 0

        encoding: Sequence[List[torch.Tensor]] = [
            [self._encode_token(token=token) for token in sentence] for sentence in encoding
        ]

        # Now we can stack sentence representations
        encoding: Sequence[torch.Tensor] = [torch.stack(sentence_encoding, dim=0) for sentence_encoding in encoding]

        return encoding


class GensimEncoder(TextEncoder):
    @staticmethod
    def _build_vector_models():
        # available_models = set(gensim.downloader.info()["models"].keys())
        model_name2bin_mode = {
            "local_fasttext": False,
            # "fasttext-wiki-news-subwords-300": False,
            # "conceptnet-numberbatch-17-06-300": False,
            "word2vec-google-news-300": True,
            "glove-wiki-gigaword-300": False,
        }
        # assert all(model in available_models for model in model_name2bin_mode.keys())

        model_name2path: Mapping[str, Path] = {
            model_name: Path(gensim.downloader.load(model_name, return_path=True))
            if model_name != "local_fasttext"
            else (PROJECT_ROOT / "data" / "fasttext" / "cc.en.300.vec")
            for model_name in tqdm(model_name2bin_mode.keys(), desc="Downloading models (with cache)")
        }
        restricted_dir: Path = Path(gensim.downloader.BASE_DIR) / "restricted"
        restricted_dir.mkdir(exist_ok=True, parents=True)

        key2occ = {}
        model_name2vectors: Dict[str, KeyedVectors] = {}
        pylogger.debug("Loading original vectors...")
        for model_name, model_path in model_name2path.items():
            pylogger.debug(f"Loading {model_name}")
            weights = KeyedVectors.load_word2vec_format(
                str(model_path), binary=model_name2bin_mode[model_name], encoding="utf-8"
            )
            model_name2vectors[model_name] = weights
            for word in weights.key_to_index.keys():
                key2occ.setdefault(word, 0)
                key2occ[word] += 1

        valid_keys: Set[str] = {key for key, occ in key2occ.items() if occ == len(model_name2path)}
        pylogger.debug(f"Keeping only {len(valid_keys)} words/vectors")
        for model_name, weights in tqdm(model_name2vectors.items(), desc="Writing restricted vectors"):
            out_path: Path = restricted_dir / f"{model_name}.txt"

            pylogger.debug(f"Restricting {model_name}")
            restricted_vectors = KeyedVectors(vector_size=weights.vector_size)

            key2weights = {key: weights.get_vector(key=key) for key in weights.key_to_index.keys() if key in valid_keys}
            keys, weights = list(zip(*key2weights.items()))

            restricted_vectors.add_vectors(keys=keys, weights=list(weights))

            pylogger.debug(f"Storing restricted {model_name}")
            restricted_vectors.save_word2vec_format(str(out_path))

    def add_stopwords(self, stopwords: Set[str]):
        self.stopwords = set.union(self.stopwords, stopwords)

    def encoding_dim(self) -> int:
        return 300

    def save(self, dst_dir: Path):
        dst_path: Path = dst_dir / "text_encoder.pt"
        assert not dst_path.exists()
        torch.save(
            dict(
                encoder_class=FastTextEncoder,
                language=self.language,
                lemmatize=self.lemmatize,
            ),
            dst_path,
        )

    def __init__(self, language: str, lemmatize: bool, model_name: str):
        super().__init__(trainable=False)
        self.language: str = language
        self.lemmatize: bool = lemmatize

        self.model: KeyedVectors = KeyedVectors.load_word2vec_format(
            fname=str((Path(gensim.downloader.BASE_DIR) / "restricted") / f"{model_name}.txt")
        )
        self.pipeline = SpacyManager.instantiate(language)
        self.stopwords = self.pipeline.Defaults.stop_words

    @lru_cache(maxsize=50_000)
    def _encode_token(self, token: str) -> Optional[torch.Tensor]:
        if token not in self.model.key_to_index:
            token = token.lower()
        if token not in self.model.key_to_index:
            return None

        return torch.tensor(self.model[token])

    @torch.no_grad()
    def encode(self, text: str) -> Optional[Sequence[torch.Tensor]]:
        document: Doc = self.pipeline(text=text)
        sentences: List[Span] = list(document.sents)
        sentences: List[List[Token]] = [list(sentence) for sentence in sentences]
        # Go to string representation and lemmatize (if needed)
        encoding: Sequence[Sequence[str]] = [
            [token.lemma_ if self.lemmatize else token.text for token in sentence] for sentence in sentences
        ]
        # Skip stopwords
        encoding: Sequence[Sequence[str]] = [
            [token for token in sentence if token.lower() not in self.stopwords] for sentence in encoding
        ]
        encoding: Sequence[List[torch.Tensor]] = [
            [self._encode_token(token=token) for token in sentence] for sentence in encoding
        ]

        encoding: Sequence[List[torch.Tensor]] = [
            [token_encoding for token_encoding in sentence if token_encoding is not None] for sentence in encoding
        ]

        encoding: Sequence[List[torch.Tensor]] = [sentence for sentence in encoding if len(sentence) > 0]
        if len(encoding) == 0:
            return None

        # Now we can stack sentence representations
        encoding: Sequence[torch.Tensor] = [torch.stack(sentence_encoding, dim=0) for sentence_encoding in encoding]

        return encoding


class TransformerEncoder(TextEncoder):
    def add_stopwords(self, stopwords: Set[str]):
        pass

    def encoding_dim(self) -> int:
        transformer_config = self.transformer.config.to_dict()
        transformer_encoding_dim = transformer_config["hidden_size" if "hidden_size" in transformer_config else "dim"]

        return transformer_encoding_dim

    def save(self, dst_dir: Path):
        dst_path: Path = dst_dir / "text_encoder.pt"
        assert not dst_path.exists()
        torch.save(
            dict(
                encoder_class=TransformerEncoder,
                transformer_name=self.transformer_name,
                encoding_level=self.encoding_level,
            ),
            dst_path,
        )

    def __init__(self, transformer_name: str, trainable: bool):
        super().__init__(trainable=trainable)
        self.transformer_name: str = transformer_name

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(transformer_name, use_fast=True)
        self.transformer: PreTrainedModel = AutoModel.from_pretrained(
            transformer_name, output_hidden_states=True, return_dict=True
        ).eval()

    @torch.no_grad()
    def encode(self, text: str) -> Optional[Sequence[torch.Tensor]]:
        encoding: BatchEncoding = self.tokenizer(text, return_tensors="pt", truncation=True)
        encoding: torch.Tensor = self.transformer(**encoding)["hidden_states"][-1]
        # encoding ~ (text, bpe, hidden)
        # TODO: remove special tokens here?
        encoding: torch.Tensor = encoding.squeeze(dim=0)[1:-1, :]
        # encoding ~ (bpe, hidden)

        # TODO: support sentence level

        return [encoding]


if __name__ == "__main__":

    GensimEncoder._build_vector_models()
