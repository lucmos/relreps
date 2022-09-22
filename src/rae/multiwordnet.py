# import nltk
#
# nltk.download("wordnet")
# nltk.download("omw-1.4")
from pathlib import Path
from typing import Mapping, Sequence

from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Lemma, Synset
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT

_LEMMA_SEP: str = ","

langs = wn.langs()
print(langs)
langs = [
    "eng",
    # "ita",
    "spa",
    "fra",
    "jpn",
]
with Path(PROJECT_ROOT / "data" / "synset_info.tsv").open("w", encoding="utf-8") as fw:
    fw.write("\t".join(["synset_id", "pos"] + langs))
    fw.write("\n")
    for synset in tqdm(wn.all_synsets(), desc="Iterating synsets"):
        synset: Synset
        lang2lemmas: Mapping[str, Sequence[Lemma]] = {lang: synset.lemmas(lang=lang) for lang in langs}
        lang2lemmas = {
            lang: [lemma for lemma in lemmas if lemma.name() != "GAP!"] for lang, lemmas in lang2lemmas.items()
        }
        if all(len(lemmas) > 0 for lemmas in lang2lemmas.values()):
            assert all(_LEMMA_SEP not in lemma.name() for lemmas in lang2lemmas.values() for lemma in lemmas)
            lang2lemmas: Mapping[str, str] = {
                lang: _LEMMA_SEP.join(lemma.name() for lemma in lemmas) for lang, lemmas in lang2lemmas.items()
            }
            line: str = "\t".join([synset.name(), synset.pos()] + [lemmas for lemmas in lang2lemmas.values()])
            fw.write(f"{line}\n")
