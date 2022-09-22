#!python
import gzip
from enum import Enum
from itertools import chain, combinations
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

from rae import PROJECT_ROOT

WIKIMATRIX_DIR: Path = PROJECT_ROOT / "data" / "wikimatrix"


class Language(Enum):
    EN = "en"
    # IT = "it"
    FR = "fr"
    ES = "es"
    # DE = "de"
    # ZH = "zh"
    # RU = "ru"
    JA = "ja"
    # AR = "ar"


def read_sentences(lang: Language, threshold: float) -> Tuple[List[str], List[str]]:
    lang: str = lang.value
    with open(str(WIKIMATRIX_DIR / f"WikiMatrix.en-{lang}-{threshold}.txt.{lang}")) as lang_f, open(
        str(WIKIMATRIX_DIR / f"WikiMatrix.en-{lang}-{threshold}.txt.en")
    ) as en_f:
        en_sentences = [x.strip() for x in en_f]
        lang_sentences = [x.strip() for x in lang_f]
        assert len(en_sentences) == len(lang_sentences)

        # Skip 'bad' sentences (those which are the same across languages)
        sentences = [x for x in zip(en_sentences, lang_sentences) if x[0] != x[1]]
        en_sentences, lang_sentences = list(zip(*sentences))

        return en_sentences, lang_sentences


def powerset(iterable):
    iterable = set(iterable)
    return chain(*map(lambda x: combinations(iterable, x), range(0, len(iterable) + 1)))


def explore_intersections(language2threshold: Dict[Language, float]):
    lang2sentences = {lang: read_sentences(lang, threshold) for lang, threshold in language2threshold.items()}
    languages = language2threshold.keys()

    for set_languages in powerset(languages):
        if len(set_languages) == 0:
            continue
        sentences = list(map(lambda x: set(lang2sentences[x][0]), set_languages))
        intersection = len(set.intersection(*sentences))

        set_languages = [x.value for x in set_languages]
        print(f"{set_languages} = {intersection} sentences")


def build_parallel_corpus(language2threshold: Dict[Language, float], dst_dir: Path):
    lang2sentences: Dict[Language, Tuple[List[str], List[str]]] = {
        lang: read_sentences(lang, threshold) for lang, threshold in language2threshold.items()
    }
    # Store only the English sentences for each language
    lang2en_sentences: Dict[Language, List[str]] = {lang: sentences[0] for lang, sentences in lang2sentences.items()}

    # Compute the intersection between English sentences across all languages
    en_sentences = list(set(x) for x in lang2en_sentences.values())
    en_intersection = set.intersection(*en_sentences)
    # en_intersection = {
    #     sentence
    #     for sentence in en_intersection
    #     if len(sentence) > 15 and sentence[0].isalpha() and not sentence[0].isdigit()
    # }
    # Transform the lang2sentences to easily iterate over couples
    lang2sentences: Dict[Language, Sequence[Tuple[str, str]]] = {
        lang: list(zip(*sentences)) for lang, sentences in lang2sentences.items()
    }

    # Keep only sentences in the intersection
    lang2sentences: Dict[Language, Sequence[Tuple[str, str]]] = {
        lang: [
            (en_sentence, lang_sentence) for en_sentence, lang_sentence in sentences if en_sentence in en_intersection
        ]
        for lang, sentences in lang2sentences.items()
    }

    # Sort all the sentences by the English version
    lang2sentences: Dict[Language, Iterator[Tuple[str, str]]] = {
        lang: sorted(sentences, key=lambda x: x[0]) for lang, sentences in lang2sentences.items()
    }

    # Now that they are aligned, remove the English version
    lang2sentences: Dict[Language, List[str]] = {
        lang: list(zip(*sentences))[1] for lang, sentences in lang2sentences.items()
    }
    lang2sentences[Language.EN] = sorted(en_intersection)

    dst_dir.mkdir(exist_ok=True, parents=True)
    for language, sentences in lang2sentences.items():
        out_file: Path = dst_dir / (
            f"WikiMatrix.aligned.%s.txt.{language.value}"
            % "-".join(["_".join([lang.value, str(t)]) for lang, t in language2threshold.items()])
        )
        with open(str(out_file), "w", encoding="utf-8") as fw:
            for sentence in sentences:
                fw.write(sentence)
                fw.write("\n")
    merged_file: Path = dst_dir / (
        "WikiMatrix.aligned.%s.txt"
        % "-".join(["_".join([lang.value, str(t)]) for lang, t in language2threshold.items()])
    )
    with open(str(merged_file), "w", encoding="utf-8") as fw:
        language_order = [Language.EN, *language2threshold.keys()]
        # head = '\t'.join(language.value for language in language_order)
        # fw.write(f'{head}\n')
        lang_sentences = zip(*(lang2sentences[language] for language in language_order))
        for sentences in lang_sentences:
            # fw.write('\t'.join(sentences))
            # fw.write('\n')
            for sentence in sentences:
                fw.write(sentence)
                fw.write("\n")
            fw.write("\n")


def extract(language2thresholds: Dict[Language, List[float]]):
    for language, thresholds in language2thresholds.items():
        for threshold in thresholds:
            ordered = tuple(sorted([language.value, Language.EN.value]))
            tsv: str = "WikiMatrix.%s-%s.tsv.gz" % ordered
            src_lang_out = f"WikiMatrix.{language.EN.value}-{language.value}-{threshold}.txt.{ordered[0]}"
            trg_lang_out = f"WikiMatrix.{language.EN.value}-{language.value}-{threshold}.txt.{ordered[1]}"

            nl = 0
            nw_src = 0
            nw_trg = 0
            print(f"Processing {tsv} with threshold {threshold}")
            with gzip.open(WIKIMATRIX_DIR / tsv, "rt", encoding="utf-8") as tsv:
                with open(WIKIMATRIX_DIR / src_lang_out, "wt", encoding="utf-8") as fsrc:
                    with open(WIKIMATRIX_DIR / trg_lang_out, "wt", encoding="utf-8") as ftrg:
                        while True:
                            line = tsv.readline()
                            if not line:
                                break
                            fields = line.split("\t")
                            cur_src = len(fields[1].split())
                            cur_trg = len(fields[2].split())
                            if float(fields[0]) < threshold:
                                break
                            fsrc.write(fields[1].strip() + "\n")
                            ftrg.write(fields[2].strip() + "\n")
                            nw_src += cur_src
                            nw_trg += cur_trg
                            nl += 1
                            if nl % 100000 == 0:
                                print("\r - {:d} lines read".format(nl), end="")

            print("\r - wrote {:d} lines".format(nl))
            print(" - with {:d} source and {:d} target words".format(nw_src, nw_trg))
            print(" - last threshold is {:.4f}".format(float(fields[0])))


if __name__ == "__main__":
    do_extract: bool = False
    if do_extract:
        language2thresholds = {
            Language.FR: [1.05, 1.06],
            Language.ES: [1.05, 1.06],
            Language.JA: [1.05, 1.06],
        }
        extract(language2thresholds)

    language2threshold = {
        Language.FR: 1.06,
        Language.ES: 1.06,
        Language.JA: 1.06,
    }
    # Sort to get a unique file name as output
    language2threshold = {k: v for k, v in sorted(language2threshold.items(), key=lambda x: x[0].value)}
    explore_intersections(language2threshold)
    build_parallel_corpus(language2threshold, WIKIMATRIX_DIR / "aligned")
