"""Text utilities shared by chunking, ranking and validation.

**Changing anything here changes chunk boundaries and therefore the chunker version**, which
changes the generation fingerprint and forces a reindex. That is intended -- it is the mechanism
that stops a "harmless" tokenizer tweak silently producing an index whose chunks no longer match
the ones an evaluation was measured on.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Final, Protocol

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\w+|[^\w\s]", re.UNICODE)

#: Candidate sentence boundaries: terminal punctuation, whitespace, then something that could
#: start a sentence. Python's ``re`` has no variable-width lookbehind, so abbreviations are
#: filtered in a second pass rather than excluded in the pattern.
_SENTENCE_BOUNDARY_RE: Final[re.Pattern[str]] = re.compile(r"([.!?])\s+(?=[\"“(\[]?[A-Z0-9])")

#: Words that end in a period without ending a sentence. Splitting after one of these produces
#: a fragment whose first clause is missing its subject, which is exactly the chunk boundary
#: that makes an embedding meaningless.
_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "inc",
        "ltd",
        "llc",
        "plc",
        "co",
        "corp",
        "gmbh",
        "ag",
        "bv",
        "nv",
        "sa",
        "etc",
        "vs",
        "eg",
        "ie",
        "cf",
        "al",
        "approx",
        "est",
        "no",
        "nr",
        "art",
        "sec",
        "fig",
        "ref",
        "para",
        "ch",
        "pp",
        "vol",
        "ed",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "mon",
        "tue",
        "wed",
        "thu",
        "fri",
        "sat",
        "sun",
    }
)

_TRAILING_WORD_RE: Final[re.Pattern[str]] = re.compile(r"([A-Za-z]+)$")


class Tokenizer(Protocol):
    """Anything that can count tokens.

    A Protocol so a real model tokenizer can be injected in production while CI keeps a
    deterministic, dependency-free counter.
    """

    def count(self, text: str) -> int: ...


class ApproxTokenizer:
    """A deterministic, dependency-free token counter.

    Counts word-ish runs and standalone punctuation, then applies a small multiplier for
    sub-word splitting. It is an *approximation* of a BPE tokenizer, deliberately: pulling in
    ``tiktoken`` or a HuggingFace tokenizer would put a model download in the CI path, which is
    the one thing the offline-first rule forbids.

    It runs about 10-20% under a real BPE count on English prose, which is the safe direction --
    chunks come out slightly smaller than the nominal target rather than overflowing a context
    budget computed from them.
    """

    id = "approx-1"

    def __init__(self, multiplier: float = 1.15) -> None:
        self.multiplier = multiplier

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, int(len(_WORD_RE.findall(text)) * self.multiplier))


DEFAULT_TOKENIZER: Final[ApproxTokenizer] = ApproxTokenizer()


def normalize(text: str) -> str:
    """NFC-normalize and collapse whitespace runs, preserving paragraph breaks.

    NFC matters for hashing: macOS hands over NFD filenames and text, Windows NFC. Without this
    a re-upload from a different machine re-embeds every chunk.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries, keeping abbreviations and initials intact.

    Only used when a single block exceeds the chunk cap -- normal chunking splits on structure,
    not on sentences. Two passes because Python's ``re`` cannot express a variable-width
    lookbehind: find every candidate boundary, then reject the ones preceded by an abbreviation
    or a single-letter initial.
    """
    if not text.strip():
        return []

    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        if match.group(1) == ".":
            preceding = text[start : match.start(1)]
            word_match = _TRAILING_WORD_RE.search(preceding)
            if word_match is not None:
                word = word_match.group(1)
                # "Sec." or "Dr." -- not a boundary.
                if word.lower() in _ABBREVIATIONS:
                    continue
                # "J." in "J. Smith" -- an initial, not a boundary.
                if len(word) == 1 and word.isupper():
                    continue
        candidate = text[start : match.end(1)].strip()
        if candidate:
            sentences.append(candidate)
        start = match.end()

    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences or [text.strip()]


def tokens(text: str) -> list[str]:
    """Lowercased word tokens. Used for near-duplicate detection, not for retrieval."""
    return [match.group().lower() for match in _WORD_RE.finditer(text) if match.group().isalnum()]


def simhash64(text: str) -> int:
    """A 64-bit SimHash for near-duplicate detection.

    Enterprise corpora repeat the same paragraph across a dozen documents. Without near-duplicate
    suppression at assembly time, a top-10 becomes one paragraph twelve times and the answer has
    a single source wearing twelve hats.

    Returned as a signed 64-bit integer because Postgres has no unsigned bigint.
    """
    vector = [0] * 64
    word_list = tokens(text)
    if not word_list:
        return 0
    for word in word_list:
        digest = int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += 1 if digest >> bit & 1 else -1
    unsigned = 0
    for bit in range(64):
        if vector[bit] > 0:
            unsigned |= 1 << bit
    return unsigned - (1 << 64) if unsigned >= 1 << 63 else unsigned


def hamming_distance(left: int, right: int) -> int:
    return ((left ^ right) & 0xFFFFFFFFFFFFFFFF).bit_count()


def jaccard(left: str, right: str) -> float:
    """Token-set overlap. The second near-duplicate signal, after SimHash."""
    left_set, right_set = set(tokens(left)), set(tokens(right))
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0
