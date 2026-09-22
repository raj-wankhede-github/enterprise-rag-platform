"""Deterministic extraction. Authoritative, and not overridable by a model.

Everything here is a regex or a lookup, which means it is reproducible, free, testable offline,
and identical in CI and production. That is the whole argument: the parts of query understanding
that *can* be deterministic should be, because the parts that cannot are already enough surface
for non-determinism.

The identifier patterns are the most load-bearing. If ``SEC-4.2.1`` is not recognised, the exact
leg never runs, the fast path never triggers, and the query falls back to dense retrieval on an
opaque code -- which is the single worst case for embeddings.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Final

from app.query.types import DateRange, ExtractedFilters

#: ``TE-2025-01``, ``SUP_4471``, ``TKT-99812``. Two to six letters, a separator, then digits.
#: The separator is required: without it, ordinary capitalised words followed by a year match
#: ("POLICY 2025"). The digit run is ``\d+`` rather than ``\d{2,}`` because ``SEC-4.2.1`` is a
#: real reference form, and requiring two leading digits silently dropped it -- the exact leg
#: then never ran for section references, which is precisely the case it exists for.
_CODE: Final[re.Pattern[str]] = re.compile(r"\b[A-Z]{2,6}[-_]\d+(?:[-_.]\d+)*\b")

#: ``4.2.1``, ``§7.3``. Section numbers need at least one dot or they swallow every bare integer.
_SECTION: Final[re.Pattern[str]] = re.compile(r"(?:§\s*)?\b\d+\.\d+(?:\.\d+)*\b")

#: ``"exactly this phrase"``. A user who quotes is asking for a phrase match.
_QUOTED: Final[re.Pattern[str]] = re.compile(r"[\"“]([^\"”]{3,})[\"”]")

_ISO_DATE: Final[re.Pattern[str]] = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_YEAR: Final[re.Pattern[str]] = re.compile(r"\b(19|20)\d{2}\b")
_QUARTER: Final[re.Pattern[str]] = re.compile(r"\bq([1-4])\s*((?:19|20)\d{2})\b", re.IGNORECASE)

_MONTHS: Final[dict[str, int]] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTH_YEAR: Final[re.Pattern[str]] = re.compile(r"\b(" + "|".join(_MONTHS) + r")\s+((?:19|20)\d{2})\b", re.IGNORECASE)

#: ``author:alice``, ``type:policy``, ``source:sharepoint``. A user typing an operator means it.
_OPERATOR: Final[re.Pattern[str]] = re.compile(
    r"\b(author|type|doc_type|source|system)\s*:\s*(\"[^\"]+\"|\S+)", re.IGNORECASE
)

#: Words that make a question explicitly about the past. Without one of these, superseded
#: documents stay filtered out -- answering "what is the rule" from a withdrawn policy is the
#: most damaging failure this product has.
_HISTORICAL: Final[frozenset[str]] = frozenset(
    {
        "was",
        "were",
        "used",
        "previous",
        "previously",
        "historical",
        "historically",
        "former",
        "formerly",
        "old",
        "older",
        "prior",
        "before",
        "superseded",
        "withdrawn",
    }
)

_FOLLOW_UP_OPENERS: Final[frozenset[str]] = frozenset(
    {"what", "and", "but", "how", "why", "when", "where", "who", "ok", "okay", "also", "then"}
)
_PRONOUNS: Final[frozenset[str]] = frozenset(
    {"it", "its", "that", "this", "these", "those", "they", "them", "their", "he", "she", "him", "her"}
)

#: Patterns that must be flagged whatever a model later concludes. A query attempting to
#: redirect the system is content to record, not an instruction to weigh.
_INJECTION: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("instruction_override", re.compile(r"\bignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\b", re.IGNORECASE)),
    ("role_override", re.compile(r"\byou\s+are\s+now\b|\bact\s+as\s+(?:a|an)\b", re.IGNORECASE)),
    ("prompt_exfiltration", re.compile(r"\b(?:system\s+prompt|your\s+instructions|initial\s+prompt)\b", re.IGNORECASE)),
    (
        "guardrail_bypass",
        re.compile(r"\b(?:disregard|bypass|override)\s+(?:the\s+)?(?:rules|policy|guardrails|safety)\b", re.IGNORECASE),
    ),
    (
        "data_exfiltration",
        re.compile(r"\b(?:list|dump|show)\s+(?:all\s+)?(?:documents|tenants|users)\b", re.IGNORECASE),
    ),
)

#: The multi-part detector. Deliberately conservative: a false positive costs a wasted LLM call
#: and three retrievals, so it only fires on shapes that are reliably two questions.
_MULTIPART: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"\?\s*\S.*\?"),  # two question marks
    re.compile(r"\b(?:and|also)\s+(?:what|how|when|where|who|why|can|do|does|is|are)\b", re.IGNORECASE),
    re.compile(r"\bcompare\b.*\b(?:and|with|to|versus|vs)\b", re.IGNORECASE),
    re.compile(r"\bdifference between\b.*\band\b", re.IGNORECASE),
)


def extract_identifiers(text: str) -> tuple[str, ...]:
    """Codes, section numbers and quoted phrases, in that order of confidence.

    Code spans are masked before section matching runs. Otherwise ``SEC-4.2.1`` yields both the
    whole code and a bare ``4.2.1``, and the bare form is a far weaker search term that dilutes
    the exact leg with a token matching every document containing that section number.
    """
    found: list[str] = [match.group().strip() for match in _CODE.finditer(text)]

    without_codes = _CODE.sub(lambda match: " " * len(match.group()), text)
    found.extend(match.group().replace("§", "").strip() for match in _SECTION.finditer(without_codes))
    found.extend(match.group(1).strip() for match in _QUOTED.finditer(text))
    return tuple(dict.fromkeys(found))


def extract_filters(text: str, *, today: date | None = None) -> ExtractedFilters:
    """Field operators and absolute dates. Relative dates are the LLM's job, not a regex's."""
    stamp = today or date.today()
    doc_types: list[str] = []
    sources: list[str] = []
    authors: list[str] = []

    for match in _OPERATOR.finditer(text):
        key, value = match.group(1).lower(), match.group(2).strip('"')
        if key in ("type", "doc_type"):
            doc_types.append(value.lower())
        elif key in ("source", "system"):
            sources.append(value.lower())
        elif key == "author":
            authors.append(value.lower())

    return ExtractedFilters(
        doc_types=tuple(dict.fromkeys(doc_types)),
        source_systems=tuple(dict.fromkeys(sources)),
        authors=tuple(dict.fromkeys(authors)),
        published=extract_date_range(text, today=stamp),
        historical=is_historical(text),
    )


def extract_date_range(text: str, *, today: date | None = None) -> DateRange:
    """Absolute dates only.

    "Last quarter" is handled because it is unambiguous relative to a known date. Anything
    fuzzier ("since the reorg", "recently") is left to the LLM, which at least has context a
    regex does not.
    """
    stamp = today or date.today()

    iso = _ISO_DATE.search(text)
    if iso:
        found = date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        return DateRange(after=found, before=found)

    month_year = _MONTH_YEAR.search(text)
    if month_year:
        month = _MONTHS[month_year.group(1).lower()]
        year = int(month_year.group(2))
        return DateRange(after=date(year, month, 1), before=_end_of_month(year, month))

    quarter = _QUARTER.search(text)
    if quarter:
        index, year = int(quarter.group(1)), int(quarter.group(2))
        start_month = 3 * (index - 1) + 1
        return DateRange(after=date(year, start_month, 1), before=_end_of_month(year, start_month + 2))

    if re.search(r"\blast quarter\b", text, re.IGNORECASE):
        start_month = 3 * ((stamp.month - 1) // 3) + 1
        start = date(stamp.year, start_month, 1)
        previous_end = start - timedelta(days=1)
        previous_start = date(previous_end.year, 3 * ((previous_end.month - 1) // 3) + 1, 1)
        return DateRange(after=previous_start, before=previous_end)

    year_match = _YEAR.search(text)
    # Only when the year is not already part of a captured identifier: "TE-2025-01" is a
    # reference, not a request for everything published in 2025.
    if year_match and not any(year_match.group() in token for token in extract_identifiers(text)):
        year = int(year_match.group())
        return DateRange(after=date(year, 1, 1), before=date(year, 12, 31))

    return DateRange()


def _end_of_month(year: int, month: int) -> date:
    if month >= 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def is_historical(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & _HISTORICAL)


def injection_flags(text: str) -> tuple[str, ...]:
    """Non-removable. A model must not be able to reason away an injection attempt."""
    return tuple(name for name, pattern in _INJECTION if pattern.search(text))


def looks_like_follow_up(text: str, *, has_history: bool) -> bool:
    """Whether the query needs its antecedent resolved before it means anything.

    Gates whether the LLM runs at all, which is why it is a rule: paying for a rewrite call on
    every query would double the latency of the common case to fix the uncommon one.
    """
    if not has_history:
        return False
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return False
    if words[0] in _FOLLOW_UP_OPENERS and len(words) <= 8:
        return True
    if set(words) & _PRONOUNS:
        return True
    return len(words) <= 4


def looks_multipart(text: str) -> bool:
    return any(pattern.search(text) for pattern in _MULTIPART)


def word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


_INTERROGATIVE: Final[frozenset[str]] = frozenset(
    {
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "explain",
        "summarise",
        "summarize",
        "describe",
        "compare",
    }
)


def is_interrogative(text: str) -> bool:
    words = re.findall(r"[a-z]+", text.lower())
    return bool(words) and (words[0] in _INTERROGATIVE or "?" in text)
