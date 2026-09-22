"""Benchmark English stemmers for the ``text_en`` analyzer.

Run against a live OpenSearch:

    uv run python bench/stemming_bench.py [--url http://localhost:9200]

Why this is committed rather than a throwaway: the stemmer choice is a genuine recall/precision
trade, and it is the kind of decision that otherwise gets re-argued every few months from
memory. Two measurements matter and they pull in opposite directions:

* **pairs** - how many singular/plural pairs collapse to the same token. Misses cost recall.
* **mangled** - how many ordinary words are stemmed into something that is not a word.
  Over-stemming collides unrelated vocabulary and costs precision, which is the entire
  comparative advantage of the lexical leg over the dense one.

Result as of the initial mapping (OpenSearch 2.19):

    light_english     pairs 17/20  mangled 0     <- shipped
    kstem             pairs 17/20  mangled 0
    minimal_english   pairs 16/20  mangled 2
    porter2           pairs 20/20  mangled 6

porter2 wins on pairs and loses the argument: "organization" -> "organ" and "business" -> "busi"
across the whole corpus is a worse trade than missing cost/costs. The three remaining gaps are
morphological variation, which is what the dense leg is for.

Revisit at build step 6, when the golden set can say what these gaps actually cost in recall@50.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import httpx

POSSESSIVE: dict[str, str] = {"type": "stemmer", "language": "possessive_english"}

CHAINS: dict[str, list[Any]] = {
    "light_english": ["lowercase", "asciifolding", POSSESSIVE, {"type": "stemmer", "language": "light_english"}],
    "kstem": ["lowercase", "asciifolding", POSSESSIVE, "kstem"],
    "minimal_english": ["lowercase", "asciifolding", POSSESSIVE, {"type": "stemmer", "language": "minimal_english"}],
    "porter2": ["lowercase", "asciifolding", POSSESSIVE, {"type": "stemmer", "language": "porter2"}],
}

#: Enterprise-document vocabulary. Deliberately includes plain "-s" plurals, which are the most
#: common English form and which an earlier version of this benchmark omitted entirely -- the
#: omission made light_english look better than it is.
PAIRS: list[tuple[str, str]] = [
    ("cost", "costs"),
    ("report", "reports"),
    ("contract", "contracts"),
    ("payment", "payments"),
    ("document", "documents"),
    ("rule", "rules"),
    ("claim", "claims"),
    ("limit", "limits"),
    ("benefit", "benefits"),
    ("supplier", "suppliers"),
    ("agreement", "agreements"),
    ("term", "terms"),
    ("invoice", "invoices"),
    ("policy", "policies"),
    ("status", "statuses"),
    ("address", "addresses"),
    ("process", "processes"),
    ("employee", "employees"),
    ("company", "companies"),
    ("expense", "expenses"),
]

#: Words that must survive stemming intact. A stemmer that mangles these collides unrelated
#: vocabulary in the one leg whose job is precision.
MUST_SURVIVE: list[str] = [
    "business",
    "organization",
    "reimbursement",
    "compliance",
    "analysis",
    "premises",
    "data",
    "status",
]


def analyze(client: httpx.Client, filters: list[Any], text: str) -> list[str]:
    response = client.post("/_analyze", json={"tokenizer": "standard", "filter": filters, "text": text})
    response.raise_for_status()
    return [token["token"] for token in response.json()["tokens"]]


def run(url: str) -> int:
    with httpx.Client(base_url=url, timeout=30.0) as client:
        try:
            client.get("/_cluster/health").raise_for_status()
        except httpx.HTTPError as exc:
            print(f"OpenSearch not reachable at {url}: {exc}", file=sys.stderr)
            return 1

        results: dict[str, dict[str, Any]] = {}
        for name, filters in CHAINS.items():
            misses = [
                f"{singular}/{plural}"
                for singular, plural in PAIRS
                if analyze(client, filters, singular) != analyze(client, filters, plural)
            ]
            mangled = [
                f"{word}->{analyze(client, filters, word)[0]}"
                for word in MUST_SURVIVE
                if analyze(client, filters, word) != [word]
            ]
            results[name] = {
                "pairs_matched": len(PAIRS) - len(misses),
                "pairs_total": len(PAIRS),
                "misses": misses,
                "mangled": mangled,
            }

        width = max(len(name) for name in results)
        for name, row in results.items():
            print(f"{name:<{width}}  pairs {row['pairs_matched']}/{row['pairs_total']}  mangled {len(row['mangled'])}")
            if row["misses"]:
                print(f"{'':<{width}}    misses:  {', '.join(row['misses'])}")
            if row["mangled"]:
                print(f"{'':<{width}}    mangled: {', '.join(row['mangled'])}")

        print()
        print(json.dumps(results, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:9200")
    args = parser.parse_args()
    return run(args.url)


if __name__ == "__main__":
    raise SystemExit(main())
