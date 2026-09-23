"""Export a model to int8 ONNX at image build time.

Two kinds, because the service serves two jobs that cannot share a model:

* ``sequence-classification`` -- the cross-encoder. Takes a (query, passage) *pair* and returns
  one relevance score. Cannot produce an embedding: there is no per-text vector to extract,
  only a judgement about a pair.
* ``feature-extraction`` -- the bi-encoder. Takes one text and returns a vector. Cannot rerank
  well, because it never sees the query and the passage together.

Quantization happens here rather than at runtime so the served artefact is exactly what was
benchmarked. A service that quantizes on boot has a different model in every replica's memory
than the one whose latency was measured.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

#: How each model family pools token vectors into one sentence vector.
#:
#: Getting this wrong does not error -- it produces vectors that are merely *worse*, which is the
#: hardest kind of bug to notice. BGE trained with CLS pooling and E5 with mean pooling; using
#: mean on a BGE model costs several points of recall and nothing anywhere reports it.
POOLING_BY_FAMILY: dict[str, str] = {
    "bge": "cls",
    "gte": "cls",
    "e5": "mean",
    "all-minilm": "mean",
    "all-mpnet": "mean",
    "nomic": "mean",
}

#: Prefixes these families expect on a *query* but not on a passage.
#:
#: The asymmetry is trained in. Omitting it silently costs recall, which is why it is recorded
#: in the artefact rather than left for a caller to remember.
QUERY_PREFIX_BY_FAMILY: dict[str, str] = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "e5": "query: ",
    "gte": "",
}

PASSAGE_PREFIX_BY_FAMILY: dict[str, str] = {
    "e5": "passage: ",
}


def family_of(model_name: str) -> str:
    lowered = model_name.lower()
    for family in POOLING_BY_FAMILY:
        if family in lowered:
            return family
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--task",
        default="sequence-classification",
        choices=["sequence-classification", "feature-extraction"],
        help="cross-encoder for reranking, bi-encoder for embeddings",
    )
    parser.add_argument(
        "--quantize",
        default="int8",
        choices=["int8", "none"],
        help="int8 roughly doubles throughput for about half a point of nDCG",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.task == "feature-extraction":
        from optimum.onnxruntime import ORTModelForFeatureExtraction

        model = ORTModelForFeatureExtraction.from_pretrained(args.model, export=True)
    else:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        model = ORTModelForSequenceClassification.from_pretrained(args.model, export=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    if args.task == "feature-extraction":
        _write_embedding_card(out, args.model, quantized=args.quantize == "int8")

    if args.quantize == "int8":
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig

        quantizer = ORTQuantizer.from_pretrained(out)
        config = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
        quantized = out / "quantized"
        quantizer.quantize(save_dir=quantized, quantization_config=config)
        for item in quantized.iterdir():
            shutil.copy2(item, out / item.name)
        shutil.rmtree(quantized, ignore_errors=True)

    print(f"exported {args.model} ({args.task}) to {out}")
    return 0


def _write_embedding_card(out: Path, model_name: str, *, quantized: bool) -> None:
    """Record how this model must be used, next to the weights.

    The pooling mode, the query prefix and the dimension are properties of the *checkpoint*, and
    a server that hard-codes them is a server that silently produces bad vectors the day someone
    changes ``EMBEDDING_MODEL``. Writing them beside the artefact makes swapping the model a
    build argument rather than a code change.

    ``embedder_id`` is the string that enters the generation fingerprint, so it carries
    everything that changes the vector space: the checkpoint, the dimension and whether the
    weights are quantized. Two indices built with different values can never blend.
    """
    family = family_of(model_name)
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    dimension = int(config.get("hidden_size") or config.get("d_model") or 768)

    card = {
        "model": model_name,
        "family": family,
        "dimension": dimension,
        "pooling": POOLING_BY_FAMILY.get(family, "mean"),
        "query_prefix": QUERY_PREFIX_BY_FAMILY.get(family, ""),
        "passage_prefix": PASSAGE_PREFIX_BY_FAMILY.get(family, ""),
        "normalized": True,
        "quantized": quantized,
        "embedder_id": f"{model_name.split('/')[-1]}@{dimension}@{'int8' if quantized else 'fp32'}",
    }
    (out / "embedding_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"embedding card: {card['embedder_id']} pooling={card['pooling']}")


if __name__ == "__main__":
    raise SystemExit(main())
