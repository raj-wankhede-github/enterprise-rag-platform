"""Export a cross-encoder to int8 ONNX at image build time.

Quantization is applied here rather than at runtime so the served artefact is exactly what was
benchmarked. A service that quantizes on boot has a different model in every replica's memory
than the one whose latency was measured.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--quantize",
        default="int8",
        choices=["int8", "none"],
        help="int8 roughly doubles throughput for about half a point of nDCG",
    )
    args = parser.parse_args()

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = ORTModelForSequenceClassification.from_pretrained(args.model, export=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

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

    print(f"exported {args.model} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
