"""Pooling in the models service, tested from here.

``deploy/models/server.py`` runs in its own container on its own Python, so nothing else in this
suite touches it. Its pooling is tested anyway, because it holds the highest-consequence silent
bug in the whole ingestion path:

**Mean pooling that includes padding produces vectors that depend on batch composition.** The
same text embedded in two differently-shaped batches gets two different vectors. Nothing errors,
nothing logs, and the index is quietly inconsistent with itself -- retrieval simply works a bit
less well than it should, forever, in a way no assertion downstream would catch.

The module is loaded by path rather than imported, because its container's dependencies
(``optimum``, ``transformers``) are not installed here. Only ``_pool`` is exercised, and it needs
nothing but numpy.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SERVER = Path(__file__).resolve().parents[3] / "deploy" / "models" / "server.py"


def load_server() -> ModuleType:
    """Import server.py without its heavy dependencies.

    ``fastapi`` and ``pydantic`` are already here; ``optimum`` and ``transformers`` are imported
    inside functions rather than at module scope, which is what makes this possible -- and is
    itself worth preserving, since it also keeps the service's import time off the cold-start
    path.
    """
    if not SERVER.exists():  # pragma: no cover
        pytest.skip("deploy/models/server.py is not present")

    spec = importlib.util.spec_from_file_location("models_server_under_test", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def server() -> ModuleType:
    return load_server()


def hidden(rows: list[list[float]]) -> np.ndarray:
    """One sequence, ``len(rows)`` tokens, one dimension per row -- shaped (1, tokens, dim)."""
    return np.asarray([rows], dtype=np.float32)


# ------------------------------------------------------------------------------------------
# The bug this file exists for
# ------------------------------------------------------------------------------------------


def test_mean_pooling_excludes_padding(server: ModuleType) -> None:
    """The one that matters.

    Two real tokens and two padding positions. Including the padding would average over four
    positions and drag the vector toward whatever the padding embedding happens to be.
    """
    states = hidden([[1.0, 1.0], [3.0, 3.0], [99.0, 99.0], [99.0, 99.0]])
    mask = np.asarray([[1, 1, 0, 0]], dtype=np.int64)

    pooled = server._pool(states, mask, "mean")

    assert pooled.tolist() == [[2.0, 2.0]], "padding leaked into the mean"


def test_the_same_text_pools_identically_whatever_the_batch_pads_to(server: ModuleType) -> None:
    """The consequence of the bug above, stated as the property that must hold.

    A short passage batched with other short passages pads to 2; batched with a long one it pads
    to 8. If those produce different vectors, the index disagrees with itself depending on the
    order documents happened to arrive in.
    """
    real = [[1.0, 2.0], [3.0, 4.0]]

    short_batch = server._pool(hidden(real), np.asarray([[1, 1]], dtype=np.int64), "mean")
    long_batch = server._pool(
        hidden([*real, *([[0.0, 0.0]] * 6)]),
        np.asarray([[1, 1, 0, 0, 0, 0, 0, 0]], dtype=np.int64),
        "mean",
    )

    assert np.allclose(short_batch, long_batch)


def test_padding_with_non_zero_values_still_does_not_leak(server: ModuleType) -> None:
    """Padding positions are not guaranteed to hold zeros -- a model's embedding for the pad
    token is usually non-zero, which is exactly why masking rather than summing is required."""
    states = hidden([[2.0], [4.0], [1000.0]])
    mask = np.asarray([[1, 1, 0]], dtype=np.int64)

    assert server._pool(states, mask, "mean").tolist() == [[3.0]]


# ------------------------------------------------------------------------------------------
# CLS
# ------------------------------------------------------------------------------------------


def test_cls_pooling_takes_the_first_token(server: ModuleType) -> None:
    """BGE trained with CLS pooling. Using mean on a BGE checkpoint costs several points of
    recall and reports nothing, which is why the mode comes from the card beside the weights."""
    states = hidden([[7.0, 8.0], [1.0, 1.0], [2.0, 2.0]])
    mask = np.asarray([[1, 1, 1]], dtype=np.int64)

    assert server._pool(states, mask, "cls").tolist() == [[7.0, 8.0]]


def test_cls_pooling_ignores_the_mask_because_position_zero_is_never_padding(server: ModuleType) -> None:
    states = hidden([[5.0], [9.0]])
    assert server._pool(states, np.asarray([[1, 0]], dtype=np.int64), "cls").tolist() == [[5.0]]


def test_the_two_modes_disagree_which_is_why_the_card_records_one(server: ModuleType) -> None:
    states = hidden([[10.0], [20.0]])
    mask = np.asarray([[1, 1]], dtype=np.int64)

    assert server._pool(states, mask, "cls").tolist() != server._pool(states, mask, "mean").tolist()


# ------------------------------------------------------------------------------------------
# Shape and edges
# ------------------------------------------------------------------------------------------


def test_a_batch_pools_per_sequence(server: ModuleType) -> None:
    states = np.asarray(
        [
            [[1.0], [3.0], [0.0]],
            [[10.0], [0.0], [0.0]],
        ],
        dtype=np.float32,
    )
    mask = np.asarray([[1, 1, 0], [1, 0, 0]], dtype=np.int64)

    assert server._pool(states, mask, "mean").tolist() == [[2.0], [10.0]]


def test_an_all_padding_sequence_does_not_divide_by_zero(server: ModuleType) -> None:
    """Should not occur -- a tokenizer always emits at least one real token -- but a NaN here
    would propagate into a stored vector and poison every comparison against it."""
    pooled = server._pool(hidden([[1.0], [2.0]]), np.asarray([[0, 0]], dtype=np.int64), "mean")
    assert not np.isnan(pooled).any()


def test_pooling_returns_float32(server: ModuleType) -> None:
    """The vectors are serialized to JSON and stored as fp16. Anything wider is wasted precision
    the storage layer discards anyway."""
    pooled = server._pool(hidden([[1.0], [2.0]]), np.asarray([[1, 1]], dtype=np.int64), "mean")
    assert pooled.dtype == np.float32


# ------------------------------------------------------------------------------------------
# The card
# ------------------------------------------------------------------------------------------


def test_every_known_family_declares_a_pooling_mode() -> None:
    """A family without one would silently fall through to a default that is wrong half the
    time."""
    from importlib.util import module_from_spec, spec_from_file_location

    path = SERVER.parent / "export_model.py"
    spec = spec_from_file_location("export_model_under_test", path)
    assert spec and spec.loader
    export = module_from_spec(spec)
    spec.loader.exec_module(export)

    for family, mode in export.POOLING_BY_FAMILY.items():
        assert mode in {"cls", "mean"}, f"{family} declares an unknown pooling mode {mode!r}"


def test_bge_uses_cls_and_e5_uses_mean() -> None:
    """The two families most likely to be deployed, and they differ. Getting either wrong is a
    silent recall loss."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("export_model_under_test2", SERVER.parent / "export_model.py")
    assert spec and spec.loader
    export = module_from_spec(spec)
    spec.loader.exec_module(export)

    assert export.POOLING_BY_FAMILY["bge"] == "cls"
    assert export.POOLING_BY_FAMILY["e5"] == "mean"
    assert export.family_of("BAAI/bge-base-en-v1.5") == "bge"
    assert export.family_of("intfloat/e5-large-v2") == "e5"


def test_bge_and_e5_prefix_queries_differently() -> None:
    """Both expect a query prefix and they are not the same string. Omitting it does not error;
    it costs recall."""
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("export_model_under_test3", SERVER.parent / "export_model.py")
    assert spec and spec.loader
    export = module_from_spec(spec)
    spec.loader.exec_module(export)

    assert export.QUERY_PREFIX_BY_FAMILY["bge"] != export.QUERY_PREFIX_BY_FAMILY["e5"]
    assert export.QUERY_PREFIX_BY_FAMILY["bge"].strip().endswith(":")
    # E5 is the family that also prefixes passages; BGE does not.
    assert export.PASSAGE_PREFIX_BY_FAMILY.get("e5") == "passage: "
    assert export.PASSAGE_PREFIX_BY_FAMILY.get("bge", "") == ""
