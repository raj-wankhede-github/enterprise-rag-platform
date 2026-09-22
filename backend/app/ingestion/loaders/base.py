"""Loader contract and the routing probe.

Routing is by ``DocumentProbe``, not by file extension. The decisive signal for PDFs is
``chars_per_page``: a scanned page has a text layer of roughly nothing, and sending it to the
cheap loader yields a document of empty chunks that index cleanly and retrieve nothing. That
failure is invisible -- the ingest succeeds, the document appears in the UI, and it simply never
comes back from a search.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.ingestion.types import DocumentProbe, ExtractedDocument, LoaderCost

#: Below this, a PDF page is treated as scanned and routed to the expensive path. A page of body
#: text carries 1500-3000 characters; a scanned page carries header/footer artefacts at most.
SCANNED_PAGE_CHAR_THRESHOLD: float = 150.0


@runtime_checkable
class DocumentLoader(Protocol):
    name: str
    version: str
    #: "cheap" runs in-process. "expensive" means the parser container, which may not be
    #: deployed -- the registry must surface that as a clean error, not an import failure.
    cost: LoaderCost

    def supports(self, probe: DocumentProbe) -> bool: ...

    async def load(self, blob: bytes, probe: DocumentProbe) -> ExtractedDocument: ...


class LoaderRegistry:
    """Picks a loader for a probe.

    Cheap loaders are tried first and in registration order; the first that claims the probe
    wins. Expensive loaders are the fallback, so a deployment without the parser container still
    handles text formats and fails clearly on the rest.
    """

    def __init__(self, loaders: list[DocumentLoader] | None = None) -> None:
        self._loaders: list[DocumentLoader] = list(loaders or [])

    def register(self, loader: DocumentLoader) -> None:
        self._loaders.append(loader)

    @property
    def loaders(self) -> tuple[DocumentLoader, ...]:
        return tuple(self._loaders)

    def select(self, probe: DocumentProbe) -> DocumentLoader | None:
        for cost in ("cheap", "expensive"):
            for loader in self._loaders:
                if loader.cost == cost and loader.supports(probe):
                    return loader
        return None
