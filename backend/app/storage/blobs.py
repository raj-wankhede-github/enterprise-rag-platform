"""Content-addressed blob storage.

The key is the SHA-256 of the bytes, which falls out of the dedup design rather than being a
separate decision: ``versioning.py`` already computes ``blob_sha256`` to answer "have we seen
these exact bytes before", and once that hash exists, using it as the storage key makes the same
file uploaded by two people occupy one object.

Two things that only matter at scale, which is the scale this is for:

**Keys are sharded two levels deep.** ``ab/cd/abcd1234...`` rather than ``abcd1234...``. A flat
directory with a million entries is slow to list on every filesystem worth naming and painful to
back up incrementally; on S3 the prefix spread also avoids hot-partitioning a single key range.
Two hex characters per level gives 65,536 leaves, which keeps a hundred-million-object store at
a few thousand per directory.

**Writes are idempotent by construction.** The key is a function of the content, so writing the
same blob twice is a no-op rather than a conflict -- which is what makes a retried ingestion job
safe. The store checks for existence before writing rather than trusting the caller to.

Deletion is refcounted at the *database* level (``blobs.refcount``), not here. The same bytes can
be two documents with different ACLs, and a store that deleted on the first document's removal
would silently empty the second.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Bytes read per chunk when hashing or streaming. Large enough that syscall overhead is
#: irrelevant, small enough that a 2 GB PDF does not become 2 GB of resident memory.
CHUNK_BYTES = 1024 * 1024


class BlobNotFoundError(Exception):
    """The blob is referenced by a row but absent from the store.

    Worth its own type because it means the database and the store have diverged, which is an
    operational problem rather than a bad request -- and the queue classifies it as permanent, so
    a job does not retry five times against a blob that will not appear.
    """


@dataclass(frozen=True, slots=True)
class BlobRef:
    sha256: str
    size: int
    #: Where the store put it. Recorded so a migration between backends can be resumed, and so
    #: an operator can find the object without re-deriving the sharding rule.
    key: str


class BlobStore(Protocol):
    async def put(self, data: bytes) -> BlobRef: ...
    async def get(self, sha256: str) -> bytes: ...
    async def exists(self, sha256: str) -> bool: ...
    async def delete(self, sha256: str) -> None: ...


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def storage_key(sha256: str) -> str:
    """``ab/cd/abcd1234…``. See the module docstring for why two levels."""
    if len(sha256) < 4:
        raise ValueError("a content hash must be at least 4 characters")
    return f"{sha256[:2]}/{sha256[2:4]}/{sha256}"


class FilesystemBlobStore:
    """Local disk. The development default, and correct for a single-node BYOC install.

    Writes land in a temporary file and are then atomically renamed into place. A process killed
    mid-write therefore leaves a temp file rather than a truncated blob under a content hash that
    promises different bytes -- which would be undetectable corruption, since every later reader
    trusts the key.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, sha256: str) -> Path:
        return self.root / storage_key(sha256)

    async def put(self, data: bytes) -> BlobRef:
        digest = sha256_of(data)
        path = self._path(digest)

        if path.exists():
            # Same content, same key: nothing to do. This is what makes a retried job free.
            return BlobRef(sha256=digest, size=len(data), key=storage_key(digest))

        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".partial")
        temp.write_bytes(data)
        temp.replace(path)
        return BlobRef(sha256=digest, size=len(data), key=storage_key(digest))

    async def get(self, sha256: str) -> bytes:
        path = self._path(sha256)
        if not path.exists():
            raise BlobNotFoundError(f"blob {sha256[:12]} is not in the store")
        return path.read_bytes()

    async def exists(self, sha256: str) -> bool:
        return self._path(sha256).exists()

    async def delete(self, sha256: str) -> None:
        self._path(sha256).unlink(missing_ok=True)


class S3BlobStore:
    """S3, or anything that speaks its API -- MinIO, R2, Azure via a gateway.

    Server-side encryption is passed through rather than assumed: a customer with their own KMS
    key expects it used, and a store that silently applied the bucket default would make their
    key configuration a no-op they would not discover until an audit.
    """

    def __init__(
        self,
        bucket: str,
        *,
        client: Any,
        prefix: str = "blobs",
        sse: str | None = None,
        sse_kms_key_id: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.client = client
        self.prefix = prefix.strip("/")
        self.sse = sse
        self.sse_kms_key_id = sse_kms_key_id

    def _key(self, sha256: str) -> str:
        return f"{self.prefix}/{storage_key(sha256)}"

    async def put(self, data: bytes) -> BlobRef:
        digest = sha256_of(data)
        key = self._key(digest)

        if await self.exists(digest):
            return BlobRef(sha256=digest, size=len(data), key=key)

        extra: dict[str, Any] = {}
        if self.sse:
            extra["ServerSideEncryption"] = self.sse
        if self.sse_kms_key_id:
            extra["SSEKMSKeyId"] = self.sse_kms_key_id

        await self.client.put_object(Bucket=self.bucket, Key=key, Body=data, **extra)
        return BlobRef(sha256=digest, size=len(data), key=key)

    async def get(self, sha256: str) -> bytes:
        try:
            response = await self.client.get_object(Bucket=self.bucket, Key=self._key(sha256))
            body: bytes = await response["Body"].read()
            return body
        except Exception as exc:
            if _is_missing(exc):
                raise BlobNotFoundError(f"blob {sha256[:12]} is not in the store") from exc
            raise

    async def exists(self, sha256: str) -> bool:
        try:
            await self.client.head_object(Bucket=self.bucket, Key=self._key(sha256))
            return True
        except Exception as exc:
            if _is_missing(exc):
                return False
            raise

    async def delete(self, sha256: str) -> None:
        await self.client.delete_object(Bucket=self.bucket, Key=self._key(sha256))


def _is_missing(exc: Exception) -> bool:
    """Whether an S3 error means "not there" rather than "could not tell".

    The distinction matters: treating a permissions failure or a network error as "absent" would
    have a caller re-upload a blob that exists, or -- far worse -- conclude a document's bytes
    are gone and mark it failed.
    """
    code = getattr(getattr(exc, "response", {}), "get", lambda _k, _d=None: None)("Error", {}) or {}
    status = str(code.get("Code", "")) if isinstance(code, dict) else ""
    return status in {"404", "NoSuchKey", "NotFound"} or exc.__class__.__name__ in {"NoSuchKey", "ClientError404"}


def build_blob_store(settings: Any) -> BlobStore:
    """Filesystem unless S3 is configured.

    Defaults to the filesystem rather than failing, so a fresh clone and a BYOC single-node
    install both work with no object storage at all.
    """
    backend = str(getattr(settings, "blob_backend", "filesystem")).lower()

    if backend in {"s3", "minio"}:
        import aioboto3  # imported here so a filesystem deployment needs no AWS dependency

        session = aioboto3.Session()
        client = session.client("s3", endpoint_url=getattr(settings, "blob_endpoint", None) or None)
        return S3BlobStore(
            str(getattr(settings, "blob_bucket", "erp-blobs")),
            client=client,
            sse=getattr(settings, "blob_sse", None),
            sse_kms_key_id=getattr(settings, "blob_sse_kms_key_id", None),
        )

    root = getattr(settings, "blob_path", None) or "./data/blobs"
    return FilesystemBlobStore(root)
