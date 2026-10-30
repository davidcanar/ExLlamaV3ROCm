"""Stage-2 NVMe overflow for whole inactive sessions (EXL3_TP_DISK_CACHE, multi-session only).

A cold tier next to tp_sessions' shared pool: when a new request needs a slot or pages, the
least-recently-used inactive session is SPILLED to a per-rank local image on each box (rank
0 on Box1, rank 1 on the rank-1 box) instead of being evicted -- its pool pages and host
stashes come back on its next turn, skipping the re-prefill. Scope: spill/restore during
the server's lifetime only (no restart persistence); S=1 (EXL3_TP_SESSIONS=1) never touches
this module's wire ops or files.

Design:
- rank 0 owns every decision and drives the dance over two control frames at settled
  request boundaries only (never mid-request, never inside a collective):
      {"op": "spill", "session": sid, "image": img, "keys": [k...]} ->
          both ranks stream their own image, rank 1 answers
          {"op": "spill_ok", "session": sid, "bytes": B} or {"op": "spill_error", ...}
          keeping its state (rank 0 falls back to eviction); rank 0 then frees the
          session's pages/host stashes and files it under the cold registry
      {"op": "load", "session": sid, "image": img, "pages": [new physical pages],
       "keys": [...], "positions": [...]} -> both ranks stream their image back into the
          NEW pages (images are logical-page indexed) and rebuild their stashes; rank 1
          answers load_ok, or load_error and BOTH sides discard the image and the request
          falls back to a fresh session (full prefill, never wrong output)
- images are STREAMED both ways: the writer asks the backend for one page blob at a time
  and writes it before requesting the next, so peak host RAM is one page blob (~0.8 MiB on
  the 512K pool) plus the pickled stashes, never the whole session; the byte cap is
  enforced while writing (an over-cap write aborts, unlinks its tmp file and raises)
- every image is one file, written atomically (tmp + fsync + rename, 0o600), format 2:
  MAGIC | [u32 len][page blob] * n | stash pickle | trailer JSON | [u32 trailer len] (the
  trailing length locates the trailer, whose SHA-256s and sizes are only knowable after
  the stream); a corrupt, missing, truncated or incompatible image is a clean load error,
  never a silent restore. Stale images and leftover tmp files of THIS rank are swept at
  startup (images are one-shot).
- the page payloads are opaque blobs the ENGINE/backend produce per logical page (the real
  backend serializes that page's cache-layer tensor slices through cache/recurrent.py's
  HostPool with chunked device->host copies; the CPU stand-ins pickle their row dicts), so
  this module needs no torch and the whole wire+file protocol is CPU testable
- the tier is per-rank local: each box enforces EXL3_TP_DISK_CACHE_GIB against its own
  usage; rank 1 reports its byte count in spill_ok so rank 0 can surface both
- THREAT MODEL: the stash region is a pickle. Checksums detect corruption, but pickles are
  not a sandbox -- the tier trusts EXL3_TP_DISK_CACHE_DIR (0o700, operator-controlled
  local NVMe). Never point it at storage an untrusted party can write to.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Callable

from tp_engine import env_float, env_truthy

FORMAT = 2                     # image format version (2: streamed writes, trailer header;
                              # 1-image readers refuse 2 and vice versa -- one-shot images
                              # live only for the server's lifetime, so the bump is safe)
MAGIC = b"EXL3DSK" + bytes([FORMAT])
CHUNK = 8 << 20                # bounded staging: blob copies run through 8 MiB buffers


class DiskError(RuntimeError):
    """A spill/load could not complete; callers fall back to eviction or re-prefill."""


class DiskCorrupt(DiskError):
    """The image exists but fails validation (checksum, format, fingerprint, size)."""


def default_dir() -> str:
    """$EXL3_TP_DISK_CACHE_DIR or <repo>/diskcache where <repo> is this folder's grandparent
    (the checkout that contains tools/ds41)."""
    root = os.environ.get("EXL3_TP_DISK_CACHE_DIR")
    if root:
        return root
    return str(Path(__file__).resolve().parent.parent.parent / "diskcache")


class _ShaWriter:
    """Write-through SHA-256 for streaming a pickle straight into the image (no full
    stash-materializing bytes copy)."""

    def __init__(self, fp):
        self.fp, self.sha = fp, hashlib.sha256()
        self.written = 0

    def write(self, data) -> int:
        self.sha.update(data)
        self.written += len(data)
        return self.fp.write(data)

    def flush(self):
        self.fp.flush()


class DiskTier:
    """One rank's local NVMe image store: one file per image id, streamed writes, atomic
    renames, a byte cap enforced while writing, stale-file sweep at startup."""

    def __init__(self, *, root: str | None = None, rank: int, cap_gib: float | None = None,
                 enabled: bool = True, fingerprint: dict | None = None,
                 log: Callable[[str], None] | None = None):
        self.rank = int(rank)
        self.enabled = bool(enabled)
        self.root = Path(root if root is not None else default_dir())
        self.cap_bytes = int((env_float("EXL3_TP_DISK_CACHE_GIB", 32.0) if cap_gib is None
                              else float(cap_gib)) * (1 << 30))
        self.fingerprint = dict(fingerprint or {})
        self.counters = {"spills": 0, "spill_bytes": 0, "spill_errors": 0, "loads": 0,
                         "load_bytes": 0, "load_errors": 0, "corrupt": 0, "dropped": 0,
                         "startup_sweep": 0}
        self._sizes: dict[int, int] = {}
        self.log = log or (lambda line: None)
        if self.enabled:
            existed = self.root.exists()
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if not existed:
                os.chmod(self.root, 0o700)      # restrictive only for dirs we create
            self._sweep_stale()
            self.log(f"tp: disk tier on (rank {self.rank}): {self.root} "
                     f"(cap {self.cap_bytes / (1 << 30):.0f} GiB, format {FORMAT})")

    def _sweep_stale(self) -> None:
        """Images are one-shot (server lifetime only), so anything left from a previous
        run is dead: remove this rank's stale images and tmp files. Never touches another
        rank's files (tests put both ranks in one directory)."""
        for pat in (f"rank{self.rank}-*.img", f".rank{self.rank}-*.tmp"):
            for p in self.root.glob(pat):
                try:
                    p.unlink()
                    self.counters["startup_sweep"] += 1
                except OSError as exc:                                      # noqa: BLE001 (best effort)
                    self.log(f"tp: disk tier: stale {p} not removed: {exc}")

    # -- bookkeeping ------------------------------------------------------------------
    def path(self, image_id: int) -> Path:
        return self.root / f"rank{self.rank}-i{int(image_id)}.img"

    def used_bytes(self) -> int:
        return sum(self._sizes.values())

    def has(self, image_id: int) -> bool:
        return int(image_id) in self._sizes and self.path(image_id).exists()

    def size_of(self, image_id: int) -> int:
        return self._sizes.get(int(image_id), 0)

    def drop(self, image_id: int) -> None:
        """Forget and unlink an image (cold eviction, load, discard)."""
        image_id = int(image_id)
        self._sizes.pop(image_id, None)
        p = self.path(image_id)
        try:
            p.unlink()
            self.counters["dropped"] += 1
        except FileNotFoundError:
            pass
        except OSError as exc:                                          # noqa: BLE001  (best effort)
            self.log(f"tp: disk tier: unlink {p} failed: {exc}")

    # -- write ------------------------------------------------------------------------
    def write_image(self, image_id: int, *, meta: dict, n_pages: int,
                    page_blob_fn: Callable[[int], bytes],
                    stash_payload: Any) -> int:
        """Stream one session image atomically. `page_blob_fn(i)` is called once per
        logical page, IN ORDER, and each blob is written (and accounted against the cap)
        before the next is requested -- peak host RAM is one blob plus the pickled
        stashes, never the whole session. The stash payload itself is pickled straight
        into the file through a hashing writer. Returns the file size; raises DiskError
        on anything wrong (caller falls back to eviction; the tmp file never survives)."""
        if not self.enabled:
            raise DiskError("disk tier disabled")
        image_id = int(image_id)
        if image_id in self._sizes:
            raise DiskError(f"image {image_id} already exists (id reuse bug?)")
        budget = self.cap_bytes - self.used_bytes()
        tmp = self.root / f".rank{self.rank}-i{image_id}.{os.getpid()}.tmp"
        try:
            # unbuffered: every blob hits the file when written, so the cap accounts for
            # bytes actually on disk and the streaming contract is observable
            with open(tmp, "wb", buffering=0, opener=lambda p, f: os.open(p, f, 0o600)) as raw:
                fp = raw
                fp.write(MAGIC)
                data_sha = hashlib.sha256()
                data_bytes = 0
                for i in range(int(n_pages)):
                    blob = page_blob_fn(i)          # one page at a time, streamed out
                    if 4 + len(blob) > budget:
                        raise DiskError(
                            f"spill into image {image_id} would exceed the "
                            f"{self.cap_bytes}-byte cap (EXL3_TP_DISK_CACHE_GIB) at page "
                            f"{i + 1}/{n_pages}")
                    head = len(blob).to_bytes(4, "little")
                    fp.write(head)
                    for a in range(0, len(blob), CHUNK):
                        fp.write(blob[a: a + CHUNK])
                    data_sha.update(head + blob)
                    data_bytes += len(blob)
                    budget -= 4 + len(blob)
                stash_at = fp.tell()
                sha_fp = _ShaWriter(fp)
                pickle.dump(stash_payload, sha_fp)  # streamed, hashed while written
                stash_bytes = sha_fp.written
                if stash_bytes + 4096 > budget:
                    raise DiskError(f"spill into image {image_id} would exceed the "
                                    f"{self.cap_bytes}-byte cap with its stashes")
                trailer = dict(meta)
                trailer.update({"format": FORMAT, "image": image_id, "rank": self.rank,
                                "fingerprint": self.fingerprint, "created": time.time(),
                                "pages": int(n_pages), "data_bytes": data_bytes,
                                "data_sha256": data_sha.hexdigest(),
                                "stash_bytes": stash_bytes,
                                "stash_sha256": sha_fp.sha.hexdigest()})
                blob_t = json.dumps(trailer).encode("utf-8")
                if len(blob_t) > 1 << 22:
                    raise DiskError(f"image {image_id} metadata too large ({len(blob_t)} bytes)")
                fp.write(blob_t)
                fp.write(len(blob_t).to_bytes(4, "little"))   # the length is the LAST 4
                raw.flush()
                os.fsync(raw.fileno())
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path(image_id))
            try:                                            # persist the rename
                dfd = os.open(self.root, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
        except DiskError:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        except Exception as exc:                                        # noqa: BLE001  (any IO failure)
            try:
                tmp.unlink()
            except OSError:
                pass
            raise DiskError(f"spill into image {image_id} failed: {type(exc).__name__}: {exc}") from exc
        size = self.path(image_id).stat().st_size
        self._sizes[image_id] = size
        self.counters["spills"] += 1
        self.counters["spill_bytes"] += size
        return size

    # -- read -------------------------------------------------------------------------
    def read_image(self, image_id: int, *, blobs_into: Callable[[int, bytes], None],
                   stash_into: Callable[[Any], None], expect: dict | None = None) -> dict:
        """Validate and stream one image back: `blobs_into(logical_page, blob)` receives
        each page's payload one at a time as it is read (page rows may land in the fresh,
        not-yet-live pages before the whole-region checksum verdict -- they are dead rows
        unless the load fully succeeds and the pages are only then claimed), and
        `stash_into(payload)` the stashes -- strictly after every checksum passed, so the
        pickle is never handed over unverified. Returns the trailer. Raises
        DiskCorrupt/DiskError on anything wrong; the caller discards the image and falls
        back."""
        image_id = int(image_id)
        p = self.path(image_id)
        if not p.exists():
            raise DiskCorrupt(f"image {image_id} is missing ({p})")
        try:
            size = p.stat().st_size
            with open(p, "rb") as raw:
                if raw.read(len(MAGIC)) != MAGIC:
                    raise DiskCorrupt(f"image {image_id} has a bad magic/format")
                # trailer: [u32 len][json] at the very end, sized from the tail
                raw.seek(-4, os.SEEK_END)
                tlen = int.from_bytes(raw.read(4), "little")
                trailer_at = size - 4 - tlen
                if not 0 < tlen <= 1 << 22 or trailer_at <= len(MAGIC):
                    raise DiskCorrupt(f"image {image_id} has a nonsense trailer size")
                raw.seek(trailer_at)
                try:
                    head = json.loads(raw.read(tlen).decode("utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise DiskCorrupt(f"image {image_id} has an unreadable trailer: {exc}") from exc
                if head.get("format") != FORMAT or head.get("image") != image_id:
                    raise DiskCorrupt(f"image {image_id} is format {head.get('format')!r}, "
                                      f"id {head.get('image')!r}")
                for key, want in (expect or {}).items():
                    if head.get("fingerprint", {}).get(key) != want:
                        raise DiskCorrupt(
                            f"image {image_id} is incompatible: {key} "
                            f"{head.get('fingerprint', {}).get(key)!r} != {want!r}")
                raw.seek(len(MAGIC))
                n_pages = int(head.get("pages", 0))
                data_sha = hashlib.sha256()
                for i in range(n_pages):
                    n = int.from_bytes(raw.read(4), "little")
                    if not 0 <= n <= 1 << 30:
                        raise DiskCorrupt(f"image {image_id} has a nonsense page size")
                    blob = raw.read(n)
                    if len(blob) != n:
                        raise DiskCorrupt(f"image {image_id} truncated in page {i}")
                    data_sha.update(n.to_bytes(4, "little") + blob)
                    blobs_into(i, blob)
                if data_sha.hexdigest() != head.get("data_sha256"):
                    raise DiskCorrupt(f"image {image_id} fails its data checksum")
                stash_end = size - 4 - tlen
                want_bytes = stash_end - raw.tell()
                pickled = raw.read(want_bytes)
                if len(pickled) != want_bytes or \
                        hashlib.sha256(pickled).hexdigest() != head.get("stash_sha256"):
                    raise DiskCorrupt(f"image {image_id} fails its stash checksum")
                stash_into(pickle.loads(pickled))          # noqa: S301  (our own file: checksummed, 0o700 dir)
                if raw.tell() != stash_end:
                    raise DiskCorrupt(f"image {image_id} has trailing garbage")
        except DiskCorrupt:
            self.counters["corrupt"] += 1
            raise
        except Exception as exc:                                        # noqa: BLE001  (any IO failure)
            raise DiskError(f"load of image {image_id} failed: {type(exc).__name__}: {exc}") from exc
        self.counters["loads"] += 1
        self.counters["load_bytes"] += size
        return head

    def status(self) -> dict:
        return {"enabled": self.enabled, "dir": str(self.root), "rank": self.rank,
                "cap_gib": round(self.cap_bytes / (1 << 30), 2),
                "used_gib": round(self.used_bytes() / (1 << 30), 3),
                "images": len(self._sizes), "counters": dict(self.counters)}


def tier_from_env(*, rank: int, fingerprint: dict, log=None, default_on: bool,
                  root: str | None = None, cap_gib: float | None = None,
                  enabled: bool | None = None) -> DiskTier:
    """Build this rank's tier. Enabled by default when multi-session serving is on
    (EXL3_TP_DISK_CACHE=1); EXL3_TP_DISK_CACHE=0 disables it (pure in-memory pool). `root`,
    `cap_gib` and `enabled` override the env (the CPU tests pin per-rank dirs and the
    eviction-only controls without env races)."""
    use = env_truthy("EXL3_TP_DISK_CACHE", bool(default_on)) if enabled is None else bool(enabled)
    return DiskTier(rank=rank, enabled=use, fingerprint=fingerprint, log=log,
                    root=root, cap_gib=cap_gib)
