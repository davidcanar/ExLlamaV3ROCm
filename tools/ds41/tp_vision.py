#!/usr/bin/env python3
"""CPU-side image handling for the two-node DS4.1 TP server's vision support (rank 0 only).

The tower + aligner itself is the exllamav3 component model (``component = "vision"``,
``exllamav3/architecture/deepseek_v41_vision.py``) loaded by ``TPModelBackend`` on rank 0.
This module is everything around it that must be testable without a GPU:

- safe image loading (adapted from Kyojin ``exllamav3/dsv41/vision.py`` / ``vision_spec.py``,
  MIT, see LICENSE.kyojin): request records -> capped bytes -> a decoded RGB PIL image.
  ``data:`` URLs, bare base64 and http(s) URLs; hard caps on fetched bytes and decoded
  pixels (decompression bombs); every failure is ``ImageError`` (a ``ValueError`` -- the
  vendored server answers 400 with its message);
- the V4.1 resize/geometry plan mirrored from
  ``DeepseekV41VisionModel.preprocess``/``plan_resize`` so ``prompt_ids`` can expand image
  placeholders to their exact token spans BEFORE the tower runs (the tower's own result is
  asserted to match at generate time -- a silent drift is impossible);
- per-image unique index ranges and text aliases for the ``indexed_embeddings`` mechanism
  (``exllamav3/tokenizer/mm_embedding.py``: dynamic token ids >= 1000000000 whose rows ride
  in ``params["indexed_embeddings"]``, the native multimodal path of this tree -- the MoE
  ``bias_vl`` routing, the Engram n-gram block and the PLE mapping all key off those ids);
- the rank 0 -> rank 1 wire codec for the tower's output rows (metadata JSON frame + raw
  binary frames on the control channel; sha256-validated).

Nothing here imports exllamav3, torch or PIL at module level: the control-protocol module
``tp_engine`` imports this file unconditionally (CPU tests, rank 1 included).
"""
from __future__ import annotations

import base64
import hashlib
import math
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any
from urllib.request import urlopen

# Kyojin's caps (vision_spec.py): 32 MiB fetched, 100 Mpx decoded. Both overridable.
DEFAULT_MAX_IMAGE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_SOURCE_PIXELS = 100_000_000
DEFAULT_FETCH_TIMEOUT = 30.0
DEFAULT_MAX_IMAGES = 8

# exllamav3.tokenizer.mm_embedding.FIRST_MM_EMBEDDING_INDEX (dynamic multimodal id base).
# Imported lazily by tp_engine where the library is available; asserted equal there.
FIRST_MM_EMBEDDING_INDEX = 1000000000

_DATA_URL = re.compile(r"^data:([^;,]*)(;[^,]*)?,", re.IGNORECASE)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


class ImageError(ValueError):
    """The image cannot be loaded or used (the vendored server answers 400)."""


# ---- safe image loading (Kyojin rules) -----------------------------------------------------------------------


def load_image_bytes(record: dict, *, allow_http: bool | None = None,
                     max_bytes: int | None = None, timeout: float | None = None) -> tuple[bytes, str]:
    """A rendered image record -> (bytes, source description).

    Records come from the vendored chat layer (ds41_chat._image_record): ``{"url": ...}`` for
    an http(s) or data: URL, ``{"data": ...}`` for base64 / a data: URL payload. Mirrors
    Kyojin ``vision.load_image_bytes``: no local file paths, no redirects off http(s), the
    response body capped mid-read, base64 tolerated with or without padding (a payload that
    does not decode as an image is rejected at PIL time with a clear message).
    """
    if allow_http is None:
        allow_http = env_truthy("EXL3_TP_IMAGE_HTTP", True)
    if max_bytes is None:
        max_bytes = env_int("EXL3_TP_IMAGE_MAX_BYTES", DEFAULT_MAX_IMAGE_BYTES)
    if timeout is None:
        timeout = env_float("EXL3_TP_IMAGE_FETCH_TIMEOUT", DEFAULT_FETCH_TIMEOUT)

    data = record.get("data") if isinstance(record, dict) else None
    try:
        if isinstance(data, (bytes, bytearray)):
            return bytes(data), "inline bytes"
        if isinstance(data, str) and data:
            if data.startswith("data:"):
                return _decode_data_url(data), "data URL"
            return base64.b64decode(data, validate=False), "base64"
        url = record.get("url") if isinstance(record, dict) else None
        if not isinstance(url, str) or not url:
            raise ImageError(f"cannot load an image from record keys {sorted(record or {})}")
        if url.startswith("data:"):
            return _decode_data_url(url), "data URL"
        if url.startswith(("http://", "https://")):
            if not allow_http:
                raise ImageError("http(s) image URLs are disabled on this server (EXL3_TP_IMAGE_HTTP=0)")
            # Operator-controlled lab server, as Kyojin: no SSRF filtering. The read is capped
            # mid-stream so a huge (or endless) body cannot exhaust the host.
            with urlopen(url, timeout=timeout) as resp:                # noqa: S310
                raw = resp.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise ImageError(f"image larger than {max_bytes >> 20} MiB (EXL3_TP_IMAGE_MAX_BYTES)")
            return raw, url[:120]
        raise ImageError("image URL must be a data: URL or an http(s) URL")
    except ImageError:
        raise
    except Exception as exc:                                           # noqa: BLE001  (one clear message)
        raise ImageError(f"cannot load image: {exc}") from exc


def _decode_data_url(url: str) -> bytes:
    header, _, payload = url.partition(",")
    match = _DATA_URL.match(url)
    if match is None or ";base64" not in header.lower():
        raise ImageError(f"unsupported data URL encoding: {header[:40]!r} (base64 only)")
    try:
        return base64.b64decode(payload, validate=False)
    except Exception as exc:                                           # noqa: BLE001
        raise ImageError(f"invalid base64 image data: {exc}") from exc


def decode_image(raw: bytes, *, max_pixels: int | None = None):
    """Image bytes -> a decoded RGB PIL image (EXIF-oriented), under the decoded-pixel cap.

    The cap plus PIL's own MAX_IMAGE_PIXELS guard is the decompression-bomb defense: a tiny
    crafted file that expands to gigabytes is rejected before the resize.
    """
    if max_pixels is None:
        max_pixels = env_int("EXL3_TP_IMAGE_MAX_PIXELS", DEFAULT_MAX_SOURCE_PIXELS)
    import io
    from PIL import Image, ImageOps
    try:
        with Image.open(io.BytesIO(raw)) as src:
            if src.width * src.height > max_pixels:
                raise ImageError(f"image of {src.width}x{src.height} exceeds the "
                                 f"{max_pixels} decoded-pixel cap (EXL3_TP_IMAGE_MAX_PIXELS)")
            image = ImageOps.exif_transpose(src).convert("RGB")       # strips EXIF orientation
    except ImageError:
        raise
    except Exception as exc:                                           # noqa: BLE001
        raise ImageError(f"cannot decode image: {exc}") from exc
    if image.width < 1 or image.height < 1:
        raise ImageError("image has no pixels")
    return image


# ---- V4.1 geometry (mirrors DeepseekV41VisionModel.preprocess / plan_resize) ---------------------------------


def plan_resize_v41(height: int, width: int, best_height: int, best_width: int, v) -> tuple[int, int, int, int]:
    """The V4.1 budget fit, line for line as ``DeepseekV41VisionModel.plan_resize`` (kept in
    sync by test: the tower's own plan is asserted equal to this one at generate time)."""
    p, cell, budget = v.patch_size, v.patch_size * v.downsample_ratio, v.max_n_token - 2

    def grid(bh, bw):
        return math.ceil(bh // p / v.downsample_ratio), math.ceil(bw // p / v.downsample_ratio)

    n_llm_h, n_llm_w = grid(best_height, best_width)
    if n_llm_h * (n_llm_w + 1) > budget:
        r = height / width
        max_w = math.sqrt(budget / r + 0.25) - 0.5
        max_h = max_w * r
        if max_w < 1.0:
            best_height, best_width = budget // 2 * cell, cell
        elif max_h < 1.0:
            best_height, best_width = cell, (budget - 1) * cell
        else:
            beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
            best_height, best_width = math.floor(height * beta / p) * p, math.floor(width * beta / p) * p
        n_llm_h, n_llm_w = grid(best_height, best_width)
    if not (n_llm_h > 0 and n_llm_w > 0 and n_llm_h * (n_llm_w + 1) <= budget):
        raise ImageError("image does not fit the vision token budget "
                         f"(grid {n_llm_h}x{n_llm_w}, budget {budget})")
    return n_llm_h, n_llm_w, best_height, best_width


def plan_grid(width: int, height: int, v) -> tuple[int, int]:
    """(width, height) of a decoded image -> the (n_llm_h, n_llm_w) token grid the tower will
    produce. The full head of ``DeepseekV4VisionModel.preprocess`` with the V4.1 override:
    aspect clamp (none in V4.1: max_wh_ratio is null), min_pixels upscale, round up to the
    patch grid, then the ``plan_resize_v41`` budget fit."""
    p = v.patch_size
    if v.max_wh_ratio is not None and width > height * v.max_wh_ratio:
        width = height * v.max_wh_ratio
    if 0 < width * height < v.min_pixels:
        ratio = (v.min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, _, _ = plan_resize_v41(height, width, best_height, best_width, v)
    return n_llm_h, n_llm_w


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """Rows of one V4.1 image block: [start] + (n_llm_w image rows + newline) * n_llm_h + [end]."""
    return n_llm_h * (n_llm_w + 1) + 2


# ---- per-image identity: index range, alias, span -------------------------------------------------------------


class MMIndexAllocator:
    """Hands out contiguous dynamic-id ranges >= FIRST_MM_EMBEDDING_INDEX, never reusing one.

    The library's own global allocator does the same for aliases typed into ``tokenizer.encode``;
    the engine drives expansion itself (occurrence order of ``<｜deepseek_image｜>``), so it keeps
    its own monotonic counter. Rank 1 never allocates: its ranges arrive over the wire, and the
    never-reuse rule keeps a stale rank-1 copy of a previous request's rows from ever aliasing a
    new image's ids."""

    def __init__(self, first: int = FIRST_MM_EMBEDDING_INDEX):
        self.next = int(first)
        self._lock = threading.Lock()

    def allocate(self, n: int) -> int:
        with self._lock:
            idx = self.next
            self.next += int(n)
            return idx


@dataclass
class PreparedImage:
    """One request image between prepare_images and generate (the same object travels through
    ``prompt_ids`` and ``generate``; ids and spans are fixed at prepare time)."""
    pil: Any                                    # decoded RGB PIL image (kept for the tower)
    n_llm_h: int
    n_llm_w: int
    first_index: int                            # dynamic-id base (>= 1e9), unique process-wide
    alias: str                                  # unique text alias (logs / metadata only)
    key: str = ""                               # sha256 of the source bytes + grid (identity)
    source: str = ""                            # where the bytes came from (logs)
    size: tuple = (0, 0)                        # original (width, height)
    rows: Any = None                            # tower + aligner output [n_tokens, hidden]; generate fills it

    @property
    def n_tokens(self) -> int:
        return num_image_tokens(self.n_llm_h, self.n_llm_w)

    @property
    def token_ids(self) -> list[int]:
        return list(range(self.first_index, self.first_index + self.n_tokens))

    @property
    def mm_length(self) -> int:
        return self.n_tokens


def prepare_one(record: dict, v, alloc: MMIndexAllocator, *, max_pixels: int | None = None) -> PreparedImage:
    """record -> PreparedImage (CPU: fetch, decode, geometry). The tower runs later, on the
    model thread, from ``pil``."""
    raw, source = load_image_bytes(record)
    image = decode_image(raw, max_pixels=max_pixels)
    n_llm_h, n_llm_w = plan_grid(image.width, image.height, v)
    first = alloc.allocate(num_image_tokens(n_llm_h, n_llm_w))
    h = hashlib.sha256()
    h.update(str((n_llm_h, n_llm_w)).encode())
    h.update(raw)
    return PreparedImage(pil=image, n_llm_h=n_llm_h, n_llm_w=n_llm_w, first_index=first,
                         alias=f"<$EMB_{first}$>", key=h.hexdigest()[:16], source=source,
                         size=(image.width, image.height))


def expand_prompt_ids(ids: list[int], images: list[PreparedImage], image_token_id: int) -> list[int]:
    """Replace each ``image_token_id`` occurrence with that image's dynamic-id span, in order.

    The ids must hold exactly one placeholder per image (the rendered prompt carries one
    ``<｜deepseek_image｜>`` per image part); anything else is a request-building bug or a
    hand-crafted prompt, refused here rather than mis-spliced."""
    if not images:
        return list(ids)
    out: list[int] = []
    it = iter(images)
    spent = 0
    for t in ids:
        if t != image_token_id:
            out.append(int(t))
            continue
        img = next(it, None)
        if img is None:
            raise ImageError(f"the prompt has more image placeholders than the {len(images)} image(s) sent")
        out.extend(img.token_ids)
        spent += 1
    if spent != len(images):
        raise ImageError(f"the prompt has {spent} image placeholder(s) but {len(images)} image(s) were sent")
    return out


# ---- rank 0 -> rank 1 wire codec (control-channel blob frames) -----------------------------------------------


def pack_rows(tensor) -> tuple[str, bytes]:
    """2-D rows tensor -> (dtype name, raw little-endian bytes). bf16 has no numpy dtype, so
    every supported dtype ships through its same-width integer view."""
    import torch
    t = tensor.detach().to("cpu").contiguous()
    name = str(t.dtype).replace("torch.", "")
    views = {"float16": torch.int16, "bfloat16": torch.int16, "float32": torch.int32}
    if t.dim() != 2 or name not in views:
        raise ValueError(f"unsupported embedding rows for shipment: {tuple(t.shape)} {t.dtype}")
    return name, t.view(views[name]).numpy().tobytes()


def unpack_rows(raw: bytes, dtype: str, rows: int, dim: int):
    """Inverse of pack_rows; returns a CPU tensor owning its memory."""
    import torch
    views = {"float16": (torch.int16, torch.float16), "bfloat16": (torch.int16, torch.bfloat16),
             "float32": (torch.int32, torch.float32)}
    if dtype not in views:
        raise ValueError(f"unsupported embedding dtype on the wire: {dtype!r}")
    iview, real = views[dtype]
    item = torch.empty((), dtype=real).element_size()
    if len(raw) != rows * dim * item:
        raise ValueError(f"embedding blob is {len(raw)} bytes, expected {rows}x{dim} {dtype} "
                         f"({rows * dim * item} bytes)")
    t = torch.frombuffer(bytearray(raw), dtype=iview).view(real).reshape(rows, dim)
    return t.clone()


def encode_imgmeta(images: list[PreparedImage]) -> tuple[dict, list[bytes]]:
    """The imgmeta control frame and the binary payloads that follow it, in order.

    ``rows`` must already be the tower's output on every image (generate fills it before
    shipping). Each entry carries the dynamic-id range it owns; rank 1 rebuilds its
    ``indexed_embeddings`` list from exactly these numbers, so the two ranks agree on which
    id maps to which row without shipping a single id again."""
    specs, blobs = [], []
    for img in images:
        if img.rows is None:
            raise ValueError(f"image {img.alias} has no tower rows to ship")
        dtype, raw = pack_rows(img.rows)
        rows, dim = int(img.rows.shape[0]), int(img.rows.shape[1])
        if rows != img.n_tokens:
            raise ValueError(f"tower produced {rows} rows for {img.alias} but the plan promised {img.n_tokens}")
        specs.append({"alias": img.alias, "first_index": int(img.first_index), "rows": rows,
                      "dim": dim, "dtype": dtype, "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest()})
        blobs.append(raw)
    return {"op": "imgmeta", "n": len(specs), "images": specs}, blobs


def decode_imgmeta(meta: dict, blobs: list[bytes], first_mm: int = FIRST_MM_EMBEDDING_INDEX) -> list[dict]:
    """Validate an imgmeta frame against its blobs (shapes, hashes, id ranges) and return one
    descriptor per image: {first_index, last_index, rows, dim, dtype, alias, tensor}. Raises
    ``ValueError`` on any mismatch -- rank 1 turns that into an error frame and both ranks
    restart, never a mid-collective fault."""
    if not isinstance(meta, dict) or meta.get("op") != "imgmeta":
        raise ValueError("not an imgmeta frame")
    specs = meta.get("images")
    if not isinstance(specs, list) or len(specs) != int(meta.get("n", -1)) or len(blobs) != len(specs):
        raise ValueError(f"imgmeta describes {meta.get('n')} image(s) but {len(blobs)} blob(s) arrived")
    out, last_end = [], first_mm
    for spec, raw in zip(specs, blobs):
        for k in ("first_index", "rows", "dim", "dtype", "bytes"):
            if k not in spec:
                raise ValueError(f"imgmeta image entry misses {k!r}")
        first, rows, dim = int(spec["first_index"]), int(spec["rows"]), int(spec["dim"])
        if first < first_mm:
            raise ValueError(f"imgmeta first_index {first} is below the dynamic-id base {first_mm}")
        if first < last_end:
            raise ValueError(f"imgmeta ranges overlap or regress at first_index {first} (previous end {last_end})")
        if rows < 2 or dim < 1 or int(spec["bytes"]) != len(raw):
            raise ValueError(f"imgmeta image entry has impossible shape {rows}x{dim} / {len(raw)} bytes")
        if hashlib.sha256(raw).hexdigest() != spec.get("sha256"):
            raise ValueError(f"imgmeta blob sha256 mismatch for {spec.get('alias')}")
        last_end = first + rows
        out.append({"alias": str(spec.get("alias") or ""), "first_index": first, "last_index": last_end,
                    "rows": rows, "dim": dim, "dtype": str(spec["dtype"]),
                    "tensor": unpack_rows(raw, str(spec["dtype"]), rows, dim)})
    return out
