"""Vision support of the two-node TP server, CPU only.

Covers the pieces of tp_vision.py / tp_engine.py that make image requests work across ranks
without any GPU, model weights or the real tower:

- safe image loading: data:/base64/http(s) records, byte + pixel caps, every failure a
  ValueError (the vendored server answers 400);
- the V4.1 geometry plan mirrors the tower's own math (checked against
  exllamav3.architecture.deepseek_v41_vision when the tree is importable);
- per-image unique dynamic-id ranges, alias expansion of `<｜deepseek_image｜>` prompts and
  expanded-length accounting;
- the rank 0 -> rank 1 shipment: JSON + binary frames on one channel, sha256-validated
  reconstruction, protocol version/handshake validation;
- indexed_embeddings on EVERY prefill chunk (both ranks, chunk offsets unchanged) and never
  on text requests;
- prefix-cache safety: image requests restore and save nothing, text requests keep the cache;
- text-path regressions at the frame level (the ops rank 1 sees are unchanged).

The engine-level pairs additionally exercise make_mm_embedding, so they need the exllamav3
tree importable (PYTHONPATH=src/exllamav3-dsv41f, as the launchers set it); they skip with a
clear reason otherwise.
"""
from __future__ import annotations

import asyncio
import base64
import importlib.util
import io
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds41_testkit as kit  # noqa: E402
import tp_engine  # noqa: E402
import tp_vision  # noqa: E402
import test_tp_control as ctrl  # noqa: E402  (loopback helpers of the main suite)
from ds41_engine import SPECIAL_TOKENS  # noqa: E402
from test_tp_chunked_prefill import FakeCore, make_backend  # noqa: E402

FIRST = tp_vision.FIRST_MM_EMBEDDING_INDEX
IMAGE_TOKEN = SPECIAL_TOKENS.index("<｜deepseek_image｜>") + 256          # ByteTokenizer id
HAS_EXL = importlib.util.find_spec("exllamav3") is not None
needs_exl = pytest.mark.skipif(not HAS_EXL, reason="exllamav3 tree not on PYTHONPATH")


class VisionTokenizer(ctrl.DetokByteTokenizer):
    """DetokByteTokenizer that knows the image placeholder id (the engine's resolver honors it)."""

    image_token_id = IMAGE_TOKEN


# The deployed model's vision config shape; small max_n_token keeps the test spans short
# (the math is under test, not the budget)
VCFG = SimpleNamespace(patch_size=14, downsample_ratio=3, max_n_token=64,
                       min_pixels=295936, max_wh_ratio=None)


def tiny_png(width: int = 64, height: int = 48, color=(200, 10, 10)) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def data_url(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


# ---- safe image loading ---------------------------------------------------------------------------------------


def test_data_url_and_bare_base64_records_decode():
    raw = tiny_png()
    for record in ({"url": data_url(raw)}, {"data": base64.b64encode(raw).decode("ascii")},
                   {"data": data_url(raw)}):
        got, source = tp_vision.load_image_bytes(record)
        assert got == raw
        assert source


def test_rejects_non_http_non_data_schemes_and_empty_records():
    with pytest.raises(tp_vision.ImageError, match="data: URL or an http"):
        tp_vision.load_image_bytes({"url": "file:///etc/passwd"})
    with pytest.raises(tp_vision.ImageError, match="data: URL or an http"):
        tp_vision.load_image_bytes({"url": "ftp://example.com/x.png"})
    with pytest.raises(tp_vision.ImageError, match="record keys"):
        tp_vision.load_image_bytes({})


def test_rejects_unsupported_data_url_encoding_and_bad_payloads():
    with pytest.raises(tp_vision.ImageError, match="unsupported data URL"):
        tp_vision.load_image_bytes({"url": "data:image/png," + "A" * 16})
    with pytest.raises(tp_vision.ImageError, match="cannot load|cannot decode|base64"):
        tp_vision.load_image_bytes({"data": "!!!not base64!!!"})
    with pytest.raises(tp_vision.ImageError, match="cannot decode"):
        tp_vision.decode_image(b"not an image at all")


def test_http_fetch_works_and_is_capped_mid_read(monkeypatch):
    raw = tiny_png()

    class Resp:
        def __init__(self, payload):
            self.payload = payload

        def read(self, n):
            return self.payload[:n]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    fetched = {}

    def fake_urlopen(url, timeout=None):
        fetched["url"] = url
        return Resp(raw)

    monkeypatch.setattr(tp_vision, "urlopen", fake_urlopen)
    got, _ = tp_vision.load_image_bytes({"url": "https://example.com/cat.png"})
    assert got == raw and fetched["url"] == "https://example.com/cat.png"

    huge = b"x" * (tp_vision.DEFAULT_MAX_IMAGE_BYTES + 1)
    monkeypatch.setattr(tp_vision, "urlopen", lambda url, timeout=None: Resp(huge))
    with pytest.raises(tp_vision.ImageError, match="larger than"):
        tp_vision.load_image_bytes({"url": "https://example.com/huge.png"})


def test_http_fetch_can_be_disabled(monkeypatch):
    monkeypatch.setenv("EXL3_TP_IMAGE_HTTP", "0")
    with pytest.raises(tp_vision.ImageError, match="disabled"):
        tp_vision.load_image_bytes({"url": "https://example.com/cat.png"})


def test_decoded_pixel_cap_blocks_decompression_bombs():
    with pytest.raises(tp_vision.ImageError, match="pixel cap"):
        tp_vision.decode_image(tiny_png(), max_pixels=4)          # 64x48 decoded > 4 px cap


def test_decode_image_returns_rgb_and_survives_exif_payloads():
    from PIL import Image
    image = tp_vision.decode_image(tiny_png())
    assert image.mode == "RGB" and image.size == (64, 48)
    src = Image.new("RGB", (3, 2), (7, 8, 9))                     # EXIF rotation 6 attached
    buf = io.BytesIO()
    src.save(buf, format="JPEG",
             exif=b"Exif\x00\x00\x12\x00\x0a\x00\x01\x00\x00\x00\x06\x00\x00\x00\x00\x00\x00")
    upright = tp_vision.decode_image(buf.getvalue())
    assert upright.size in ((3, 2), (2, 3))                       # decoded, oriented, RGB


# ---- geometry: the plan matches the tower's own math ----------------------------------------------------------


def test_plan_grid_matches_the_architecture_plan_resize():
    if not HAS_EXL:
        pytest.skip("exllamav3 tree not on PYTHONPATH")
    from exllamav3.architecture.deepseek_v41_vision import DeepseekV41VisionModel
    v = SimpleNamespace(patch_size=14, downsample_ratio=3, max_n_token=1024,
                        min_pixels=0, max_wh_ratio=None)     # isolate the resize math under test
    cfg = SimpleNamespace(config=SimpleNamespace(vision=v))
    import math
    for width, height in ((1, 1), (14, 14), (100, 30), (512, 512), (1920, 1080), (1080, 1920),
                          (295936, 1), (1, 295936), (10000, 40), (40, 10000), (65536, 65536)):
        p = v.patch_size
        bw, bh = math.ceil(width / p) * p, math.ceil(height / p) * p
        want = DeepseekV41VisionModel.plan_resize(cfg, bh, bw, bh, bw)   # (n_llm_h, n_llm_w, ..)
        got_h, got_w = tp_vision.plan_grid(width, height, v)
        assert (got_h, got_w) == (want[0], want[1]), (width, height, (got_h, got_w), want[:2])
        assert tp_vision.num_image_tokens(got_h, got_w) <= v.max_n_token


def test_num_image_tokens_is_the_v41_block_layout():
    # [start] + (n_llm_w image rows + newline) * n_llm_h + [end]
    assert tp_vision.num_image_tokens(1, 1) == 4
    assert tp_vision.num_image_tokens(3, 5) == 3 * 6 + 2
    assert tp_vision.num_image_tokens(7, 8) == 7 * 9 + 2


def test_plan_of_a_tiny_image_is_the_budget_fit():
    n_h, n_w = tp_vision.plan_grid(64, 48, VCFG)                  # min_pixels upscales, budget caps
    assert n_h >= 1 and n_w >= 1 and tp_vision.num_image_tokens(n_h, n_w) <= VCFG.max_n_token


# ---- identity: allocator, aliases, placeholder expansion -------------------------------------------------------


def test_allocator_is_monotonic_and_never_reuses():
    alloc = tp_vision.MMIndexAllocator()
    a = alloc.allocate(4)
    b = alloc.allocate(120)
    assert a == FIRST and b == FIRST + 4
    threads, seen, lock = [], set(), threading.Lock()

    def grab():
        for _ in range(50):
            with lock:
                seen.add(alloc.allocate(1))

    for _ in range(4):
        t = threading.Thread(target=grab)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    assert len(seen) == 200                                        # every range unique under threads


def test_prepare_one_assigns_unique_alias_and_span():
    alloc = tp_vision.MMIndexAllocator()
    one = tp_vision.prepare_one({"data": base64.b64encode(tiny_png()).decode()}, VCFG, alloc)
    two = tp_vision.prepare_one({"data": base64.b64encode(tiny_png(32, 32)).decode()}, VCFG, alloc)
    assert one.first_index == FIRST and two.first_index == FIRST + one.n_tokens
    assert one.alias != two.alias and one.key != two.key
    assert one.token_ids == list(range(FIRST, FIRST + one.n_tokens))
    assert one.n_tokens == tp_vision.num_image_tokens(one.n_llm_h, one.n_llm_w)


def test_expand_prompt_ids_splices_each_occurrence_in_order():
    alloc = tp_vision.MMIndexAllocator()
    imgs = [tp_vision.prepare_one({"data": base64.b64encode(tiny_png(w, 32)).decode()}, VCFG, alloc)
            for w in (64, 32)]
    ids = [10, 11, IMAGE_TOKEN, 12, IMAGE_TOKEN, 13]
    out = tp_vision.expand_prompt_ids(ids, imgs, IMAGE_TOKEN)
    assert out[:2] == [10, 11] and out[2:2 + imgs[0].n_tokens] == imgs[0].token_ids
    rest = out[2 + imgs[0].n_tokens:]
    assert rest[0] == 12
    assert rest[1:1 + imgs[1].n_tokens] == imgs[1].token_ids
    assert rest[-1] == 13
    assert len(out) == len(ids) - 2 + imgs[0].n_tokens + imgs[1].n_tokens


def test_expand_prompt_ids_refuses_count_mismatches():
    alloc = tp_vision.MMIndexAllocator()
    img = [tp_vision.prepare_one({"data": base64.b64encode(tiny_png()).decode()}, VCFG, alloc)]
    with pytest.raises(tp_vision.ImageError, match="more image placeholders"):
        tp_vision.expand_prompt_ids([IMAGE_TOKEN, IMAGE_TOKEN], img, IMAGE_TOKEN)
    with pytest.raises(tp_vision.ImageError, match="placeholder"):
        tp_vision.expand_prompt_ids([IMAGE_TOKEN], img * 2, IMAGE_TOKEN)


# ---- the wire codec --------------------------------------------------------------------------------------------


def _prepared_with_rows(n_tokens: int = 6, dim: int = 8, first: int = FIRST, dtype=torch.float16):
    return SimpleNamespace(alias=f"<$EMB_{first}$>", first_index=first, n_tokens=n_tokens,
                           rows=torch.arange(n_tokens * dim, dtype=dtype).reshape(n_tokens, dim))


def test_pack_unpack_rows_round_trip_every_dtype():
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        rows = torch.randn(5, 7).to(dtype)
        name, raw = tp_vision.pack_rows(rows)
        back = tp_vision.unpack_rows(raw, name, 5, 7)
        assert back.dtype == dtype and back.shape == (5, 7)
        assert torch.equal(back, rows)                             # bit-exact, no float text anywhere


def test_imgmeta_round_trip_reconstructs_ranges_and_rows():
    preps = [_prepared_with_rows(6, 8, FIRST), _prepared_with_rows(10, 8, FIRST + 6, dtype=torch.bfloat16)]
    meta, blobs = tp_vision.encode_imgmeta(preps)
    described = tp_vision.decode_imgmeta(meta, blobs)
    assert [d["first_index"] for d in described] == [FIRST, FIRST + 6]
    assert [d["rows"] for d in described] == [6, 10]
    assert torch.equal(described[0]["tensor"], preps[0].rows)
    assert torch.equal(described[1]["tensor"], preps[1].rows)     # bf16 rides an int16 view


def test_imgmeta_refuses_corruption_and_bad_ranges():
    preps = [_prepared_with_rows(6, 8)]
    meta, blobs = tp_vision.encode_imgmeta(preps)
    corrupt = bytearray(blobs[0])
    corrupt[0] ^= 0xFF
    with pytest.raises(ValueError, match="sha256"):
        tp_vision.decode_imgmeta(meta, [bytes(corrupt)])
    with pytest.raises(ValueError, match="bytes"):
        tp_vision.decode_imgmeta(meta, [blobs[0][:10]])           # truncated blob
    bad = json.loads(json.dumps(meta))
    bad["images"][0]["first_index"] = FIRST - 1
    with pytest.raises(ValueError, match="dynamic-id base"):
        tp_vision.decode_imgmeta(bad, blobs)
    overlap = json.loads(json.dumps({**meta, "images": meta["images"] + [dict(meta["images"][0])]}))
    overlap["n"] = 2
    with pytest.raises(ValueError, match="overlap|regress"):
        tp_vision.decode_imgmeta(overlap, blobs + blobs)
    with pytest.raises(ValueError, match="unsupported"):
        tp_vision.unpack_rows(blobs[0], "float64", 6, 8)


def test_encode_imgmeta_refuses_rows_that_deny_the_plan():
    lying = SimpleNamespace(alias="<$EMB_x$>", first_index=FIRST, n_tokens=7, rows=torch.zeros(6, 8))
    with pytest.raises(ValueError, match="promised"):
        tp_vision.encode_imgmeta([lying])


# ---- binary frames on the control channel -----------------------------------------------------------------------


def test_json_and_blob_frames_share_one_channel_without_desync():
    port = ctrl.free_port()
    got = {}

    def server_side():
        chan = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
        got["hello"] = chan.recv()
        got["blob1"] = chan.recv_blob()
        got["meta"] = chan.recv()
        got["blob2"] = chan.recv_blob()
        chan.send({"op": "ok", "saw": len(got["blob1"]) + len(got["blob2"])})
        chan.close()

    t = threading.Thread(target=server_side, daemon=True)
    t.start()
    cli = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
    payload1 = bytes(range(256)) * 41                              # > one TCP segment's worth
    payload2 = b"\x7b\x00\x01json-looking bytes inside a blob"
    cli.send({"op": "hello", "x": 1})
    cli.send_blob(payload1)
    reply = cli.request_with_blobs({"op": "meta"}, [payload2], timeout=10.0)
    t.join(10.0)
    assert got["hello"] == {"op": "hello", "x": 1}
    assert got["blob1"] == payload1 and got["blob2"] == payload2
    assert reply["op"] == "ok" and reply["saw"] == len(payload1) + len(payload2)
    cli.close()


def test_recv_refuses_a_blob_where_json_is_expected_and_vice_versa():
    port = ctrl.free_port()
    errs = []

    def server_side():
        try:
            chan = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
            try:
                chan.recv()                                        # a blob arrives: must refuse
                errs.append("recv accepted a blob as JSON")
            except tp_engine.ProtocolError:
                pass
            chan.send({"op": "bye"})
            chan.close()
        except BaseException as exc:                               # noqa: BLE001
            errs.append(exc)

    t = threading.Thread(target=server_side, daemon=True)
    t.start()
    cli = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
    cli.send_blob(b"not json")
    assert cli.recv() == {"op": "bye"}
    t.join(10.0)
    cli.close()

    port2 = ctrl.free_port()
    errs2 = []

    def server_side2():
        try:
            chan = tp_engine.ControlChannel.listen("127.0.0.1", port2, accept_timeout=20.0)
            try:
                chan.recv_blob()                                   # JSON arrives: must refuse
                errs2.append("recv_blob accepted JSON")
            except tp_engine.ProtocolError:
                pass
            chan.close()
        except BaseException as exc:                               # noqa: BLE001
            errs2.append(exc)

    t2 = threading.Thread(target=server_side2, daemon=True)
    t2.start()
    cli2 = tp_engine.ControlChannel.connect("127.0.0.1", port2, deadline_s=20.0)
    cli2.send({"op": "hello"})
    t2.join(10.0)
    cli2.close()
    assert not errs and not errs2


def test_proto_mismatch_is_refused_at_the_handshake():
    port = ctrl.free_port()
    errs = []

    def worker():
        try:
            chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
            tp_engine.Rank1Worker(ctrl.ScriptedBackend([1, 2, 3]), chan).serve_forever()
        except tp_engine.ProtocolError:
            pass                                                   # expected: old rank 0 refused
        except BaseException as exc:                               # noqa: BLE001
            errs.append(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    cli.send({"op": "hello", "proto": tp_engine.PROTO - 1, "vocab_size": 32, "dspark": False,
              "vision": False})
    cli.set_recv_timeout(5.0)
    with pytest.raises((tp_engine.ChannelClosed, tp_engine.ProtocolError, OSError, AssertionError)):
        reply = cli.recv()
        if reply.get("op") == "error":
            raise tp_engine.ProtocolError(reply["message"])
        raise AssertionError(f"rank 1 accepted an old protocol: {reply}")
    t.join(10.0)
    cli.close()
    assert not errs and not t.is_alive()


# ---- the engine over a loopback channel with a scripted vision backend ------------------------------------------


class FakeTower:
    """Tower + aligner stand-in: rows are deterministic from the image's planned grid (the
    engine hands it the PIL image; the real tower derives the same grid from the pixels)."""

    def __init__(self, cfg, dim: int = 8):
        self.cfg, self.dim = cfg, dim
        self.seen = []

    def get_image_embeddings(self, tokenizer, image, text_alias=None):
        n_h, n_w = tp_vision.plan_grid(image.width, image.height, self.cfg)
        n = tp_vision.num_image_tokens(n_h, n_w)
        rows = torch.arange(n * self.dim, dtype=torch.float16).reshape(n, self.dim) / 8.0
        self.seen.append((n_h, n_w))
        from exllamav3.tokenizer.mm_embedding import MMEmbedding
        return MMEmbedding(embeddings=rows.clone(),
                           token_string=torch.full((1, n), -1, dtype=torch.long))


class VisionBackend(ctrl.ScriptedBackend):
    """ScriptedBackend + prefill recording with embeddings + a tower handle and vision cfg.

    Carries minimal checkpoint ops (stash/restore/drop_stash over position bookkeeping) so the
    engine's prefix cache stays enabled, as on the real backend."""

    def __init__(self, script, *, span=2048, cache_tokens=1 << 16, dim=8, vocab_size=129280):
        super().__init__(script, vocab_size=vocab_size)
        self.span = span
        self.cache_tokens = cache_tokens
        self.vision = FakeTower(VCFG, dim=dim)
        self.vision_cfg = VCFG
        self.prefill_calls = []
        self._pos: dict[str, int] = {}

    def new_state(self):
        state = super().new_state()
        self._pos[state] = 0
        return state

    def restore(self, stash, position):
        state = stash["state"]
        self.calls.append(("restore", int(position)))
        self._pos[state] = int(position)
        return state

    def stash(self, state):
        self.calls.append(("stash", self._pos.get(state, -1)))
        return {"state": state, "pos": self._pos.get(state, -1)}

    def drop_stash(self, stash):
        self.calls.append(("drop_stash", stash["pos"]))

    def prefill(self, ids, state, spec=False, start=0, embeddings=None):
        self.prefill_calls.append({
            "ids": [int(t) for t in ids], "spec": bool(spec), "start": int(start),
            "embeddings": None if embeddings is None else list(embeddings),
        })
        self._pos[state] = len(ids) - 1
        return {}

    def forward(self, token, position, state, embeddings=None):
        self.calls.append(("forward", int(token), int(position),
                           None if embeddings is None else len(embeddings)))
        tok = self.script[self._k] if self._k < len(self.script) else self.script[-1]
        self._k += 1
        row = torch.zeros(self.vocab_size)
        row[tok % self.vocab_size] = 8.0
        return row


class VisionPair:
    """Rank-0 TPEngine + rank-1 Rank1Worker over a real loopback channel, vision-capable."""

    def __init__(self, script, *, ctx=4096, span=8, dim=8):
        self.be0 = VisionBackend(script, span=span, dim=dim)
        self.be1 = VisionBackend(script, span=span, dim=dim)
        self.port = ctrl.free_port()
        self.errors: list[BaseException] = []
        self.thread = None
        self.chan0 = None
        self.engine = None
        self.ctx = ctx

    def start(self):
        def rank1():
            try:
                chan1 = tp_engine.ControlChannel.connect("127.0.0.1", self.port, deadline_s=30.0)
                chan1.set_recv_timeout(None)
                tp_engine.Rank1Worker(self.be1, chan1).serve_forever()
            except tp_engine.ChannelClosed:
                pass
            except BaseException as exc:                              # noqa: BLE001
                self.errors.append(exc)

        self.thread = threading.Thread(target=rank1, daemon=True)
        self.thread.start()
        self.chan0 = tp_engine.ControlChannel.listen("127.0.0.1", self.port, accept_timeout=30.0)
        self.chan0.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": self.be0.vocab_size,
                         "dspark": False, "vision": True})
        ok = self.chan0.recv()
        assert ok.get("op") == "hello_ok" and ok.get("vision_capable") is True
        self.engine = tp_engine.TPEngine(self.be0, self.chan0, VisionTokenizer(),
                                         ctx=self.ctx, eos_id=1)

    def run(self, prompt: str, images, *, max_tokens=4, **kw):
        async def go():
            out = []
            async for piece in self.engine.generate(prompt, images=images, max_tokens=max_tokens,
                                                    temperature=0.0, top_p=1.0, stop=[], **kw):
                out.append(piece)
            return out

        out = asyncio.run(go())
        if images is not None:
            self.sync()
        return out

    def sync(self, timeout: float = 5.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if len(self.be1.prefill_calls) >= len(self.be0.prefill_calls) \
                    and self.be1.released >= self.be0.released:
                return
            time.sleep(0.005)
        raise AssertionError(f"rank 1 did not catch up: {self.errors}")

    def stop(self):
        if self.engine is not None:
            try:
                self.engine.close()
            except Exception:                                         # noqa: BLE001
                pass
        if self.thread is not None:
            self.thread.join(timeout=10.0)
            assert not self.thread.is_alive(), f"rank-1 thread stuck: {self.errors}"


def _two_images(engine):
    return engine.prepare_images([
        {"data": base64.b64encode(tiny_png(64, 48)).decode("ascii")},
        {"data": base64.b64encode(tiny_png(32, 40)).decode("ascii")},
    ])


@needs_exl
def test_engine_prepare_prompt_ids_and_generate_end_to_end():
    pair = VisionPair(ctrl.ids_of("vision holds!"), span=8)
    try:
        pair.start()
        eng = pair.engine
        assert eng.supports_vision and eng.image_token_id == IMAGE_TOKEN
        images = _two_images(eng)
        assert images[0].first_index != images[1].first_index
        prompt = "a<｜deepseek_image｜>b<｜deepseek_image｜>c"
        ids = eng.prompt_ids(prompt, images)
        plain = eng.prompt_ids(prompt)                              # text view: 2 placeholder ids
        assert len(ids) == len(plain) - 2 + images[0].n_tokens + images[1].n_tokens
        assert ids.count(images[0].token_ids[0]) == 1               # ranges never overlap
        out = "".join(pair.run(prompt, images))
        assert out == "visi"
        stats = eng.last_stats
        assert stats["images"] == 2
        assert stats["image_tokens"] == images[0].n_tokens + images[1].n_tokens
        assert stats["cached_tokens"] == 0                           # cache bypassed for images
        # both ranks prefilled the SAME expanded ids, with the SAME embedding ranges attached
        assert [c["ids"] for c in pair.be0.prefill_calls] == [c["ids"] for c in pair.be1.prefill_calls]
        for be in (pair.be0, pair.be1):
            assert all(c["embeddings"] is not None for c in be.prefill_calls)
            embs = be.prefill_calls[0]["embeddings"]
            assert [e.first_index for e in embs] == [im.first_index for im in images]
            assert [int(e.mm_length) for e in embs] == [im.n_tokens for im in images]
        # rank 1's rows are bit-exact reconstructions of rank 0's
        for e0, e1 in zip(pair.be0.prefill_calls[0]["embeddings"], pair.be1.prefill_calls[0]["embeddings"]):
            assert torch.equal(e0.embeddings, e1.embeddings)
        assert not pair.errors
    finally:
        pair.stop()


@needs_exl
def test_image_request_left_the_prefix_cache_unharmed_for_text():
    pair = VisionPair(ctrl.ids_of("cache!"), span=8)
    try:
        pair.start()
        eng = pair.engine
        assert eng.prefix is not None
        images = _two_images(eng)
        pair.run("look<｜deepseek_image｜> and <｜deepseek_image｜> here", images)
        assert eng.prefix.ckpts == []                               # no checkpoint saved (bypass)
        out = "".join(pair.run("plain text only", None))            # text requests still serve
        assert out == "cach"
        assert eng.last_stats["spec_used"] is False
        assert not pair.errors
    finally:
        pair.stop()


@needs_exl
def test_text_requests_carry_no_indexed_embeddings_and_unchanged_calls():
    pair = VisionPair(ctrl.ids_of("text stays!"), span=8)
    try:
        pair.start()
        pair.run("plain prompt", None)
        assert all(c["embeddings"] is None for c in pair.be0.prefill_calls)
        assert all(c["embeddings"] is None for c in pair.be1.prefill_calls)
        # the decode loop's forward calls attach nothing (3-arg style, as before vision)
        assert all(c[3] is None for c in pair.be0.calls if c[0] == "forward")
        assert all(c[3] is None for c in pair.be1.calls if c[0] == "forward")
        assert not pair.errors
    finally:
        pair.stop()


@needs_exl
def test_engine_refuses_broken_image_prompts_before_any_dispatch():
    pair = VisionPair(ctrl.ids_of("refuse!"), span=8)
    try:
        pair.start()
        eng = pair.engine
        images = _two_images(eng)
        # MM ids without images (e.g. a /v1/completions prompt of bare 1e9+ ids)
        with pytest.raises(ValueError, match="no images were sent"):
            pair.run("", None, prompt_ids=[5, FIRST, FIRST + 1, 7])
        # images but no placeholder in the prompt
        with pytest.raises(ValueError, match="placeholder"):
            pair.run("no marker here", images)
        # placeholder count mismatch (two images, one placeholder)
        with pytest.raises(ValueError, match="placeholder"):
            pair.run("only one<｜deepseek_image｜>end", images)
        # prompt ending on an image span (the decode loop's first forward must be text)
        with pytest.raises(ValueError, match="end with a text token"):
            pair.run("ends on<｜deepseek_image｜>", images[:1])
        # expanded length beyond the context
        eng.ctx = 8
        with pytest.raises(ValueError, match="does not fit the context"):
            pair.run("x<｜deepseek_image｜>y", images[:1])
        eng.ctx = pair.ctx
        assert pair.be0.prefill_calls == [] and pair.be1.prefill_calls == []   # nothing dispatched
        assert not pair.errors
    finally:
        pair.stop()


@needs_exl
def test_overlong_image_prompt_counts_the_expanded_length():
    pair = VisionPair(ctrl.ids_of("long!"), span=8)
    try:
        pair.start()
        eng = pair.engine
        images = eng.prepare_images([{"data": base64.b64encode(tiny_png(64, 64)).decode("ascii")}])
        eng.ctx = images[0].n_tokens + 4                            # the span alone almost fills it
        with pytest.raises(ValueError, match="does not fit the context"):
            pair.run("image<｜deepseek_image｜>tail tail tail tail tail", images)
        assert pair.be0.prefill_calls == [] and pair.be1.prefill_calls == []
    finally:
        pair.stop()


@needs_exl
def test_rank1_refuses_image_ids_without_a_shipment():
    port = ctrl.free_port()
    errs = []

    def worker():
        try:
            chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
            chan.set_recv_timeout(None)
            tp_engine.Rank1Worker(VisionBackend([1, 2, 3]), chan).serve_forever()
        except tp_engine.ChannelClosed:
            pass
        except ValueError:
            pass                                                   # reported as an error frame, then dies
        except BaseException as exc:                               # noqa: BLE001
            errs.append(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    cli.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": 129280,
              "dspark": False, "vision": True})
    assert cli.recv()["op"] == "hello_ok"
    cli.send({"op": "begin", "req": 0, "ids": [5, FIRST, FIRST + 1, 7], "spec": False})
    reply = cli.recv()
    assert reply["op"] == "error" and "no imgmeta" in reply["message"]
    t.join(10.0)
    cli.close()
    assert not t.is_alive() and not errs                           # clean death, no hang


@needs_exl
def test_rank1_rejects_ids_outside_the_shipped_ranges():
    port = ctrl.free_port()
    errs = []

    def worker():
        try:
            chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
            chan.set_recv_timeout(None)
            tp_engine.Rank1Worker(VisionBackend([1, 2, 3]), chan).serve_forever()
        except tp_engine.ChannelClosed:
            pass
        except ValueError:
            pass
        except BaseException as exc:                               # noqa: BLE001
            errs.append(exc)

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    cli = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    cli.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": 129280,
              "dspark": False, "vision": True})
    assert cli.recv()["op"] == "hello_ok"
    meta, blobs = tp_vision.encode_imgmeta([_prepared_with_rows(6, 8)])
    cli.send_with_blobs({**meta, "req": 0}, blobs)
    assert cli.recv()["op"] == "imgmeta_ok"
    outside = FIRST + 6 + 100                                      # past the shipped range
    cli.send({"op": "begin", "req": 0, "ids": [5, FIRST, outside, 7], "spec": False})
    reply = cli.recv()
    assert reply["op"] == "error" and "outside every shipped range" in reply["message"]
    t.join(10.0)
    cli.close()
    assert not t.is_alive() and not errs


@needs_exl
def test_shipment_reply_mismatch_stops_the_request_before_any_model_call():
    """A rank 1 that answers imgmeta_ok with the wrong row count stops the request on rank 0
    before any model call (the lockstep equivalent of a refused shipment)."""
    pair = VisionPair(ctrl.ids_of("ship!"), span=8)
    try:
        pair.start()
        images = pair.engine.prepare_images(
            [{"data": base64.b64encode(tiny_png()).decode("ascii")}])

        def lying_ship(self, imgs, embeddings):
            reply = {"op": "imgmeta_ok", "images": len(embeddings), "rows": 3}    # wrong row count
            raise RuntimeError(f"rank 1 rejected the image shipment: {reply}")

        original = tp_engine.TPEngine._ship_embeddings
        tp_engine.TPEngine._ship_embeddings = lying_ship
        try:
            with pytest.raises(RuntimeError, match="rejected the image shipment"):
                pair.run("x<｜deepseek_image｜>y", images)
        finally:
            tp_engine.TPEngine._ship_embeddings = original
        assert pair.be0.prefill_calls == []                        # nothing reached the model
    finally:
        pair.stop()


# ---- chunk offsets with embeddings (the real chunking, a fake core) ----------------------------------------------


class RecordingCore(FakeCore):
    def _snap(self, ids, params):
        snap = super()._snap(ids, params)
        snap["indexed"] = len(params["indexed_embeddings"]) if params.get("indexed_embeddings") else None
        return snap


def _mm_standin(n, dim=8, first=FIRST):
    """A minimal duck-typed stand-in for MMEmbedding (the params path only reads these)."""
    return SimpleNamespace(first_index=first, mm_length=n, embeddings=torch.zeros(n, dim),
                           deepstack_embeddings=None)


def test_every_prefill_chunk_carries_indexed_embeddings_at_its_offset():
    be = make_backend(span=8, cache_tokens=64, page_size=16, core=RecordingCore())
    embeds = [_mm_standin(6), _mm_standin(10, first=FIRST + 6)]
    ids = [1, 2, FIRST, FIRST + 1, FIRST + 2, FIRST + 3, FIRST + 4, FIRST + 5, 3,
           FIRST + 6, FIRST + 7, 4, 5]
    be.prefill(ids, "state", embeddings=embeds)
    calls = be.model.prefill_calls
    assert [c["past_len"] for c in calls] == [0, 8]                # chunking unchanged by images
    assert all(c["indexed"] == 2 for c in calls)                   # the list rides EVERY chunk
    assert [t for c in calls for t in c["ids"]] == ids[:-1]        # each row exactly once, in order
    # text-only prefill: no indexed_embeddings key at all (byte-for-byte the old params)
    be2 = make_backend(span=8, cache_tokens=64, page_size=16, core=RecordingCore())
    be2.prefill(list(range(20)), "state")
    assert all(c["indexed"] is None for c in be2.model.prefill_calls)


# ---- make_mm_embedding: the object both ranks feed the library ----------------------------------------------------


@needs_exl
def test_make_mm_embedding_binds_the_explicit_range():
    rows = torch.arange(12, dtype=torch.float16).reshape(6, 2)
    mme = tp_engine.make_mm_embedding(FIRST + 40, rows, alias="<$EMB_test$>")
    assert mme.first_index == FIRST + 40 and mme.mm_length == 6 and mme.last_index == FIRST + 46
    assert mme.embeddings is rows and mme.deepstack_embeddings is None
    assert tp_engine.FIRST_MM_EMBEDDING_INDEX == FIRST             # engine base == library base


@needs_exl
def test_tower_plan_drift_is_refused_not_mis_spliced():
    pair = VisionPair(ctrl.ids_of("drift!"), span=8)
    try:
        pair.start()
        images = pair.engine.prepare_images(
            [{"data": base64.b64encode(tiny_png()).decode("ascii")}])

        def lying_tower(tokenizer, image, text_alias=None):
            n_h, n_w = tp_vision.plan_grid(image.width, image.height, VCFG)
            n = tp_vision.num_image_tokens(n_h, n_w) + 1           # one row more than planned
            from exllamav3.tokenizer.mm_embedding import MMEmbedding
            return MMEmbedding(embeddings=torch.zeros(n, 8, dtype=torch.float16),
                               token_string=torch.full((1, n), -1, dtype=torch.long))

        pair.be0.vision.get_image_embeddings = lying_tower
        with pytest.raises(RuntimeError, match="drifted"):
            pair.run("x<｜deepseek_image｜>y", images)
        assert pair.be0.prefill_calls == []
    finally:
        pair.stop()


# ---- the vendored HTTP layer over the real TP engine ------------------------------------------------------------


@needs_exl
def test_vendored_server_image_request_end_to_end():
    """render_chat_vl -> engine.prepare_images -> prompt_ids(prompt, images) -> generate(images=)
    through the vendored server, with usage.prompt_tokens counting the EXPANDED prompt."""
    from aiohttp.test_utils import TestClient, TestServer
    serve = kit.serve_module()
    script = ctrl.ids_of("a picture!</think>It is a red rectangle.")
    pair = VisionPair(script, ctx=4096, span=8)
    pair.start()

    async def go():
        client = TestClient(TestServer(serve.create_app(pair.engine, "tp-vision-test")))
        await client.start_server()
        try:
            body = {"model": "tp-vision-test", "max_tokens": 6, "temperature": 0.0,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "what is in these?"},
                        {"type": "image_url", "image_url": {"url": data_url(tiny_png(64, 48))}},
                        {"type": "image_url", "image_url": {"url": data_url(tiny_png(32, 40))}},
                    ]}]}
            r = await client.post("/v1/chat/completions", json=body)
            assert r.status == 200, await r.text()
            whole = await r.json()
            stats = pair.engine.last_stats
            assert stats["images"] == 2 and stats["image_tokens"] > 0
            # the server counted the EXPANDED prompt, and rank 0 prefilled exactly those ids
            # (the fake backend records the full id list the engine hands it)
            assert whole["usage"]["prompt_tokens"] == stats["prompt_tokens"] == len(
                pair.be0.prefill_calls[0]["ids"])
            assert any(t >= FIRST for t in pair.be0.prefill_calls[0]["ids"])
            assert [c["ids"] for c in pair.be0.prefill_calls] == [c["ids"] for c in pair.be1.prefill_calls]
            # a broken image is a clean 400 that reaches neither rank's model
            bad = {"model": "tp-vision-test", "max_tokens": 4, "temperature": 0.0,
                   "messages": [{"role": "user", "content": [
                       {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                   ]}]}
            before = len(pair.be0.prefill_calls)
            r = await client.post("/v1/chat/completions", json=bad)
            assert r.status == 400 and "image" in (await r.text()).lower()
            assert len(pair.be0.prefill_calls) == before
            h = await (await client.get("/health")).json()
            assert h["vision"] is True
        finally:
            await client.close()

    asyncio.run(go())
    pair.sync()
    pair.stop()
