"""Stage-2 NVMe overflow (tp_disk + the spill/load control frames) over loopback channels.

The disk stand-ins extend the session stand-ins with the backend hooks the engine ducks
onto: a page's blob is its row dict (so a wrong logical->physical mapping at reload
changes the output exactly like it would on the GPU), and stash payloads round-trip
through the same live-checkpoint registry the drop path asserts against. Every reuse
scenario runs against an eviction-only control (tier disabled): replies must be
identical, and the disk path must only change HOW MUCH is re-prefilled, never the output.
The tier itself is also driven directly (roundtrip, truncation, bit flips, magic,
fingerprint, cap, atomicity, permissions). CPU only; tmp dirs, no GPU, no model files.
"""
from __future__ import annotations

import asyncio
import os
import pickle
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tp_disk  # noqa: E402
import tp_engine  # noqa: E402
import tp_sessions  # noqa: E402
from test_tp_control import free_port  # noqa: E402
from test_tp_prefix_cache import Tok, ids_of, prompt  # noqa: E402
from test_tp_sessions import (PAGE, SessionBackend, SessionRanks, SessionSpecBackend,  # noqa: E402
                              _ms_hello, _ms_worker)

DISK_HOOKS = ("disk_fingerprint", "disk_page_blob", "disk_page_store", "disk_stash_pack",
              "disk_stash_unpack")


def _disk_subclass(cls):
    """Build a DISK-CAPABLE SUBCLASS of a session stand-in (never mutate the base: a
    combined `pytest test_tp_sessions.py test_tp_disk.py` run must not leak tier behavior
    into the stage-1 suites -- their backends keep lacking the hooks, so those engines
    stay eviction-only and write no images)."""
    doc = f"Disk-tier hooks bolted onto {cls.__name__} (subclass; the base stays clean)."

    def disk_fingerprint(self):
        return {"layout": "standin", "cache_tokens": self.cache_tokens,
                "page_size": self.page_size, "num_pages": self.pool_pages}

    def disk_page_blob(self, page):
        self.calls.append(("disk_blob", int(page)))
        return pickle.dumps(self.rows.get(int(page), {}))

    def disk_page_store(self, page, blob):
        self.calls.append(("disk_store", int(page)))
        self.rows[int(page)] = pickle.loads(blob)                      # noqa: S301 (test stand-in)

    def disk_stash_pack(self, stash):
        return {"pos": stash["pos"], "hist": list(stash["hist"]), "sid": stash["sid"],
                "pages": list(stash["pages"])}

    def disk_stash_unpack(self, payload):
        s = dict(payload)
        # a reloaded conversation may land in a different slot with different physical
        # pages: the stash carries no session identity past the reload (row correctness
        # is proven by the byte-exact control comparison in every scenario)
        s["sid"] = None
        s["pages"] = []
        self.live[id(s)] = s
        return s

    return type(f"Disk{cls.__name__}", (cls,), {
        "__doc__": doc,
        "disk_fingerprint": disk_fingerprint,
        "disk_page_blob": disk_page_blob,
        "disk_page_store": disk_page_store,
        "disk_stash_pack": disk_stash_pack,
        "disk_stash_unpack": disk_stash_unpack,
    })


DiskBackend = _disk_subclass(SessionBackend)
DiskSpecBackend = _disk_subclass(SessionSpecBackend)


def _assert_bases_clean():
    """The stage-1 stand-ins must stay tier-free even after this module was imported
    (the hygiene guarantee the combined run depends on)."""
    for cls in (SessionBackend, SessionSpecBackend):
        for hook in DISK_HOOKS:
            assert not hasattr(cls, hook), f"{cls.__name__} leaked the {hook} disk hook"


class DiskHarness:
    """A SessionRanks-equivalent built directly from the disk-capable stand-ins (per-rank
    tier dirs, no environment): rank-0 engine + the real Rank1Worker over loopback."""

    def __init__(self, *, spec: bool = False, sessions: int = 2, pool_pages: int = 32,
                 disk_root: Path, cap_gib: float | None = None, disk_on: bool = True,
                 record: bool = False):
        cls = DiskSpecBackend if spec else DiskBackend
        self.be0, self.be1 = cls(pool_pages, PAGE), cls(pool_pages, PAGE)
        self.sessions = sessions
        self.errors: list[BaseException] = []
        self.frames: list[dict] = []
        self.disk_dirs = (disk_root / "rank0", disk_root / "rank1")
        for d in self.disk_dirs:
            d.mkdir(parents=True, exist_ok=True)
        on = {"enabled": True} if disk_on else {"enabled": False}
        self.opts0 = {"root": str(self.disk_dirs[0]), "cap_gib": cap_gib, **on}
        self.opts1 = {"root": str(self.disk_dirs[1]), "cap_gib": cap_gib, **on}
        port = free_port()

        def rank1():
            try:
                chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=30.0)
                chan.set_recv_timeout(None)
                self.worker = tp_engine.Rank1Worker(self.be1, chan, sessions=sessions,
                                                    disk_opts=self.opts1, log=lambda *_: None)
                self.worker.serve_forever()
            except BaseException as exc:                                  # noqa: BLE001
                self.errors.append(exc)

        self.thread = threading.Thread(target=rank1, daemon=True)
        self.thread.start()
        self.chan = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=30.0)
        self.chan.send({"op": "hello", "proto": tp_engine.PROTO_SESSIONS, "vocab_size": 129280,
                        "dspark": spec, "sessions": sessions, "pool_pages": pool_pages,
                        "page_size": PAGE})
        ok = self.chan.recv()
        assert ok.get("op") == "hello_ok" and ok.get("sessions") == sessions, ok
        if record:
            from test_tp_sessions import _RecordingChannel
            self.chan = _RecordingChannel(self.chan, self.frames)
        loud = print if os.environ.get("TP_DISK_DEBUG") else (lambda *_: None)
        self.engine = tp_engine.TPEngine(
            self.be0, self.chan, Tok(), ctx=pool_pages * PAGE - 2, eos_id=None,
            sessions=sessions, session_opts={"prefix_slots": 2, "min_tokens": 8},
            disk_opts=self.opts0, log=loud)
        assert self.engine.mgr is not None
        if disk_on:
            assert self.engine.tier is not None and self.engine.tier.enabled
            assert self.worker.tier is not None and self.worker.tier.enabled

    run = SessionRanks.run
    sync = SessionRanks.sync
    mirrored = staticmethod(SessionRanks.mirrored)
    sid_of = SessionRanks.sid_of

    def check(self):
        assert self.mirrored(self.be0) == self.mirrored(self.be1), "rank-0/rank-1 model calls diverged"
        mgr = self.engine.mgr
        assert set(mgr.sessions) == set(self.worker.sess)
        for sid, s in mgr.sessions.items():
            rec = self.worker.sess[sid]
            assert s.pages == rec["pages"], (sid, s.pages, rec["pages"])
            assert sorted(c.key for c in s.px.ckpts) == sorted(rec["stashes"])
        claimed = [p for rec in self.worker.sess.values() for p in rec["pages"]]
        assert len(claimed) == len(set(claimed))
        assert len(claimed) == mgr.pool.used
        n_ck = sum(len(s.px.ckpts) for s in mgr.sessions.values())
        assert len(self.be0.live) == len(self.be1.live) == n_ck
        assert not self.errors, self.errors

    def stop(self):
        self.engine.close()
        self.thread.join(timeout=10.0)
        assert not self.thread.is_alive() and not self.errors, self.errors

    def status(self):
        return self.engine.session_status()


@pytest.fixture(scope="module")
def disk_root(tmp_path_factory):
    return tmp_path_factory.mktemp("tp-disk")


# ---- the tier, driven directly ------------------------------------------------------------------------------

def test_tier_roundtrip_checksums_and_permissions(disk_root):
    tier = tp_disk.DiskTier(root=str(disk_root / "unit"), rank=0, cap_gib=1,
                            fingerprint={"layout": "x", "cache_tokens": 64, "page_size": PAGE})
    blobs = [pickle.dumps({i: i * 7}) for i in range(4)]
    payload = {(1, 2): (3, {"pos": 3, "hist": [1, 2, 3]})}
    size = tier.write_image(5, meta={"positions": {"1": 2}}, n_pages=len(blobs),
                            page_blob_fn=lambda i: blobs[i], stash_payload=payload)
    assert tier.has(5) and 0 < size < 4096
    assert stat.S_IMODE(os.stat(tier.path(5)).st_mode) == 0o600
    got = {}

    def blobs_into(i, blob):
        got[i] = pickle.loads(blob)                                     # noqa: S301

    def stash_into(p):
        got["stash"] = p

    head = tier.read_image(5, blobs_into=blobs_into, stash_into=stash_into)
    assert [got[i] for i in range(4)] == [{i: i * 7} for i in range(4)]
    assert got["stash"] == payload and head["positions"] == {"1": 2}

    def bytes_into(i, blob):                          # no unpickling: corrupted reads
        pass                                          # deliver dead bytes only

    # fingerprint mismatch refuses
    with pytest.raises(tp_disk.DiskCorrupt, match="incompatible"):
        tier.read_image(5, blobs_into=bytes_into, stash_into=stash_into,
                        expect={"layout": "OTHER"})
    # truncation and bit flips refuse. (bytes_into accepts the streamed blobs without
    # unpickling: page blobs stream into the caller BEFORE the whole-region checksum
    # verdict -- the documented contract; the pages are dead rows unless the load fully
    # succeeds. The stash pickle is only handed over after every checksum passed, which
    # is the security property.)
    tier.drop(5)
    tier.write_image(5, meta={"positions": {"1": 2}}, n_pages=len(blobs),
                     page_blob_fn=lambda i: blobs[i], stash_payload=payload)
    raw = tier.path(5).read_bytes()
    tier.path(5).write_bytes(raw[: len(raw) - 10])
    with pytest.raises(tp_disk.DiskCorrupt):
        tier.read_image(5, blobs_into=bytes_into, stash_into=stash_into)
    tier.path(5).write_bytes(raw[:24] + bytes([raw[24] ^ 0xFF]) + raw[25:])
    with pytest.raises(tp_disk.DiskCorrupt):
        tier.read_image(5, blobs_into=bytes_into, stash_into=stash_into)
    tier.drop(5)
    assert not tier.has(5)
    with pytest.raises(tp_disk.DiskCorrupt, match="missing"):
        tier.read_image(5, blobs_into=bytes_into, stash_into=stash_into)
    # no temp files left behind
    assert not [p for p in Path(tier.root).iterdir() if p.name.startswith(".")]


def test_tier_cap_enforced_and_accounted(disk_root):
    tier = tp_disk.DiskTier(root=str(disk_root / "unit-cap"), rank=1, cap_gib=0.00005,
                            fingerprint={})
    big = b"x" * 65536                                   # ~64 KiB: over the ~53 KiB cap
    with pytest.raises(tp_disk.DiskError, match="cap"):
        tier.write_image(1, meta={}, n_pages=1, page_blob_fn=lambda i: big,
                        stash_payload=None)
    assert not list(tier.root.glob("*.img")) and tier.used_bytes() == 0
    assert not list(tier.root.glob("*.tmp"))             # the aborted write left nothing


def test_tier_sweeps_stale_images_and_tmp_files_at_startup(disk_root):
    """L2: images are one-shot, so anything a previous run left is dead weight -- the tier
    removes its own rank's stale images and tmp files at startup, never another rank's."""
    root = disk_root / "unit-sweep"
    root.mkdir(parents=True, exist_ok=True)
    stale_img = root / "rank0-i7.img"
    stale_img.write_bytes(b"old")
    stale_tmp = root / ".rank0-i7.123.tmp"
    stale_tmp.write_bytes(b"partial")
    other_rank = root / "rank1-i7.img"
    other_rank.write_bytes(b"not mine")
    tier = tp_disk.DiskTier(root=str(root), rank=0, cap_gib=1, fingerprint={})
    assert tier.counters["startup_sweep"] == 2
    assert not stale_img.exists() and not stale_tmp.exists()
    assert other_rank.exists()                          # another rank's files untouched


def test_tier_streams_pages_without_materializing_the_list(disk_root):
    """H1: write_image asks for one page blob at a time and writes it before requesting
    the next -- proven by watching the tmp file grow between successive page_blob_fn
    calls (a list-materializing writer would call the fn for every page first)."""
    tier = tp_disk.DiskTier(root=str(disk_root / "unit-stream"), rank=0, cap_gib=1,
                            fingerprint={})
    blobs = [bytes([65 + i]) * (1000 * (i + 1)) for i in range(6)]
    tmp_sizes: list[int] = []
    seen: list[int] = []

    def page_blob_fn(i):
        tmp = next(iter(tier.root.glob(".rank0-i9.*.tmp")), None)
        tmp_sizes.append(tmp.stat().st_size if tmp is not None else 0)
        seen.append(i)
        return blobs[i]

    tier.write_image(9, meta={}, n_pages=len(blobs), page_blob_fn=page_blob_fn,
                     stash_payload={"k": 1})
    assert seen == list(range(6))                        # exactly one call per page, in order
    assert tmp_sizes[0] == len(tp_disk.MAGIC)            # only the magic precedes page 0
    for k in range(1, len(blobs)):                       # each blob landed before the next ask
        assert tmp_sizes[k] == tmp_sizes[k - 1] + 4 + len(blobs[k - 1]), (k, tmp_sizes)
    # and the image reads back identical
    got = {}
    tier.read_image(9, blobs_into=lambda i, b: got.setdefault(i, b),
                    stash_into=lambda p: got.setdefault("stash", p))
    assert [got[i] for i in range(6)] == blobs and got["stash"] == {"k": 1}


# ---- spill/reload over the engine ---------------------------------------------------------------------------

def _spill_scenario(h, *, a, b, c):
    """Three conversations in a pool/slot budget that spills (not evicts) the LRU."""
    out = [h.run(a, max_tokens=4)]
    out.append(h.run(b, max_tokens=4))
    out.append(h.run(c, max_tokens=4))          # spills the LRU (a) to make room
    return out


def test_spill_then_reload_restores_instead_of_reprefill(tmp_path):
    """A third conversation spills the LRU session to both ranks' images; its return
    RELOADS it (restore + prefill of the new turn only) instead of re-prefilling, and the
    replies match an eviction-only control byte for byte."""
    a, b, c = prompt(1, 40), prompt(2, 40), prompt(3, 40)
    on = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "on")
    off = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "off", disk_on=False)
    try:
        got_on = _spill_scenario(on, a=a, b=b, c=c)
        on.check()
        st = on.status()
        assert len(st["cold_sessions"]) == 1 and st["cold_sessions"][0]["session"] == 1
        assert st["cold_sessions"][0]["image"] == 1          # monotonic image ids
        assert on.engine.tier.counters["spills"] == 1
        assert (on.disk_dirs[0] / "rank0-i1.img").exists()
        assert (on.disk_dirs[1] / "rank1-i1.img").exists()
        got_off = _spill_scenario(off, a=a, b=b, c=c)
        off.check()
        # a returns: reload vs eviction
        a2 = a + ids_of(got_on[0][0]) + prompt(4, 12)
        got_on.append(on.run(a2, max_tokens=4))
        got_off.append(off.run(a + ids_of(got_off[0][0]) + prompt(4, 12), max_tokens=4))
        on.check()
    finally:
        on.stop()
        off.stop()
    assert [t for t, _ in got_on] == [t for t, _ in got_off]
    reloaded, evicted = got_on[3][1], got_off[3][1]
    assert reloaded["cached_tokens"] > 0                       # the reload restored
    assert evicted["cached_tokens"] == 0                       # the control re-prefilled
    assert reloaded["cached_tokens"] == 40 + 4 - 1             # a's end checkpoint
    assert on.engine.mgr.counters["cold_loads"] == 1


def test_four_hot_sessions_plus_cold_overflow(tmp_path):
    """Four conversations stay hot; a fifth spills the LRU to NVMe; everyone still gets
    correct replies and the cold one reloads on return."""
    convs = {s: prompt(10 + s, 40) for s in range(4)}
    on = DiskHarness(sessions=4, pool_pages=15, disk_root=tmp_path / "on")
    off = DiskHarness(sessions=4, pool_pages=15, disk_root=tmp_path / "off", disk_on=False)
    try:
        firsts = [on.run(convs[s], max_tokens=4) for s in range(4)]
        on.check()
        assert len(on.status()["sessions"]) == 4
        fifth = prompt(20, 40)
        f_on = on.run(fifth, max_tokens=4)
        on.check()
        st = on.status()
        assert len(st["cold_sessions"]) == 1                   # the LRU spilled, not died
        assert len(st["sessions"]) == 4                        # four hot slots
        f_off = off.run(prompt(20, 40), max_tokens=4)
        firsts_off = [off.run(convs[s], max_tokens=4) for s in range(4)]
        # the spilled conversation returns through the disk
        cold0 = min(on.engine.mgr.cold.values(), key=lambda c: c.tick)
        s0 = cold0.sid
        back = convs[s0] + ids_of(firsts[s0 - 1][0]) + prompt(30, 10)
        got = on.run(back, max_tokens=4)
        back_off = convs[s0] + ids_of(firsts_off[s0 - 1][0]) + prompt(30, 10)
        got_off = off.run(back_off, max_tokens=4)
        on.check()
        assert got[1]["cached_tokens"] > 0
    finally:
        on.stop()
        off.stop()
    assert got[0] == got_off[0] and f_on[0] == f_off[0]
    assert [t for t, _ in firsts] == [t for t, _ in firsts_off]


def _corrupt(path: Path):
    """Flip one byte INSIDE the page-data region (after the magic and the first blob's
    length header): a flip in the trailing JSON could legitimately still parse, but data
    bytes are covered by the data checksum and must always refuse."""
    raw = path.read_bytes()
    i = min(24, max(len(tp_disk.MAGIC) + 4, len(raw) // 4))
    path.write_bytes(raw[:i] + bytes([raw[i] ^ 0x5A]) + raw[i + 1:])


def test_corrupt_image_falls_back_to_reprefill_never_wrong_output(tmp_path):
    """A flipped bit (either rank's image) or a missing rank-0 file is a clean fallback:
    the conversation re-prefills as a fresh session, replies stay byte-exact, the bad
    image is dropped, and the counters say so."""
    a, b, c = prompt(41, 40), prompt(42, 40), prompt(43, 40)
    cases = ("rank0-corrupt", "rank1-corrupt", "rank0-missing")
    for case in cases:
        on = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / f"c-{case}")
        off = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / f"o-{case}",
                          disk_on=False)
        try:
            r_a = on.run(a, max_tokens=4)
            on.run(b, max_tokens=4)
            on.run(c, max_tokens=4)                  # spills the LRU (session 1 -> image 1)
            assert on.status()["cold_sessions"], case
            img0 = on.disk_dirs[0] / "rank0-i1.img"
            img1 = on.disk_dirs[1] / "rank1-i1.img"
            if case == "rank0-corrupt":
                _corrupt(img0)
            elif case == "rank0-missing":
                img0.unlink()
                assert img1.exists()
            else:
                _corrupt(img1)
            a2 = a + ids_of(r_a[0]) + prompt(44, 12)
            got = on.run(a2, max_tokens=4)           # the reload fails cleanly
            on.check()
            r_off = off.run(a, max_tokens=4)
            off.run(b, max_tokens=4)
            off.run(prompt(43, 40), max_tokens=4)
            got_off = off.run(a + ids_of(r_off[0]) + prompt(44, 12), max_tokens=4)
            assert got[0] == got_off[0], case        # never wrong output
            assert got[1]["cached_tokens"] == 0, case   # fell back to a fresh prefill
            assert on.engine.mgr.counters["spill_restore_fallbacks"] >= 1, case
            assert not img0.exists(), case           # the spent image is gone either way
        finally:
            on.stop()
            off.stop()


def test_cap_exceeded_spill_falls_back_to_eviction(tmp_path):
    """A tiny EXL3_TP_DISK_CACHE_GIB refuses the write on both ranks: the victim evicts
    (no cold record, no image), replies stay correct, and /tp/status shows the cap."""
    a, b, c = prompt(51, 40), prompt(52, 40), prompt(53, 40)
    on = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "cap",
                     cap_gib=0.000001)
    off = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "cap-off", disk_on=False)
    try:
        r_a = on.run(a, max_tokens=4)
        on.run(b, max_tokens=4)
        got = on.run(c, max_tokens=4)
        on.check()
        st = on.status()
        assert not st["cold_sessions"]                        # nothing parked
        assert on.engine.tier.counters["spill_errors"] >= 1
        assert st["disk"]["cap_gib"] < 0.001
        back = a + ids_of(r_a[0]) + prompt(54, 8)
        got2 = on.run(back, max_tokens=4)
        r_off = off.run(a, max_tokens=4)
        off.run(b, max_tokens=4)
        off.run(prompt(53, 40), max_tokens=4)
        got2_off = off.run(a + ids_of(r_off[0]) + prompt(54, 8), max_tokens=4)
        on.check()
    finally:
        on.stop()
        off.stop()
    assert [got[0], got2[0]] == [got[0], got2_off[0]]
    assert got2[1]["cached_tokens"] == 0


def test_spilled_session_survives_salt_reset_and_status_reports_disk(tmp_path):
    a = prompt(61, 40)
    on = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "reset")
    try:
        on.run(a, max_tokens=4)
        on.run(prompt(62, 40), max_tokens=4)
        on.run(prompt(63, 40), max_tokens=4)          # spill the LRU
        assert on.status()["cold_sessions"]
        asyncio.run(on.engine.reset_prefix_cache())
        assert not on.engine.mgr.sessions and not on.engine.mgr.cold
        assert not list((tmp_path / "reset" / "rank0").glob("*.img"))
        _, stats = on.run(prompt(64, 20), max_tokens=4)
        assert stats["cached_tokens"] == 0
        on.check()
        st = on.status()
        assert st["disk"]["enabled"] and st["disk"]["rank"] == 0 and "counters" in st["disk"]
    finally:
        on.stop()


def test_s1_engine_never_builds_a_tier(tmp_path):
    """EXL3_TP_SESSIONS=1 keeps the whole disk tier out (no dir, no wire changes) even with
    the hooks present and EXL3_TP_DISK_CACHE on."""
    be = DiskBackend(16, PAGE)
    port = free_port()
    errors: list[BaseException] = []

    def rank1():
        try:
            chan = tp_engine.ControlChannel.connect("127.0.0.1", port, deadline_s=20.0)
            chan.set_recv_timeout(None)
            tp_engine.Rank1Worker(be, chan).serve_forever()
        except BaseException as exc:                                  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=rank1, daemon=True)
    t.start()
    chan = tp_engine.ControlChannel.listen("127.0.0.1", port, accept_timeout=20.0)
    chan.send({"op": "hello", "proto": tp_engine.PROTO, "vocab_size": 129280, "dspark": False})
    assert chan.recv()["op"] == "hello_ok"
    engine = tp_engine.TPEngine(be, chan, Tok(), ctx=254, eos_id=None,
                                prefix=tp_engine.PrefixCache(slots=2, min_tokens=8))
    try:
        assert engine.sessions == 1 and engine.mgr is None and engine.tier is None
        asyncio.run(_consume(engine, prompt(71, 40)))
        assert engine.tier is None                    # still no tier after serving
    finally:
        engine.close()
        t.join(5.0)
    assert not errors, errors


async def _consume(engine, ids):
    out = []
    async for piece in engine.generate("", prompt_ids=ids, max_tokens=4, temperature=0.0,
                                       top_p=1.0, stop=[]):
        out.append(piece)
    return "".join(out)


# ---- protocol: the frames, their acks, and their failure paths ----------------------------------------------

def test_spill_and_load_frames_on_the_wire(tmp_path):
    """Frame-level lockstep: the spill frame precedes the begin of the request that needed
    room, carries the image id and the session's keys, and is acked with spill_ok + a byte
    count; the load frame precedes the begin of the returning request, carries the NEW
    pages and the checkpoint keys/positions, and is acked with load_ok."""
    a, b, c = prompt(81, 40), prompt(82, 40), prompt(83, 40)
    h = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "wire", record=True)
    try:
        h.run(a, max_tokens=4)
        h.run(b, max_tokens=4)
        h.frames.clear()
        h.run(c, max_tokens=4)                        # spills a
        spills = [f for f in h.frames if f.get("op") == "spill"]
        assert len(spills) == 1 and spills[0]["session"] == 1 and spills[0]["keys"]
        assert isinstance(spills[0].get("image"), int)
        img = spills[0]["image"]
        h.frames.clear()
        h.run(a + prompt(84, 10), max_tokens=4)       # reloads a (header turn)
        loads = [f for f in h.frames if f.get("op") == "load"]
        assert len(loads) == 1
        lf = loads[0]
        assert lf["image"] == img and lf["pages"] and lf["keys"] and lf["positions"]
        begins = [f for f in h.frames if f.get("op") == "begin"]
        assert begins and begins[0].get("session") == lf["session"]
        h.check()
    finally:
        h.stop()


def test_cold_prefix_twin_chain_falls_back_to_fresh_not_raise(tmp_path):
    """Review fix (cold replan thrash): when the best-match cold image fails AND a second
    cold prefix-twin also matches, the routing tries each matching cold AT MOST ONCE and
    then serves a fresh session -- never the old 'session routing did not settle' 500,
    never wrong output, and no cold record or image is orphaned."""
    from array import array as _array
    a, b, c = prompt(201, 40), prompt(202, 40), prompt(203, 40)
    on = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "twins")
    off = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "twins-off",
                      disk_on=False)
    try:
        r_a = on.run(a, max_tokens=4)                 # session 1
        on.run(b, max_tokens=4)                       # session 2
        on.run(c, max_tokens=4)                       # spills 1 -> image 1
        real = next(iter(on.engine.mgr.cold.values()))
        assert real.seq[:4] and real.ckpt_meta
        # a prefix TWIN: a second cold record with the SAME committed sequence, whose
        # image does not exist (the failure mode the old code turned into a 500)
        twin = tp_sessions.ColdSession(sid=2, img=99, seq=_array("i", real.seq),
                                       ckpt_meta=list(real.ckpt_meta),
                                       drafts=dict(real.drafts),
                                       last_prompt_len=real.last_prompt_len,
                                       pages_n=real.pages_n, tick=real.tick + 5)
        on.engine.mgr.cold[99] = twin
        _corrupt(on.disk_dirs[0] / "rank0-i1.img")    # the real image fails too
        a2 = a + ids_of(r_a[0]) + prompt(204, 12)
        got = on.run(a2, max_tokens=4)                # twin fails, real fails, FRESH
        on.check()
        r_off = off.run(a, max_tokens=4)
        off.run(b, max_tokens=4)
        off.run(prompt(203, 40), max_tokens=4)
        got_off = off.run(a + ids_of(r_off[0]) + prompt(204, 12), max_tokens=4)
    finally:
        on.stop()
        off.stop()
    assert got[0] == got_off[0]                       # never wrong output
    assert got[1]["cached_tokens"] == 0               # served fresh, no raise
    assert on.engine.mgr.counters["spill_restore_fallbacks"] >= 2   # both colds tried once
    # the two ATTEMPTED colds are gone; anything left is a later, legitimate spill
    assert {1, 99}.isdisjoint(set(on.engine.mgr.cold))
    assert not (tmp_path / "twins" / "rank0" / "rank0-i1.img").exists()


def test_disk_hooks_do_not_leak_into_stage1_backends():
    """Review fix (test hygiene): importing test_tp_disk must leave the stage-1 stand-ins
    tier-free -- combined runs keep stage-1 engines eviction-only (no images written)."""
    import test_tp_sessions as stage1
    _assert_bases_clean()
    for cls in (stage1.SessionBackend, stage1.SessionSpecBackend):
        for hook in DISK_HOOKS:
            assert not hasattr(cls, hook), f"{cls.__name__} leaked the {hook} disk hook"
    # and the disk flavors really carry them (subclasses, not the bases)
    for cls in (DiskBackend, DiskSpecBackend):
        for hook in DISK_HOOKS:
            assert hasattr(cls, hook)


def test_explicit_sid_refused_for_spilled_slot_never_loads_wrong_cold(tmp_path):
    """Review fix (explicit sid vs cold registry): the cold registry is keyed by IMAGE id,
    so \"session\": N must never index it -- an arbitrary image would come back as the
    wrong conversation. An explicit id on a FREE slot that a cold conversation was spilled
    FROM is a clear error; the same prompt without the override reloads the RIGHT image,
    and no cold record or image is orphaned or wrongly consumed."""
    from array import array as _array
    a, b, c = prompt(211, 40), prompt(212, 40), prompt(213, 40)
    on = DiskHarness(sessions=3, pool_pages=8, disk_root=tmp_path / "explicit")
    try:
        r_a = on.run(a, max_tokens=4)                 # session 1
        on.run(b, max_tokens=4)                       # session 2
        on.run(c, max_tokens=4)                       # spills session 1 -> image 1
        colds = [x for x in on.engine.mgr.cold.values()]
        assert len(colds) == 1 and colds[0].sid == 1 and colds[0].img == 1
        # slot 1 is free (its conversation spilled; the newcomer took slot 3) AND a cold
        # record claims it (synthetic, unrelated sequence): pinning it explicitly must
        # refuse -- never reload some arbitrary image by image-id
        other = tp_sessions.ColdSession(sid=1, img=77, seq=_array("i", prompt(999, 40)),
                                        ckpt_meta=[], drafts={},
                                        last_prompt_len=40, pages_n=3, tick=colds[0].tick + 3)
        on.engine.mgr.cold[77] = other
        with pytest.raises(ValueError, match="spilled to the disk tier .*WITHOUT"):
            on.run(prompt(216, 30), max_tokens=4, session=1)
        on.check()
        assert 77 in on.engine.mgr.cold               # the refusal consumed nothing
        assert not (tmp_path / "explicit" / "rank0" / "rank0-i77.img").exists()
        # the spilled conversation itself: no override -> reloads the RIGHT image
        a2 = a + ids_of(r_a[0]) + prompt(215, 10)
        got = on.run(a2, max_tokens=4)
        on.check()
        assert got[1]["cached_tokens"] > 0            # reloaded, not re-prefilled
        assert on.engine.mgr.counters["cold_loads"] == 1
        assert 1 not in on.engine.mgr.cold            # the right cold consumed
        assert 77 in on.engine.mgr.cold               # the unrelated cold untouched
        # an explicit id for a plain hot slot stays a normal pin (slot 3 holds the third
        # conversation and was never spilled; slot 2's conversation is cold by now and
        # correctly refuses)
        t, st = on.run(prompt(217, 30), max_tokens=4, session=3)
        assert st["session"] == 3
        with pytest.raises(ValueError, match="spilled to the disk tier"):
            on.run(prompt(218, 30), max_tokens=4, session=2)
    finally:
        on.stop()


def test_worker_spill_failure_is_graceful(tmp_path):
    """A rank 1 that cannot write (its tier dir is unwritable) answers spill_error and
    stays alive; rank 0 falls back to eviction and keeps serving."""
    a, b = prompt(91, 40), prompt(92, 40)
    root = tmp_path / "dead"
    (root / "rank1").mkdir(parents=True)
    (root / "rank1").chmod(0o500)                     # no write permission
    h = DiskHarness(sessions=2, pool_pages=12, disk_root=root)
    try:
        h.run(a, max_tokens=4)
        h.run(b, max_tokens=4)
        got = h.run(prompt(93, 40), max_tokens=4)     # would spill a; rank 1 refuses
        h.check()
        assert not h.status()["cold_sessions"]
        assert h.engine.tier.counters["spill_errors"] >= 1
        text, stats = h.run(prompt(94, 20), max_tokens=4)
        h.check()
    finally:
        (root / "rank1").chmod(0o700)
        h.stop()


# ---- review fixes: carry/reset, cold-id reuse, disk-reply drain ----------------------------------------------

def test_tp_disk_hooks_dispatch_to_workers_not_parent_layers(tmp_path):
    """GPU-blocker regression ('cache tensors span None pages, the pool has 2048'): in TP
    mode the driver's cache.layers are unallocated shells -- the paged tensors live in the
    TP worker processes -- so geometry, page export and page import must all run through
    the worker dispatch (the library's own kv_modules/tp_cache_lookup contract), never
    the parent-side layers. Driven with a fake model whose dispatch runs the REAL
    mp_disk_* functions against a fake worker context holding actual tensors."""
    import torch

    PAGE_TOK, POOL, DIM = 16, 4, 8
    worker_a = torch.arange(POOL * PAGE_TOK * DIM, dtype=torch.float32).view(POOL, PAGE_TOK, DIM) * 1.0
    worker_b = (torch.arange(POOL * PAGE_TOK * DIM, dtype=torch.float32).view(POOL, PAGE_TOK, DIM)
                * -0.5)                                        # a second "layer" in the same worker

    class FakeLayer:
        def __init__(self, tensors):
            self._tensors = tensors

        def get_tensors(self):
            return list(self._tensors)

    class FakeModule:
        def __init__(self, layer, cache_id):
            self.tp_cache_lookup = {cache_id: layer}

    class FakeParentCache:
        layers = {}                                     # empty shells, like the TP driver

    class FakeTPModel:
        """Implements the library's dispatch contract verbatim, including cross-host
        localization: only this rank's own device runs, but the per-device suffix is
        indexed by the FULL active_devices position (dev_args[active.index(device)]) --
        the exact trap that broke the third GPU attempt on rank 1 (one-entry dev_args,
        active position 1)."""

        loaded_tp = True
        tp_cross_host = True
        active_devices = [0, 1]                     # both hosts' devices

        def __init__(self, local_ctx, tp_rank=0):
            self.local = local_ctx
            self.tp_rank = tp_rank
            self.calls = []

        def tp_localize_devices(self, devices):
            return [d for d in devices if d == self.tp_rank]   # the library's rule

        def tp_worker_dispatch_wait_multi(self, devices, fn, args, dev_args=None):
            assert isinstance(args, tuple), "dispatch args must be a tuple"
            if dev_args is not None:
                assert all(isinstance(d, tuple) for d in dev_args), \
                    "dev_args entries must be tuples (the library concatenates them)"
                assert len(dev_args) == len(devices), \
                    "dev_args must be padded to full active_devices order"
            self.calls.append((fn.__name__, args, dev_args))
            out = []
            for device in self.tp_localize_devices(devices):
                d_args = args + dev_args[devices.index(device)] if dev_args is not None else args
                out.append(fn(self.local, *d_args))
            return out

    cache = FakeParentCache()
    local = {"kv_modules": [FakeModule(FakeLayer([worker_a, worker_b]), id(cache))]}
    model = FakeTPModel(local)

    class TokStub:
        actual_vocab_size = 129280

    be = tp_engine.TPModelBackend(model, cache, TokStub(), span=64,
                                  cache_tokens=POOL * PAGE_TOK, rank=0, page_size=PAGE_TOK)
    # geometry comes from the worker, not the (empty) parent layers
    fp = be.disk_fingerprint()
    assert fp["num_pages"] == POOL and fp["page_size"] == PAGE_TOK
    assert fp["cache_tokens"] == POOL * PAGE_TOK and len(fp["layout"]) == 16
    assert any(c[0] == "mp_disk_geometry" for c in model.calls)

    # page export/import round-trips through the dispatch, parent layers never touched
    blob = be.disk_page_blob(2)
    assert any(c[0] == "mp_disk_page_out" and c[1][1] == 2 for c in model.calls)
    worker_a[2] *= 0.0
    worker_b[2] += 100.0
    be.disk_page_store(2, blob)
    assert any(c[0] == "mp_disk_page_in" for c in model.calls)
    assert torch.equal(worker_a[2], torch.arange(POOL * PAGE_TOK * DIM, dtype=torch.float32)
                       .view(POOL, PAGE_TOK, DIM)[2] * 1.0)
    assert torch.equal(worker_b[2], torch.arange(POOL * PAGE_TOK * DIM, dtype=torch.float32)
                       .view(POOL, PAGE_TOK, DIM)[2] * -0.5)
    assert not torch.equal(worker_a[2], worker_a[0])          # only page 2 was restored

    # the rank-1 shape of the same contract (the third GPU attempt's failure): rank 1
    # localizes to device 1 of [0, 1], so its suffix must sit at ACTIVE position 1 -- a
    # one-entry dev_args list raises IndexError there, exactly as the box did
    rank1 = FakeTPModel(local, tp_rank=1)
    be_r1 = tp_engine.TPModelBackend(rank1, cache, TokStub(), span=64,
                                     cache_tokens=POOL * PAGE_TOK, rank=1, page_size=PAGE_TOK)
    assert be_r1.disk_fingerprint()["num_pages"] == POOL
    part_src = be_r1.disk_page_blob(3)
    part_r1 = pickle.loads(part_src)[0]                 # plain byte records, not tensors
    assert all(set(r) == {"shape", "dtype", "data"} for r in part_r1)
    assert be_r1._tp_dev_args([part_r1]) == [(), (part_r1,)]   # padded to active order
    worker_a[3] *= 0.0
    be_r1.disk_page_store(3, part_src)                         # padded dev_args, position 1
    assert torch.equal(worker_a[3], torch.arange(POOL * PAGE_TOK * DIM, dtype=torch.float32)
                       .view(POOL, PAGE_TOK, DIM)[3] * 1.0)
    assert any(c[0] == "mp_disk_page_in" and len(c[2]) == 2 and c[2][0] == ()
               and c[2][1] != () for c in rank1.calls)
    # and the historical failure mode, verbatim: the library's loop with a one-entry
    # suffix on rank 1 (active position 1) is an IndexError
    with pytest.raises(IndexError):
        one_entry = [(part_r1,)]
        for device in [d for d in rank1.active_devices if d == rank1.tp_rank]:
            (id(cache), 3) + one_entry[rank1.active_devices.index(device)]
    # rank-1 stash unpack rides the same padded suffix
    local.setdefault("recurrent_cache", {})
    local["recurrent_cache"][9001] = [torch.full((2, 2), 7.0)]
    packed_r1 = be_r1.disk_stash_pack({"position": 5, "tp_handle": 9001})
    stash_r1 = be_r1.disk_stash_unpack(packed_r1)
    assert torch.equal(local["recurrent_cache"][stash_r1["tp_handle"]][0],
                       torch.full((2, 2), 7.0))

    # a layout change between spill and load refuses loudly (never wrong rows)
    bad = pickle.loads(blob)
    bad[0] = bad[0][:-1]
    with pytest.raises(Exception):
        be.disk_page_store(2, pickle.dumps(bad))

    # the stash unpack rides the same strict contract with REAL shapes: worker parts
    # fetched by mp_disk_pack at spill time as byte records, pushed back under a fresh
    # checkpoint handle and rebuilt to tensors worker-side (the second GPU attempt failed
    # on list dev_args here; the fourth on torch tensors crossing the pipe into shm)
    local.setdefault("recurrent_cache", {})[4242] = "stale-handle-data"
    parts = pickle.loads(blob)                             # one part per worker, records inside
    payload = {"position": 39, "window_beg": 32, "checkpoint_size": 4096,
               "_worker_parts": [list(parts[0])]}
    stash = be.disk_stash_unpack(payload)
    assert "tp_handle" in stash and stash["tp_handle"] != 4242
    restored = local["recurrent_cache"][stash["tp_handle"]]
    assert isinstance(restored[0], torch.Tensor) and isinstance(restored[1], torch.Tensor)
    assert torch.equal(restored[0], worker_a[2])           # records rebuilt to real tensors
    assert any(c[0] == "mp_disk_unpack" for c in model.calls)
    # pack/unpack round-trip through the dispatch keeps the worker data intact
    local["recurrent_cache"][4243] = [torch.ones(2, 3)]
    packed = be.disk_stash_pack({"position": 1, "tp_handle": 4243})
    assert torch.equal(tp_engine._record_to_tensor(packed["_worker_parts"][0][0]),
                       torch.ones(2, 3))

    # geometry mismatch disables the tier cleanly (the loud-log path on the real box)
    worker_c = torch.zeros(POOL + 1, PAGE_TOK, DIM)           # one tensor off by a page
    local2 = {"kv_modules": [FakeModule(FakeLayer([worker_a, worker_c]), id(cache))]}
    be2 = tp_engine.TPModelBackend(FakeTPModel(local2), cache, TokStub(), span=64,
                                   cache_tokens=POOL * PAGE_TOK, rank=0, page_size=PAGE_TOK)
    with pytest.raises(ValueError, match="page-major|span"):
        be2.disk_fingerprint()
    # and a worker with no tensors at all is the original blocker, reported as such
    be3 = tp_engine.TPModelBackend(FakeTPModel({"kv_modules": []}), cache, TokStub(), span=64,
                                   cache_tokens=POOL * PAGE_TOK, rank=0, page_size=PAGE_TOK)
    with pytest.raises(ValueError, match="no allocated cache tensors"):
        be3.disk_fingerprint()

    # non-TP backends keep reading their own in-process layers (same byte-record format,
    # no dispatch involved -- and no exllamav3 import needed anymore)
    class FakePlainModel:
        loaded_tp = False

    parent = FakeParentCache()
    parent.layers = {(0, 0): FakeLayer([worker_a])}
    be4 = tp_engine.TPModelBackend(FakePlainModel(), parent, TokStub(), span=64,
                                   cache_tokens=POOL * PAGE_TOK, rank=0, page_size=PAGE_TOK)
    assert be4.disk_fingerprint()["num_pages"] == POOL
    blob4 = be4.disk_page_blob(1)
    rec4 = pickle.loads(blob4)["(0, 0)"][0]
    assert set(rec4) == {"shape", "dtype", "data"}
    worker_a[1] *= 0.0
    be4.disk_page_store(1, blob4)
    assert torch.equal(worker_a[1], torch.arange(POOL * PAGE_TOK * DIM, dtype=torch.float32)
                       .view(POOL, PAGE_TOK, DIM)[1] * 1.0)


def test_carry_cleared_by_reset_survives_the_next_begin(tmp_path):
    """M1: evictions carried for a failed load must not outlive a prefix-cache reset --
    rank 1's drop_all forgets every record, so a later begin carrying stale evict/drop
    entries would name unknown sessions and wedge the worker. Simulate the carry (the
    exact failure path needs a rank-0 image fault mid-load), reset, then serve."""
    a = prompt(101, 40)
    h = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "carry")
    try:
        h.run(a, max_tokens=4)
        h.engine._carry_evict.extend([1, 2])          # as a failed load would have left
        h.engine._carry_drop.extend([77, 78, 79])
        h.engine._drop_images.append(5)
        asyncio.run(h.engine.reset_prefix_cache())
        assert not h.engine._carry_evict and not h.engine._carry_drop
        assert not h.engine._drop_images
        _, stats = h.run(prompt(102, 30), max_tokens=4)   # begin carries nothing stale
        assert stats["cached_tokens"] == 0
        h.check()                                     # worker alive and in sync
    finally:
        h.stop()


def test_cold_image_survives_hot_slot_reuse(tmp_path):
    """M2: after session 1 spills, a NEW conversation may reuse hot slot 1 and later spill
    itself -- monotonic image ids keep the two images and cold records distinct, and the
    ORIGINAL conversation still reloads correctly afterwards."""
    a, b, c = prompt(111, 40), prompt(112, 40), prompt(113, 40)
    on = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "reuse")
    off = DiskHarness(sessions=2, pool_pages=12, disk_root=tmp_path / "reuse-off",
                      disk_on=False)
    try:
        r_a = on.run(a, max_tokens=4)                 # session 1
        on.run(b, max_tokens=4)                       # session 2
        on.run(c, max_tokens=4)                       # spills 1 -> image 1; c takes slot 1
        st = on.status()
        assert [c0["image"] for c0 in st["cold_sessions"]] == [1]
        img_a0 = on.disk_dirs[0] / "rank0-i1.img"
        assert img_a0.exists()
        # grow the slot-1 newcomer until IT spills too: a second, distinct image
        conv = c + ids_of(on.run(c, max_tokens=4)[0]) + prompt(114, 24)
        on.run(conv, max_tokens=4)
        on.run(prompt(115, 40), max_tokens=4)         # pressure: spills the LRU (conv)
        st = on.status()
        imgs = sorted(c0["image"] for c0 in st["cold_sessions"])
        assert len(imgs) == 2 and 1 in imgs and imgs != [1, 1]
        assert img_a0.exists()                        # the old image was NOT overwritten
        assert (on.disk_dirs[0] / f"rank0-i{imgs[-1]}.img").exists()
        # the ORIGINAL conversation returns through ITS image
        a2 = a + ids_of(r_a[0]) + prompt(116, 10)
        got = on.run(a2, max_tokens=4)
        on.check()
        r_off = off.run(a, max_tokens=4)
        off.run(b, max_tokens=4)
        off.run(c, max_tokens=4)
        off.run(c + prompt(114, 24), max_tokens=4)
        off.run(prompt(115, 40), max_tokens=4)
        got_off = off.run(a + ids_of(r_off[0]) + prompt(116, 10), max_tokens=4)
        assert got[0] == got_off[0]
        assert got[1]["cached_tokens"] > 0            # reloaded, not re-prefilled
    finally:
        on.stop()
        off.stop()


def test_disk_request_drains_a_late_reply_exactly_once(tmp_path):
    """M4: a disk frame that exceeds EXL3_TP_DISK_TIMEOUT must not desync the channel --
    the engine blocks for the one owed reply and honors it; the next request then reads
    its OWN answer. Driven with a stub channel whose first request times out."""
    import types

    class StubChannel:
        def __init__(self):
            self.sent, self.timeouts = [], 0

        def request(self, msg, timeout=30.0):
            self.sent.append((dict(msg), timeout))
            self.timeouts += 1
            raise TimeoutError("simulated slow NVMe")

        def recv(self):
            return {"op": "spill_ok", "session": 3, "bytes": 123}

    eng = types.SimpleNamespace(channel=StubChannel(), log=lambda *_: None,
                                _disk_timeout=0.05)
    reply = tp_engine.TPEngine._disk_request(eng, {"op": "spill", "session": 3})
    assert reply["op"] == "spill_ok" and reply["bytes"] == 123   # the drained ack honored
    assert eng.channel.timeouts == 1 and eng.channel.sent[0][1] == 0.05
    # the next request starts from a clean channel (no stale reply left to misread)
    reply2 = tp_engine.TPEngine._disk_request(eng, {"op": "load", "session": 3})
    assert eng.channel.timeouts == 2                              # a fresh request, not a drain
