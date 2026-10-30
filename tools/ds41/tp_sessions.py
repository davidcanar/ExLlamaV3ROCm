"""Stage-1 multi-session support for the two-node DS4.1 TP server (EXL3_TP_SESSIONS).

Rank-0-side, pure host bookkeeping: which conversations are cached, which pool pages each
one owns, and who to evict when a new request does not fit. Everything here is driven by
token ids and page numbers only -- no torch, no sockets -- so the whole policy is CPU
testable on its own. tp_engine.py owns the wire; this module owns the decisions:

- PageAllocator: dynamic, noncontiguous allocation of physical pages from the one shared
  pool (EXL3_TP_CACHE_TOKENS // PAGE_SIZE pages). No fixed per-session cap: a session
  grows while the total pool fits, page by page, wherever free pages are.
- Session: one cached conversation: its own PrefixCache namespace (checkpoints), its page
  list (logical page i -> pages[i]), the length of the last request's prompt (the header
  position the next turn's template may rewrite) and an LRU tick.
- SessionManager: returning-session recognition, slot/page pressure handling and
  whole-session LRU eviction.

Returning-session recognition (OpenCode sends no session id): a prompt returns to its
session when it still contains that session's LAST request's prompt whole -- the common
prefix of the session's committed sequence and the new prompt reaches last_prompt_len - 1.
The -1 tolerates the existing generation-header 1-token behavior (the next turn's template
may rewrite the last token of the previous prompt before echoing its reply; that position,
len(prompt) - 1, is exactly where the single-session prefix cache restores). A prompt that
shared only the system prompt diverges INSIDE the previous prompt, below that bound, so it
can never take an existing session over. An explicit per-request "session" id overrides the
search (validated; wrong type/range is a clean error).

use_cache ("cache_prompt": false) skips checkpoint restore/save for one request but never
changes session identity: the request still runs in the conversation it belongs to (its
writes must land on that session's rows), exactly as the single-session engine still
updates its record on no-cache requests. Image and warm-up requests are ephemeral: they
allocate a slot and pages for one request, never match, and free themselves at end.

Stage 2 (tp_disk.py) layers NVMe whole-session spill/restore on top of the same records;
this module stays importable and testable without it.
"""
from __future__ import annotations

from array import array
from typing import Any, Iterable

from tp_engine import PrefixCache, _common_prefix, env_int, env_truthy


class PoolOverflow(RuntimeError):
    """A request needs more pages than the shared pool can hold (clean 400, never dispatched)."""


class PageAllocator:
    """Physical pages of the one shared DSA pool, handed out noncontiguously.

    Deterministic on purpose: alloc() always returns the lowest free pages. Rank 0 decides
    and ships the full page list in the begin frame; rank 1 validates it against its own
    records, so the determinism is a defense in depth, not a correctness requirement.
    """

    def __init__(self, pool_pages: int):
        self.pool_pages = int(pool_pages)
        self._free = sorted(range(self.pool_pages), reverse=True)   # pop() = lowest-first
        self._used: set[int] = set()
        self.counters = {"page_allocs": 0, "page_frees": 0, "peak_pages": 0}

    def alloc(self, n: int) -> list[int]:
        n = int(n)
        if n < 0:
            raise ValueError(f"page allocation of {n} pages")
        if n > len(self._free):
            raise PoolOverflow(f"need {n} more pages, {len(self._free)} free of "
                               f"{self.pool_pages} (EXL3_TP_CACHE_TOKENS pool)")
        pages = sorted(self._free[-n:]) if n else []
        del self._free[-n:]
        self._used.update(pages)
        self.counters["page_allocs"] += n
        self.counters["peak_pages"] = max(self.counters["peak_pages"], len(self._used))
        return pages

    def release(self, pages: Iterable[int]) -> int:
        out = 0
        for p in pages:
            p = int(p)
            if p not in self._used:
                raise ValueError(f"double free of page {p}")
            self._used.discard(p)
            self._free.append(p)
            out += 1
        self._free.sort(reverse=True)
        self.counters["page_frees"] += out
        return out

    @property
    def used(self) -> int:
        return len(self._used)

    @property
    def free(self) -> int:
        return len(self._free)


class Session:
    """One cached conversation slot: its checkpoints, pages and LRU bookkeeping."""

    __slots__ = ("sid", "px", "pages", "tick", "active", "ephemeral", "last_prompt_len")

    def __init__(self, sid: int, px: PrefixCache):
        self.sid = int(sid)
        self.px = px                     # this session's checkpoint namespace (kv_ids = committed seq)
        self.pages: list[int] = []       # logical page i -> physical page pages[i]
        self.tick = 0
        self.active = False
        self.ephemeral = False           # image/warm-up requests: freed at end, never matched
        self.last_prompt_len = 0         # h: position h-1 is the generation header the next turn rewrites

    @property
    def seq(self) -> array:
        """Committed sequence: the tokens at positions [0, E) of this session's rows."""
        return self.px.kv_ids

    @property
    def committed(self) -> int:
        return len(self.px.kv_ids)


class ColdSession:
    """A whole session parked on the NVMe tier (stage 2, tp_disk): the pool pages and both
    ranks' checkpoint stashes live in each box's local image (`img`, a process-unique
    monotonic id -- hot slot ids are reused freely, image ids never are, so a reused slot
    can never overwrite an older cold conversation's image); what stays in RAM is cheap
    and needed for matching -- the committed token sequence, the checkpoint keys/positions,
    the (small, rank-0-only) drafter windows and the header bookkeeping."""

    __slots__ = ("sid", "img", "seq", "ckpt_meta", "drafts", "last_prompt_len", "pages_n",
                 "tick", "bytes", "bytes_peer")

    def __init__(self, sid: int, img: int, seq, ckpt_meta: list, drafts: dict,
                 last_prompt_len: int, pages_n: int, tick: int):
        self.sid = int(sid)              # the hot slot it was spilled from (informational)
        self.img = int(img)              # the image id both ranks' files are keyed by
        self.seq = seq                    # array snapshot of the committed sequence
        self.ckpt_meta = ckpt_meta        # [(key, pos)] in checkpoint order
        self.drafts = drafts              # {key: drafter window} (rank 0 keeps these in RAM)
        self.last_prompt_len = int(last_prompt_len)
        self.pages_n = int(pages_n)
        self.tick = int(tick)
        self.bytes = 0                    # rank 0's image size
        self.bytes_peer = 0               # rank 1's image size (from spill_ok)

    @property
    def committed(self) -> int:
        return len(self.seq)


class SessionManager:
    """EXL3_TP_SESSIONS > 1: S conversation slots over one shared page pool.

    The engine routes every request through plan()/commit() on the model thread (GPU stays
    serialized: at most one session is active, so "inactive" is every other session and
    LRU eviction is always safe). commit() returns exactly what the begin frame must tell
    rank 1: the session, its page list after this request's allocation, the checkpoints to
    drop (the request's own prefix-slot evictions plus every evicted session's), and the
    whole sessions to evict.
    """

    def __init__(self, *, slots: int, pool_pages: int, page_size: int,
                 prefix_slots: int = 2, min_tokens: int = 256, save_end: bool = True,
                 min_match: int = 1):
        self.slots = max(1, int(slots))
        self.page_size = int(page_size)
        self.pool = PageAllocator(pool_pages)
        self.prefix_slots = max(1, int(prefix_slots))
        self.min_tokens = int(min_tokens)
        self.save_end = bool(save_end)
        self.min_match = max(1, int(min_match))
        self.sessions: dict[int, Session] = {}
        self.cold: dict[int, ColdSession] = {}   # stage 2: parked sessions, keyed by image id
        self._tick = 0
        self._key = 0                     # globally unique checkpoint keys across sessions
        self._img = 0                     # monotonic image ids (never reused: a reused hot
        #                                     slot can never overwrite an older cold image)
        self.counters = {"requests": 0, "matched": 0, "new_sessions": 0, "session_evictions": 0,
                         "page_pressure_evictions": 0, "ephemeral_freed": 0, "spill_fallbacks": 0,
                         "spill_restore_fallbacks": 0, "spills": 0, "cold_loads": 0}

    def next_image_id(self) -> int:
        self._img += 1
        return self._img

    # -- checkpoint keys: unique across every session (rank 1 namespaces by session, but
    #    the begin/end frames carry bare ints; global uniqueness keeps them unambiguous)
    def next_key(self) -> int:
        self._key += 1
        return self._key

    def _touch(self, s: Session) -> None:
        self._tick += 1
        s.tick = self._tick

    # -- returning-session recognition ------------------------------------------------------
    def match(self, ids: array, *, skip_cold: frozenset = frozenset()) -> tuple[Any, int]:
        """(session or cold record, effective match length) for this prompt -- longest common
        prefix wins, ties go to the most recently used. Hot sessions and cold (spilled)
        sessions compete on equal terms; None means a new conversation.

        `skip_cold` bounds cold-reload retries: image ids there were already attempted (and
        failed) this request and are never matched again, so one request tries each matching
        cold conversation at most once and then falls back to a fresh session.

        A prompt returns to its session when it still contains that session's LAST request's
        prompt whole: common_prefix(seq, ids) >= last_prompt_len - 1, where the -1 tolerates
        the generation-header rewrite (the next turn's template may replace the last token
        of the previous prompt before echoing its reply -- exactly the checkpoint position
        the single-session cache restores at, len(prompt) - 1). A prompt that shared only
        the system prompt would diverge INSIDE the previous prompt, below last_prompt_len-1,
        so it can never take an existing session over.

        Ephemeral sessions never match (their ids are unique per request anyway); an active
        session cannot (the GPU is serialized and this one request is the only candidate)."""
        best = None
        best_len = 0
        best_tick = -1
        for pool in (self.sessions, self.cold):
            for s in pool.values():
                if getattr(s, "ephemeral", False) or getattr(s, "active", False):
                    continue
                if pool is self.cold and s.img in skip_cold:
                    continue
                seq = s.seq
                if not len(seq):
                    continue
                m = _common_prefix(seq, ids)
                eff = m if m >= max(1, s.last_prompt_len - 1) else 0
                if eff < max(self.min_match, 1):
                    continue
                if eff > best_len or (eff == best_len and s.tick > best_tick):
                    best, best_len, best_tick = s, eff, s.tick
        return best, best_len

    # -- slots and pages ---------------------------------------------------------------------
    def _free_sid(self, wanted: int | None = None, freed: set[int] = frozenset()) -> int | None:
        """A free slot id, or None. `freed` counts sessions a plan has virtually evicted
        (plan() is pure; the victims are still in self.sessions until commit())."""
        if wanted is not None:
            return wanted if (wanted not in self.sessions or wanted in freed) else None
        for sid in range(1, self.slots + 1):
            if sid not in self.sessions or sid in freed:
                return sid
        return None

    def _lru_victim(self, *, exclude: set[int] = frozenset()) -> Session | None:
        pool = [s for s in self.sessions.values() if not s.active and s.sid not in exclude]
        return min(pool, key=lambda s: s.tick) if pool else None

    def evict(self, s: Session, *, reason: str) -> list:
        """Drop a whole session: its checkpoints (the caller drops the rank-0 stashes and
        ships the keys + sid to rank 1) and every page it owns. Returns the Checkpoints so
        the engine can drop_stash() them on rank 0."""
        ckpts = list(s.px.ckpts)
        s.px.ckpts.clear()
        s.px.kv_ids = array("i")
        if s.pages:
            self.pool.release(s.pages)
        s.pages = []
        self.sessions.pop(s.sid, None)
        self.counters["session_evictions"] += 1
        if reason == "pressure":
            self.counters["page_pressure_evictions"] += 1
        return ckpts

    def clear(self) -> list:
        """Forget every session, hot or cold (cache_salt reset / rank-0 bookkeeping
        failure): returns all Checkpoints so the engine can drop the rank-0 stashes; rank 1
        drops its copies and records at the next begin ("drop_all"), and the caller drops
        the images."""
        out: list = []
        for s in list(self.sessions.values()):
            out += list(s.px.ckpts)
            s.px.ckpts.clear()
            s.px.kv_ids = array("i")
            if s.pages:
                self.pool.release(s.pages)
            s.pages = []
        self.sessions.clear()
        self.cold.clear()
        return out

    # -- the per-request routing (model thread; rank 0 decides, rank 1 obeys) ----------------
    #
    # Two phases on purpose: plan() is PURE (match + deterministic victim choice + page math;
    # it mutates nothing), so the engine can run everything that may still fail cleanly --
    # vision encoding, the imgmeta shipment -- between plan and commit. commit() then applies
    # exactly the planned evictions and growth; by then nothing between it and the begin frame
    # can fail, so rank 0's records and rank 1's can never quietly diverge.
    def plan(self, ids: array, *, need_tokens: int, explicit_sid: int | None = None,
             ephemeral: bool = False, skip_cold: frozenset = frozenset(),
             cold_ok: bool = True) -> dict[str, Any]:
        self.counters["requests"] += 1
        n_pages = -(-int(need_tokens) // self.page_size)
        if n_pages > self.pool.pool_pages:
            raise PoolOverflow(
                f"a single request needs {n_pages} pages but the whole pool holds "
                f"{self.pool.pool_pages} (EXL3_TP_CACHE_TOKENS); refused before any dispatch")
        session = None
        is_new = False
        if ephemeral:
            pass                                        # never matched, never restored
        elif explicit_sid is not None:
            if not isinstance(explicit_sid, int) or isinstance(explicit_sid, bool) \
                    or not 1 <= explicit_sid <= self.slots:
                raise ValueError(f"session id must be an integer in 1..{self.slots}, got "
                                 f"{explicit_sid!r}")
            # "session": N pins a HOT slot. The cold registry is keyed by image id (slot
            # ids are reused after spills), so an explicit sid must NEVER index it -- an
            # arbitrary cold image would come back as the wrong conversation. If a cold
            # record was spilled FROM this slot, refuse loudly: the supported path for a
            # returning conversation is no override at all (auto-match reloads it).
            session = self.sessions.get(explicit_sid)
            if session is None:
                spilled_here = [c for c in self.cold.values() if c.sid == explicit_sid]
                if spilled_here:
                    c = spilled_here[0]
                    raise ValueError(
                        f"session {explicit_sid} is spilled to the disk tier (image {c.img}); "
                        f"send the request WITHOUT \"session\" and the conversation reloads "
                        f"automatically (explicit ids pin hot slots only)")
        else:
            session, _eff = self.match(ids, skip_cold=skip_cold if cold_ok else frozenset())
            if not cold_ok and isinstance(session, ColdSession):
                session = None                          # reload budget spent: fresh session

        load_cold = None
        if isinstance(session, ColdSession):
            load_cold, session = session, None

        victims: list[Session] = []
        picked: set[int] = set()          # virtually-evicted sids (plan is pure; commit applies)
        if session is None and load_cold is None:
            wanted = explicit_sid if not ephemeral and explicit_sid is not None else None
            free_sid = self._free_sid(wanted, picked)
            while free_sid is None:
                # evict LRU sessions (never the spilled; never the active one -- there is
                # none but ours) until the wanted slot is virtually free
                victim = self._lru_victim(exclude=picked | ({wanted} if wanted else set()))
                if victim is None:
                    raise PoolOverflow(f"all {self.slots} session slots are busy; cannot "
                                       f"start session {wanted!r} (retry, or raise "
                                       f"EXL3_TP_SESSIONS)")
                victims.append(victim)
                picked.add(victim.sid)
                free_sid = self._free_sid(wanted, picked)
            session = Session(free_sid, PrefixCache(slots=self.prefix_slots,
                                                    min_tokens=self.min_tokens,
                                                    save_end=self.save_end))
            session.ephemeral = bool(ephemeral)
            is_new = True
        elif load_cold is not None:
            # the load re-heats the cold session under ANY free slot (the original sid may
            # have been reused after the spill; the load frame carries the image sid
            # separately), with pages wherever the pool has them -- at least as many as it
            # had, plus this request's growth
            free_sid = self._free_sid(None, picked)
            while free_sid is None:
                victim = self._lru_victim(exclude=picked)
                if victim is None:
                    raise PoolOverflow(f"all {self.slots} session slots are busy; cannot "
                                       f"reload session {load_cold.sid} from the disk tier")
                victims.append(victim)
                picked.add(victim.sid)
                free_sid = self._free_sid(None, picked)

        # page growth for this request (prompt + bounded generation): deterministically pick
        # the LRU victims that free enough pages, so commit()/begin_load cannot run short
        have = len(session.pages) if session is not None else 0
        if load_cold is not None:
            have = 0
            n_pages = max(n_pages, load_cold.pages_n)
        free = self.pool.free + sum(len(v.pages) for v in victims)
        if n_pages > have:
            need_more = n_pages - have
            guard = set() if (is_new or load_cold is not None) else {session.sid}
            while free < need_more:
                victim = self._lru_victim(exclude=picked | guard)
                if victim is None:
                    raise PoolOverflow(
                        f"the request needs {n_pages} pages, {free} free after evictions; the "
                        f"pool holds {self.pool.pool_pages} (EXL3_TP_CACHE_TOKENS)")
                victims.append(victim)
                picked.add(victim.sid)
                free += len(victim.pages)
        return {"session": session, "is_new": is_new, "victims": victims, "load_cold": load_cold,
                "n_pages": n_pages, "need_tokens": int(need_tokens), "ids": ids,
                "explicit_sid": explicit_sid, "ephemeral": bool(ephemeral)}

    def commit(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Apply a plan(): evict its victims (whole sessions, checkpoint lists returned for
        the rank-0 stash drops and the begin frame), then create/grow the session. Nothing
        after this point may fail before the begin frame is sent."""
        session = plan["session"]
        evicted: list[tuple[int, list]] = []
        for victim in plan["victims"]:
            ckpts = self.evict(victim, reason="pressure")
            evicted.append((victim.sid, ckpts))
        if plan["is_new"]:
            self.sessions[session.sid] = session
            self.counters["new_sessions"] += 1
        n_pages = plan["n_pages"]
        have = len(session.pages)
        if n_pages > have:
            session.pages += self.pool.alloc(n_pages - have)   # plan proved the pages exist
        self._touch(session)
        session.active = True
        if not plan["is_new"] and not session.ephemeral:
            self.counters["matched"] += 1
        return {"session": session, "is_new": plan["is_new"], "evicted": evicted}

    def route(self, ids: array, *, need_tokens: int, explicit_sid: int | None = None,
              ephemeral: bool = False) -> dict[str, Any]:
        """plan() + commit() in one step (what a request without fallible pre-work uses)."""
        return self.commit(self.plan(ids, need_tokens=need_tokens, explicit_sid=explicit_sid,
                                     ephemeral=ephemeral))

    # -- stage 2: the cold (disk) tier -------------------------------------------------------
    def spill(self, s: Session, img: int) -> tuple[ColdSession, list]:
        """Park a whole hot session on the disk tier under image id `img` (allocated by the
        caller via next_image_id() BEFORE the frames were sent): its pages return to the
        pool and its checkpoints (returned, so the engine drop_stash()es the rank-0 copies
        after the image is safe) become metadata. The caller writes the images FIRST --
        this runs only once both ranks' writes succeeded."""
        ckpts = list(s.px.ckpts)
        cold = ColdSession(s.sid, img, array("i", s.seq), [(c.key, c.pos) for c in ckpts],
                           {c.key: c.draft for c in ckpts}, s.last_prompt_len, len(s.pages),
                           s.tick)
        s.px.ckpts.clear()
        s.px.kv_ids = array("i")
        if s.pages:
            self.pool.release(s.pages)
        s.pages = []
        self.sessions.pop(s.sid, None)
        self.cold[img] = cold              # keyed by image id: hot slot reuse never collides
        self.counters["spills"] += 1
        return cold, ckpts

    def spill_bytes_note(self, cold: ColdSession, mine: int, peer: int) -> None:
        cold.bytes, cold.bytes_peer = int(mine), int(peer)

    def discard_cold(self, img: int) -> ColdSession | None:
        """Give up on a cold session (image corrupt/missing, load failed, salt reset): the
        caller drops the image; the conversation re-prefills next time. Counted even when
        the record is already gone (begin_load pops it before the image is validated)."""
        cold = self.cold.pop(int(img), None)
        self.counters["spill_restore_fallbacks"] += 1
        return cold

    def apply_victims(self, plan: dict) -> list[tuple[int, list]]:
        """Evict (and only evict) the plan's victims -- the load path applies them before
        the disk load, the normal path defers them to commit()."""
        evicted: list[tuple[int, list]] = []
        for victim in plan.get("victims", ()):
            ckpts = self.evict(victim, reason="pressure")
            evicted.append((victim.sid, ckpts))
        plan["victims"] = []
        return evicted

    def begin_load(self, cold: ColdSession, n_pages: int) -> Session:
        """Re-heating, step 1: allocate the pages the image fills (the engine restores the
        tensors and stashes around this) and create the hot record under the cold session's
        slot. Call only after apply_victims made room."""
        sid = cold.sid if cold.sid not in self.sessions else None
        if sid is None:
            for cand in range(1, self.slots + 1):
                if cand not in self.sessions:
                    sid = cand
                    break
        if sid is None:
            raise PoolOverflow(f"no free slot to reload session {cold.sid} from the disk tier")
        pages = self.pool.alloc(max(int(n_pages), 1))
        session = Session(sid, PrefixCache(slots=self.prefix_slots, min_tokens=self.min_tokens,
                                           save_end=self.save_end))
        session.pages = list(pages)
        session.last_prompt_len = cold.last_prompt_len
        self.sessions[sid] = session
        self.cold.pop(cold.img, None)
        self._touch(session)
        return session

    def abort_load(self, session: Session) -> None:
        """A failed load: the half-restored session never existed (no checkpoints yet -- the
        engine drops the partially unpacked stashes itself, mirrored by rank 1's cleanup)."""
        if session.pages:
            self.pool.release(session.pages)
        session.pages = []
        self.sessions.pop(session.sid, None)

    def reattach(self, session: Session, cold: ColdSession) -> list:
        """Re-heating, step 2: both ranks restored the image; rebuild the hot record's
        committed sequence and checkpoints from the cold metadata (stash objects arrive via
        the engine). Returns the Checkpoints it created (the engine owns the stash handles;
        the cold_loads counter ticks only when the whole load succeeds)."""
        from tp_engine import Checkpoint
        session.px.kv_ids = array("i", cold.seq)
        out = []
        for key, pos in cold.ckpt_meta:
            out.append(Checkpoint(key, pos, array("i", cold.seq[:pos]), None, 0,
                                  cold.drafts.get(key)))
        # the stash objects are attached by the engine right after this (it holds them)
        session.px.ckpts = out
        return out

    # -- per-request checkpoint accounting (the engine calls these; see tp_engine) -----------
    def reserve(self, session: Session, protect=None) -> tuple[int, list]:
        """A fresh globally-unique checkpoint key, and the checkpoints this session's own
        prefix slots evicted (LRU) to make room."""
        _, evicted = session.px.reserve(protect=protect)
        return self.next_key(), evicted

    def finish(self, session: Session, *, prompt_len: int | None) -> None:
        """Settle the session record after a request (the engine already updated px).
        `prompt_len` is None for failed requests: their prompt never committed, so the
        header position (which the next turn's match tolerates) stays where it was.

        Pages are grow-only within a session's life (documented Stage-1 policy): rows past
        the committed length are dead and get rewritten by the next turn of the session,
        and whole-page freedom comes back with whole-session eviction."""
        session.active = False
        if prompt_len is not None:
            session.last_prompt_len = int(prompt_len)

    def release_ephemeral(self, session: Session) -> list:
        """Drop an ephemeral session (image/warm-up requests can never match: unique ids).
        Returns its Checkpoints (there are none: ephemeral sessions never save)."""
        assert session.ephemeral, "release_ephemeral on a persistent session"
        self.counters["ephemeral_freed"] += 1
        return self.evict(session, reason="ephemeral")

    # -- status --------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        sessions = []
        for s in sorted(self.sessions.values(), key=lambda x: x.sid):
            sessions.append({
                "session": s.sid, "committed": s.committed, "pages": len(s.pages),
                "checkpoints": sorted(c.pos for c in s.px.ckpts),
                "last_prompt_len": s.last_prompt_len, "active": s.active,
                "ephemeral": s.ephemeral, "spilled": False, "lru_tick": s.tick,
            })
        cold = []
        for c in sorted(self.cold.values(), key=lambda x: x.img):
            cold.append({
                "image": c.img, "session": c.sid, "committed": c.committed, "pages": c.pages_n,
                "checkpoints": sorted(p for _, p in c.ckpt_meta), "spilled": True,
                "bytes": c.bytes, "bytes_peer": c.bytes_peer, "lru_tick": c.tick,
            })
        return {
            "slots": self.slots, "pool_pages": self.pool.pool_pages,
            "pages_used": self.pool.used, "pages_free": self.pool.free,
            "sessions": sessions, "cold_sessions": cold,
            "counters": {**self.counters, **self.pool.counters},
        }

    @classmethod
    def from_env(cls, *, pool_pages: int, page_size: int, log=None) -> "SessionManager | None":
        """EXL3_TP_SESSIONS=1 (default) keeps the single-session engine byte-compatible;
        S >= 2 switches rank 0 to this manager (and the control protocol to proto 5)."""
        sessions = env_int("EXL3_TP_SESSIONS", 1)
        if sessions < 2:
            return None
        mgr = cls(slots=sessions, pool_pages=pool_pages, page_size=page_size,
                  prefix_slots=env_int("EXL3_TP_PREFIX_SLOTS", 2),
                  min_tokens=env_int("EXL3_TP_PREFIX_MIN", 256),
                  save_end=env_truthy("EXL3_TP_PREFIX_END", True),
                  min_match=env_int("EXL3_TP_SESSION_MIN_MATCH", 1))
        if log is not None:
            log(f"tp: sessions {mgr.slots} over a shared {pool_pages}-page pool "
                f"({pool_pages * page_size} tokens); dynamic noncontiguous pages, "
                f"LRU whole-session eviction")
        return mgr


def spill_slack(spec: bool, dspark_block: int = 5) -> int:
    """Rows a request may write past prompt + max_tokens: the first forward feeds the last
    prompt token (+1), a spec round settles only after writing up to k+1 verify rows, and
    the decode loop samples the stop token before feeding it (+1)."""
    return 2 + (int(dspark_block) if spec else 0)
