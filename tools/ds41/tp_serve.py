#!/usr/bin/env python3
"""Rank-0 entry point of the two-node OpenAI-compatible DS4.1 server (run this on Box1).

The HTTP/chat/parser layer is Kyojin's, vendored verbatim in this folder (MIT, see LICENSE.kyojin):
`ds41_serve.py` (create_app and all request handling), `ds41_chat.py` (prompt rendering),
`ds41_parse.py` (thinking / tool-call parsing), `ds41_engine.py` (engine contract + stub) and the
`serve_*` / `startup_health` helpers. This module only wires that layer to the two-node
tensor-parallel engine of `tp_engine.py`: load Box1's half of the weights exactly like
scripts/tp_ds41_generate.py does, accept the rank-1 control channel, then serve HTTP. The port
answers during the load with progress (vendored startup_health early listener): /health reports
503 "loading" with the current stage until the model, rank 1 and the warm-up are done.

Endpoints (all inherited from the vendored layer): /health, /v1/models, /metrics,
/v1/chat/completions (stream and non-stream), /v1/completions, /completion, /tokenize,
/apply-template -- plus /tp/status (this module): rank-1 liveness, request counter, context.

Box2 runs tp_rank1.py for the lifetime of this server (see tools/ds41/tp_serve_odl.sh, which
starts both ranks with the same environment scripts/run_rank.sh uses).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from aiohttp import web

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ds41_engine  # noqa: E402
import ds41_parse  # noqa: E402
import ds41_serve as serve  # noqa: E402  (vendored Kyojin tools/ds41/serve.py, verbatim)
import startup_health  # noqa: E402
import tp_engine  # noqa: E402

DEFAULT_MODEL = os.environ.get("EXL3_TP_MODEL_DIR", tp_engine.DEFAULT_MODEL_DIR)
DEFAULT_CONTROL_HOST = os.environ.get("EXLLAMA_MASTER_ADDR", "10.0.2.1")
DEFAULT_CONTROL_PORT = tp_engine.env_int("EXL3_TP_CONTROL_PORT", 29651)
DEFAULT_SAFE_CTX = tp_engine.env_int("EXL3_TP_SAFE_CTX", 750)

# The vendored server surface, re-exported for the vendored tests (they load this module).
Defaults = serve.Defaults
MODEL_CARD_SAMPLING = serve.MODEL_CARD_SAMPLING
build_defaults = serve.build_defaults
pack_sampling = serve.pack_sampling
parse_engine_opts = serve.parse_engine_opts
effort_arg = serve.effort_arg

ROUTES = {**serve.ROUTES, "/tp/status": ("GET",)}


def create_app(engine: Any, model_id: str, defaults: Defaults | None = None) -> web.Application:
    """The vendored app plus /tp/status and the TP teardown hook."""
    app = serve.create_app(engine, model_id, defaults)
    app.router.add_get("/tp/status", _tp_status)
    if hasattr(engine, "close"):
        app.on_cleanup.append(_tp_cleanup)
    return app


async def _tp_cleanup(app: web.Application) -> None:
    close = getattr(app["engine"], "close", None)
    if close is not None:
        try:
            await asyncio.get_running_loop().run_in_executor(None, close)
        except Exception:                                              # noqa: BLE001  (best effort at shutdown)
            pass


async def _tp_status(request: web.Request) -> web.Response:
    engine = request.app["engine"]
    ping = getattr(engine, "rank1_ping", None)
    rank1 = "unknown"
    if ping is not None:
        rank1 = await asyncio.get_running_loop().run_in_executor(None, ping)
    sessions = int(getattr(engine, "sessions", 1))
    out = {
        "status": "ok" if rank1 in ("ok", "busy", "unknown") else "degraded",
        "source": "exl3-tp-serve", "rank1": rank1, "proto": tp_engine.PROTO,
        "dspark": bool(getattr(engine, "dspark", False)),
        "vision": bool(getattr(engine, "supports_vision", False)),
        "requests": getattr(engine, "requests", None),
        "ctx": getattr(engine, "ctx", None), "sessions": sessions,
    }
    if sessions > 1:
        out["proto"] = tp_engine.PROTO_SESSIONS
        status = getattr(engine, "session_status", None)
        if callable(status):
            out["pool"] = status()          # slots, pages used/free, per-session detail, counters
    return web.json_response(out)


def build_tp_engine(args: argparse.Namespace) -> Any:
    """Load the rank-0 model half, wait for rank 1, verify it, and build the TPEngine."""
    rank = tp_engine.env_int("EXL3_TP_NODE_RANK", 0)
    world = tp_engine.env_int("EXL3_TP_WORLD_SIZE", 2)
    if rank != 0:
        raise SystemExit("tp_serve.py serves rank 0; rank 1 runs tp_rank1.py (tools/ds41/tp_rank1.sh)")
    startup_health.stage("model")
    backend = tp_engine.TPModelBackend.load(rank=rank, world_size=world,
                                            model_dir=str(Path(args.model).expanduser()), log=print)
    startup_health.stage("rank 1")
    channel = tp_engine.ControlChannel.listen(
        args.control_host, args.control_port,
        accept_timeout=tp_engine.env_float("EXL3_TP_ACCEPT_TIMEOUT", 1800.0), log=print)
    vision = getattr(backend, "vision", None) is not None
    # Multi-session (EXL3_TP_SESSIONS >= 2, opt-in) bumps the control protocol to 5 and
    # validates the session/pool geometry in the handshake: a stale or differently
    # configured rank 1 refuses loudly instead of desyncing page ownership mid-request
    sessions = max(1, int(getattr(args, "sessions", 0) or 0) or
                   tp_engine.env_int("EXL3_TP_SESSIONS", 1))
    env_sessions = tp_engine.env_int("EXL3_TP_SESSIONS", 1)
    if args.sessions is not None and env_sessions != sessions:
        # --sessions wins, but the launcher forwards the ENV var to the rank-1 box: warn so
        # the handshake mismatch below does not surprise anyone
        print(f"tp-serve: --sessions {sessions} overrides EXL3_TP_SESSIONS={env_sessions}; the "
              f"launcher can only forward the env var -- export it on both boxes too", flush=True)
    proto = tp_engine.PROTO if sessions < 2 else tp_engine.PROTO_SESSIONS
    page_size = int(getattr(backend, "page_size", 256) or 256)
    pool_pages = int(backend.cache_tokens) // page_size
    hello = {"op": "hello", "proto": proto, "vocab_size": backend.vocab_size,
             "dspark": bool(backend.dspark), "vision": vision}
    if proto >= tp_engine.PROTO_SESSIONS:
        hello.update({"sessions": sessions, "pool_pages": pool_pages, "page_size": page_size})
    channel.send(hello)
    reply = channel.recv()
    if reply.get("op") != "hello_ok" or int(reply.get("proto", -1)) != proto \
            or int(reply.get("rank", -1)) != 1:
        startup_health.abort(f"rank 1 handshake failed: {reply}")
        raise SystemExit(f"tp-serve: rank 1 handshake failed: {reply}")
    if not bool(reply.get("vision_capable", False)):
        startup_health.abort("rank 1 does not speak this control protocol's vision frames "
                             f"(proto {proto}); sync tools/ds41 on both boxes and restart")
        raise SystemExit("tp-serve: rank 1 lacks vision support (old tp_engine.py on Box2?) -- "
                         "restart both ranks with the same tools/ds41")
    if int(reply.get("vocab_size", -1)) != backend.vocab_size:
        startup_health.abort("rank 1 loaded a different vocabulary")
        raise SystemExit(f"tp-serve: vocab mismatch with rank 1 ({reply.get('vocab_size')} != {backend.vocab_size});"
                         " the two model folders differ")
    if bool(reply.get("dspark")) != bool(backend.dspark):
        which = "restart BOTH ranks with EXL3_TP_DSPARK set the same way"
        startup_health.abort("rank 1 DSpark state mismatch")
        raise SystemExit(f"tp-serve: DSpark mismatch with rank 1 (rank 0 {bool(backend.dspark)}, "
                         f"rank 1 {bool(reply.get('dspark'))}); {which}")
    if proto >= tp_engine.PROTO_SESSIONS:
        if int(reply.get("sessions", -1)) != sessions:
            startup_health.abort("rank 1 session count mismatch")
            raise SystemExit(
                f"tp-serve: rank 1 serves {reply.get('sessions')} sessions, this rank "
                f"{sessions}. The rank-1 box must run with the same count: export "
                f"EXL3_TP_SESSIONS={sessions} there (tp_serve_odl.sh or exl3ds4.sh "
                f"forward it) and restart both ranks. Note --sessions only configures rank 0; "
                f"the env var is what reaches rank 1.")
        if int(reply.get("pool_pages", -1)) != pool_pages:
            startup_health.abort("rank 1 pool geometry mismatch")
            raise SystemExit(
                f"tp-serve: rank 1's pool holds {reply.get('pool_pages')} pages, this rank's "
                f"{pool_pages}; EXL3_TP_CACHE_TOKENS must match on both ranks")
    print(f"tp-serve: rank 1 connected and verified (vocab {backend.vocab_size}, "
          f"dspark {'on' if backend.dspark else 'off'}, proto {proto})", flush=True)
    if sessions > 1:
        print(f"tp-serve: multi-session serving ON ({sessions} slots over the shared "
              f"{pool_pages}-page pool, dynamic noncontiguous pages; per-request \"session\" "
              f"ids 1..{sessions}; /tp/status reports the pool)", flush=True)
    if vision:
        print("tp-serve: vision on (rank 0 tower + aligner loaded; image requests ship their rows "
              "to rank 1 per request; image prompts bypass the prefix cache)", flush=True)
    else:
        print("tp-serve: vision off (image requests are refused with 400; EXL3_TP_VISION=0, a "
              "degraded load or a text-only model -- see the load log)", flush=True)
    if backend.dspark:
        which = "greedy and sampled requests" if tp_engine.SPEC_SAMPLED else "greedy requests (temperature 0)"
        print(f"tp-serve: DSpark speculative decoding on (block {backend.dspark_block}, max_history "
              f"{tp_engine.MAX_HISTORY}); {which} draft by default -- "
              f"send \"speculative\": false per request to opt out, \"spec_k\": 1..{backend.dspark_block} to cap",
              flush=True)
    tokenizer = ds41_engine.HFTokenizer(Path(args.model).expanduser() / "tokenizer.json")
    ctx = int(args.max_ctx) if args.max_ctx else DEFAULT_SAFE_CTX
    ctx = max(16, min(ctx, backend.cache_tokens - 2))
    return tp_engine.TPEngine(backend, channel, tokenizer, ctx=ctx,
                              eos_id=tp_engine.eos_id_of(args.model, tokenizer), seed=args.seed,
                              sessions=sessions,
                              log=lambda line: print(line, flush=True))   # the launchers log to a file


def _prefix_cache_line(engine) -> str:
    """The startup log's prefix-cache sentence. Single-session engines report the one
    PrefixCache; multi-session engines (mgr) hold one PrefixCache NAMESPACE PER SESSION,
    so 'engine.prefix is None' there does NOT mean caching is off -- it means per-session
    caching is on (governed by EXL3_TP_PREFIX_SLOTS/_MIN, not EXL3_TP_PREFIX_CACHE alone)."""
    px = getattr(engine, "prefix", None)
    mgr = getattr(engine, "mgr", None)
    if mgr is not None:
        return ("on, per session (" + f"{mgr.prefix_slots} checkpoints of the recurrent state per slot, "
                f"prompts from {mgr.min_tokens} tokens, end-of-reply checkpoints "
                f"{'on' if mgr.save_end else 'off'}; per request \"cache_prompt\": false opts out)")
    if px is not None:
        return (f"on ({px.slots} checkpoints of the recurrent state, prompts from {px.min_tokens} tokens, "
                f"end-of-reply checkpoints {'on' if px.save_end else 'off'}; per request "
                f"\"cache_prompt\": false opts out)")
    return "off (EXL3_TP_PREFIX_CACHE=0): every request re-prefills"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model folder (weights, config.json, tokenizer.json)")
    parser.add_argument("--model-id", default="deepseek-v4.1-flash", help="id reported by /v1/models and echoed in replies")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind address (default 0.0.0.0, the LAN head node)")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    parser.add_argument("-c", "--max-ctx", type=int, default=None,
                        help=f"serving context in tokens (default {DEFAULT_SAFE_CTX}, the validated safe limit; "
                             "prompt + reply tokens must stay within it until the DSV4 wshift readback is fixed)")
    parser.add_argument("--loader", default="tp",
                        help="'tp' (default: the two-node engine), 'stub' (canned replies, no GPU, single host, "
                             "for HTTP-layer checks) or module:function")
    parser.add_argument("--control-host", default=DEFAULT_CONTROL_HOST,
                        help="address rank 1 connects to (default $EXLLAMA_MASTER_ADDR or 10.0.2.1)")
    parser.add_argument("--control-port", type=int, default=DEFAULT_CONTROL_PORT,
                        help="rank-0 control channel port (default $EXL3_TP_CONTROL_PORT or 29651)")
    parser.add_argument("--seed", type=int, default=None, help="sampling generator seed (temperature > 0 requests)")
    parser.add_argument("--sessions", type=int, default=None,
                        help="cached conversation slots (default $EXL3_TP_SESSIONS or 1; >= 2 enables the "
                             "multi-session engine: dynamic shared-pool pages, proto 5)")
    parser.add_argument("--engine-opt", action="append", default=[], metavar="KEY=VALUE",
                        help="extra keyword argument for a module:function loader (JSON value or plain string); repeatable")
    parser.add_argument("--default-temperature", type=float, default=None,
                        help="used when a request omits temperature (default: generation_config.json, else 1.0)")
    parser.add_argument("--default-top-p", type=float, default=None,
                        help="used when a request omits top_p (default: generation_config.json, else 1.0)")
    parser.add_argument("--default-max-tokens", type=int, default=None,
                        help="reply budget when a request omits max_tokens, thinking included")
    parser.add_argument("--default-reasoning-effort", type=serve.effort_arg, default=None,
                        help="thinking effort when a request omits it: minimal, low, medium, high, xhigh, max or an integer 1..100")
    parser.add_argument("--default-thinking", choices=("on", "off"), default="on",
                        help="thinking mode when a request does not choose (default: on)")
    args = parser.parse_args()

    engine_opts = serve.parse_engine_opts(args.engine_opt)
    defaults = serve.build_defaults(args)

    if args.loader == "tp":
        startup_health.check_model_dir("tp-serve", args.model, ("config.json", "tokenizer.json"))
        startup_health.start("tp-serve", ["model", "rank 1", "warm-up"], args.host, args.port, routes=ROUTES)
        try:
            engine = build_tp_engine(args)
        except BaseException as exc:                                  # noqa: BLE001
            startup_health.abort(f"{type(exc).__name__}: {exc}")
            raise
    else:
        startup_health.check_model_dir("tp-serve", args.model, ("config.json", "tokenizer.json"))
        startup_health.start("tp-serve", ["model", "warm-up"], args.host, args.port, routes=ROUTES)
        try:
            startup_health.stage("model")
            engine = ds41_engine.load_model(str(Path(args.model).expanduser()),
                                            int(args.max_ctx or 65536), loader=args.loader, **engine_opts)
        except BaseException as exc:                                  # noqa: BLE001
            startup_health.abort(f"{type(exc).__name__}: {exc}")
            raise

    # One tiny request end to end (both ranks) before serving; EXL3_SERVE_WARMUP=0 skips it.
    if hasattr(engine, "warmup") and os.environ.get("EXL3_SERVE_WARMUP", "1") != "0":
        startup_health.stage("warm-up")
        print(f"tp-serve: warm-up {engine.warmup():.1f} s", flush=True)

    app = create_app(engine, args.model_id, defaults)
    reqlog = app["reqlog"]
    print(f"tp-serve: {reqlog.describe()}", flush=True)
    print(f"tp-serve: {app['ping'].describe()}", flush=True)
    print(f"tp-serve: defaults temperature {defaults.temperature}, top_p {defaults.top_p}, thinking "
          f"{'on' if defaults.thinking else 'off'}, effort {defaults.reasoning_effort}", flush=True)
    be = getattr(engine, "backend", None)
    print(f"tp-serve: ctx {getattr(engine, 'ctx', '?')} tokens"
          + (f" (cache {be.cache_tokens}, prefill chunk {be.span} rows, block tables sized to the cache)"
             if be is not None else "")
          + f", {int(getattr(engine, 'sessions', 1))} session slot(s), one in-flight request"
          + (" (multi-session: GPU work serialized, queueing in the server)" if int(getattr(engine, "sessions", 1)) > 1 else ""),
          flush=True)
    if be is not None:
        print("tp-serve: prefix cache " + _prefix_cache_line(engine), flush=True)
    reqlog.log("serve_start", model=Path(args.model).name, model_id=args.model_id, pid=os.getpid(),
               ctx=getattr(engine, "ctx", None), sessions=int(getattr(engine, "sessions", 1)),
               heartbeat_s=reqlog.heartbeat_s, sse_ping_s=app["ping"].interval_s or None, loader=args.loader)
    sock = startup_health.finish()          # the early listener's socket: no refused connections
    if sock is not None:
        web.run_app(app, sock=sock)
    else:
        web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
