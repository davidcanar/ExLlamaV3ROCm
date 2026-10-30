#!/usr/bin/env python3
"""One-shot vendoring: copy the MIT-licensed Kyojin HTTP/chat/parser/test layer into
this tools/ds41 folder with attribution headers. Idempotent (overwrites).

Run on the rank-0 box:  <exl3ds4>/.venv/bin/python vendor_kyojin.py
Source:                  <exl3ds4>/src/kyojin  (Yamz-Labs/kyojin @ e2f3756, release 1.8.2;
                         cloned by setup.sh; override with EXL3_KYOJIN_SRC)
Root defaults to two directories above this file; override with EXL3DS4_ROOT.

ds41_serve.py (from Kyojin serve.py) additionally carries three small, clearly marked
LOCAL blocks that forward per-request fields of the two-node engines to generate():
`speculative` / `spec_k` (DSpark) and `session` (multi-session slot pinning, tp_sessions);
everything else stays byte-identical to upstream.
"""
import os
import shutil
from pathlib import Path

_ROOT = os.environ.get(
    "EXL3DS4_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
SRC = Path(os.environ.get("EXL3_KYOJIN_SRC", os.path.join(_ROOT, "src/kyojin")))
DST = Path(os.environ.get("EXL3_DS41_DIR", os.path.dirname(os.path.abspath(__file__))))
KY = "Kyojin <https://github.com/Yamz-Labs/kyojin> commit e2f37564bbae01a6ea8ad4176dfd97db09c253aa (release 1.8.2)"

# source path in the Kyojin tree -> file name here
FILES = {
    "tools/ds41/ds41_chat.py": "ds41_chat.py",
    "tools/ds41/ds41_parse.py": "ds41_parse.py",
    "tools/ds41/ds41_engine.py": "ds41_engine.py",
    "tools/ds41/serve.py": "ds41_serve.py",
    "tools/ds41/serve_metrics.py": "serve_metrics.py",
    "tools/ds41/serve_ping.py": "serve_ping.py",
    "tools/ds41/serve_reqlog.py": "serve_reqlog.py",
    "tools/ds41/conftest.py": "conftest.py",
    "tools/ds41/test_ds41_chat.py": "test_ds41_chat.py",
    "tools/ds41/test_ds41_parse.py": "test_ds41_parse.py",
    "tools/ds41/test_ds41_serve.py": "test_ds41_serve.py",
    "tools/serve_compat.py": "serve_compat.py",
    "tools/startup_health.py": "startup_health.py",
    "tools/test_serve_compat.py": "test_serve_compat.py",
}


def header(rel: str, note: str = "") -> str:
    text = (
        f"# Vendored from {KY}.\n"
        f"# Source path in the Kyojin tree: {rel}. MIT License, Copyright (c) 2025 Turboderp -- see LICENSE.kyojin.\n"
    )
    if note:
        text += note
    else:
        text += (
            "# Unmodified verbatim copy for the two-node EXL3 TP server; entry points here are\n"
            "# tp_serve.py (rank 0, HTTP) and tp_rank1.py (rank 1, mirror worker).\n"
        )
    return text


def patch_serve(text: str) -> str:
    """The three LOCAL blocks of ds41_serve.py: per-request engine fields.

    1. validation in completions(): `speculative` (bool) and `spec_k` (positive int), plus a 400
       when speculative is explicitly asked for but the engine has no DSpark;
    2. the worker's generate() call forwards `speculative` / `spec_k` when the request sent
       them, and `session` (1..EXL3_TP_SESSIONS, multi-session slot pinning; the engine
       validates and refuses it on single-session servers -- unknown by other engines
       through **_).
    Everything else stays byte-identical to upstream serve.py (asserted by the anchors)."""
    anchor_a = (
        '            stops = body.get("stop") or []\n'
        '            body["_stop"] = [stops] if isinstance(stops, str) else list(stops)\n'
        '            if not all(isinstance(x, str) for x in body["_stop"]):\n'
        '                raise web.HTTPBadRequest(reason="stop must be a string or a list of strings")\n'
    )
    insert_a = anchor_a + (
        '            # LOCAL (exl3-tp): per-request speculative decoding (tp_engine.TPEngine; ignored by\n'
        '            # other engines through **_). speculative: DSpark for this request when the server\n'
        '            # loaded it (null = the server default); spec_k caps the drafted tokens per round.\n'
        '            if body.get("speculative") is not None and not isinstance(body["speculative"], bool):\n'
        '                raise web.HTTPBadRequest(reason="speculative must be boolean")\n'
        '            if body.get("spec_k") is not None and (not isinstance(body["spec_k"], int) or isinstance(body["spec_k"], bool) or body["spec_k"] < 1):\n'
        '                raise web.HTTPBadRequest(reason="spec_k must be a positive integer")\n'
        '            if body.get("speculative") is True and not getattr(engine, "dspark", False):\n'
        '                raise web.HTTPBadRequest(reason="speculative decoding is not enabled on this server")\n'
    )
    anchor_b = '                            **({"images": body["_images"]} if body.get("_images") else {}),\n'
    insert_b = anchor_b + (
        '                            **({"speculative": body["speculative"]} if body.get("speculative") is not None else {}),   # LOCAL (exl3-tp)\n'
        '                            **({"spec_k": body["spec_k"]} if body.get("spec_k") is not None else {}),                 # LOCAL (exl3-tp)\n'
        '                            **({"session": body["session"]} if body.get("session") is not None else {}),              # LOCAL (exl3-tp): explicit session slot\n'
    )
    assert anchor_a in text, "serve.py validation anchor not found (upstream changed?)"
    assert anchor_b in text, "serve.py worker-call anchor not found (upstream changed?)"
    out = text.replace(anchor_a, insert_a, 1).replace(anchor_b, insert_b, 1)
    # the session block must survive re-vendoring: exactly one forwarded field line,
    # after spec_k, or the multi-session override silently disappears from the HTTP API
    assert out.count('"session": body["session"]') == 1, "session LOCAL block not applied exactly once"
    return out


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for src_rel, name in FILES.items():
        text = (SRC / src_rel).read_text(encoding="utf-8")
        if name == "ds41_serve.py":
            note = (
                "# Near-verbatim copy for the two-node EXL3 TP server; entry points here are\n"
                "# tp_serve.py (rank 0, HTTP) and tp_rank1.py (rank 1, mirror worker).\n"
                "# LOCAL modifications (marked 'LOCAL (exl3-tp)', applied by vendor_kyojin.py): the\n"
                "# per-request `speculative` / `spec_k` fields are validated and forwarded to\n"
                "# engine.generate for the DSpark engine, and the optional `session` field pins a\n"
                "# multi-session slot (tp_sessions; the engine validates the id). Everything else\n"
                "# is byte-identical upstream.\n"
            )
            text = patch_serve(text)
            (DST / name).write_text(header(src_rel, note) + text, encoding="utf-8")
            print(f"vendored {name}  <-  {src_rel}  (+ LOCAL speculative/spec_k/session patch)")
            continue
        (DST / name).write_text(header(src_rel) + text, encoding="utf-8")
        print(f"vendored {name}  <-  {src_rel}")
    # golden-test data: the reference encoding + golden prompt/reply files already carry their own
    # DeepSeek-AI MIT attribution comments; copied byte for byte
    if (DST / "testdata").exists():
        shutil.rmtree(DST / "testdata")
    shutil.copytree(SRC / "tools/ds41/testdata", DST / "testdata")
    print("vendored testdata/  <-  tools/ds41/testdata")
    shutil.copy(SRC / "LICENSE", DST / "LICENSE.kyojin")
    print("vendored LICENSE.kyojin  <-  LICENSE")


if __name__ == "__main__":
    main()
