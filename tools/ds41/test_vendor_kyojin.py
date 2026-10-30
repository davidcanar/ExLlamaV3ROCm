"""The vendoring layer must preserve every LOCAL (exl3-tp) block through a re-vendor.

ds41_serve.py carries three marked forwarding lines (speculative / spec_k / session); if a
future re-vendor against a new upstream drops one, the multi-session HTTP override (or the
DSpark per-request fields) would silently disappear from the API. patch_serve() asserts
its anchors and, now, that the session line is applied exactly once.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vendor_kyojin  # noqa: E402

ANCHOR_A = (
    '            stops = body.get("stop") or []\n'
    '            body["_stop"] = [stops] if isinstance(stops, str) else list(stops)\n'
    '            if not all(isinstance(x, str) for x in body["_stop"]):\n'
    '                raise web.HTTPBadRequest(reason="stop must be a string or a list of strings")\n'
)
ANCHOR_B = '                            **({"images": body["_images"]} if body.get("_images") else {}),\n'


def test_patch_serve_applies_all_three_local_blocks():
    text = ANCHOR_A + "\n" + ANCHOR_B
    out = vendor_kyojin.patch_serve(text)
    for field in ("speculative", "spec_k", "session"):
        assert f'**({{"{field}": body["{field}"]}}' in out, field
    assert out.count('"session": body["session"]') == 1
    # the session line rides after spec_k, in the worker's generate() call
    i_spec = out.index('"spec_k": body["spec_k"]')
    i_sess = out.index('"session": body["session"]')
    assert 0 < i_sess - i_spec < 200


def test_patch_serve_refuses_when_anchors_drift():
    for drop in ("A", "B"):
        parts = {"A": ANCHOR_A, "B": ANCHOR_B}
        text = "".join(v for k, v in parts.items() if k != drop)  # the other anchor only
        try:
            vendor_kyojin.patch_serve(text)
        except AssertionError:
            continue
        raise AssertionError(f"patch_serve accepted a drifted anchor ({drop} missing)")


def test_revendoring_reproduces_the_deployed_serve_layer():
    """Against the real kyojin tree, patch_serve() + the attribution header must rebuild
    the deployed ds41_serve.py byte for byte (skipped when the source tree is absent)."""
    src = vendor_kyojin.SRC / "tools/ds41/serve.py"
    if not src.exists():
        import pytest
        pytest.skip("kyojin source tree not present")
    deployed = (Path(__file__).resolve().parent / "ds41_serve.py").read_text(encoding="utf-8")
    shebang = "#!/usr/bin/env python3"
    assert deployed.count(shebang) == 1, "deployed ds41_serve.py lacks its upstream body"
    body = deployed[deployed.index(shebang):]
    rebuilt = vendor_kyojin.patch_serve(src.read_text(encoding="utf-8"))
    assert body == rebuilt, "re-vendoring would change the deployed ds41_serve.py body"
