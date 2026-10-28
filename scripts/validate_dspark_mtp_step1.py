#!/usr/bin/env python3
"""
MTP/DSpark validation, step 1 (single Strix Halo host, no model weights).

Probes, in order:

  A. Tiny synthetic checkpoints: does any existing testlib builder produce a checkpoint with a
     valid DSpark/MTP head?  tests/testlib.tiny_models.make_dsv4_checkpoint is the only tiny
     random-weight builder; it is checked as-is and with num_nextn_predict_layers/compress_ratios
     overrides, and the full converted DS4.1 at $EXL3DS4_ROOT/model is probed (headers only, no
     weight load) to confirm a real head exists for the later two-node step.

  B. The DSpark draft attention core: dsa_attn(..., nc_block = True) called with the EXACT
     argument shape of DSparkAttention.forward (paged flat main-kv plane read through the block
     table, non-causal kv_chunk, per-head sinks, fused eq.26 de-rotation, group-major store,
     page_size 256), compared against an fp64 reference of the documented NC_BLOCK semantics
     (every query row sees the SAME range [q_pos0 - hist, q_pos0 + block): paged window history
     ++ the whole block, one softmax with sinks).

     On ROCm (gfx1151) the MQA prefill kernel (dsa_mqa.py) takes over where the shape allows
     (H >= 16, power-of-2 head_dim); the upstream Triton one-shot kernel runs otherwise. Both
     dispatches are covered: T* cases force the Triton kernel (H = 8), M* cases take the MQA
     dispatch (incl. the real head shape H = 64, D = 448 + 64). Which kernel served each case
     is recorded. Each case also runs twice and must be bitwise deterministic.

No model weights are created or loaded: part A builds the sanctioned random tiny checkpoint, part
B drives the kernel directly on synthetic tensors, in the pattern of tests/kernels/dsa/test_dsa_kernels.py.

Usage (box1):  source ~/exl3ds4/env.sh && python scripts/validate_dspark_mtp_step1.py
Exit code 0 = all checks passed.
"""

import json
import os
import sys
import tempfile

import torch

EXL3_ROOT = os.environ.get("EXL3_ROOT", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "exllamav3-dsv41f"))
sys.path.insert(0, EXL3_ROOT)
sys.path.insert(0, os.path.join(EXL3_ROOT, "tests"))

PAGE_SIZE = 256          # DSpark paged main-kv plane: fixed page size (dspark.py: page_size = 256)
TOL = 6e-3               # rel-err tolerance vs fp64, same as tests/kernels/dsa/test_dsa_kernels.py

DEVICE = torch.device("cuda:0")

FAILURES = []


def ok(msg):
    print(f"  [PASS] {msg}")


def fail(msg):
    print(f"  [FAIL] {msg}")
    FAILURES.append(msg)


def section(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


# ---------------------------------------------------------------- part A: checkpoint probe

def scan_mtp_tensors(model_dir):
    from safetensors import safe_open
    import glob
    keys = set()
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, framework = "pt") as fh:
            keys.update(fh.keys())
    return keys


def part_a_tiny():
    section("A1. Existing tiny builder (tests/testlib/tiny_models.make_dsv4_checkpoint)")
    from testlib.tiny_models import make_dsv4_checkpoint, DSV4_TINY
    from exllamav3 import Config

    out = tempfile.mkdtemp(prefix = "dspark_val_a1_")
    make_dsv4_checkpoint(out, seed = 7)
    cfg = json.load(open(os.path.join(out, "config.json")))
    keys = scan_mtp_tensors(out)
    n_mtp = sum(1 for k in keys if k.startswith("mtp."))
    print(f"  tiny DSV4 checkpoint: {len(keys)} tensors, {n_mtp} mtp.* tensors, "
          f"num_nextn_predict_layers = {cfg.get('num_nextn_predict_layers')}, "
          f"num_hidden_layers = {cfg['num_hidden_layers']}")
    config = Config.from_directory(out)
    has_mtp = "mtp" in config.model_classes
    print(f"  Config model_classes: {sorted(config.model_classes)}  -> mtp component: {has_mtp}")
    if n_mtp == 0 and not has_mtp:
        ok("tiny checkpoint as built has NO MTP/DSpark head (expected: builder emits none)")
    else:
        fail("tiny checkpoint unexpectedly carries an MTP head")

    print("\nA2. Same builder pushed toward an MTP head (num_nextn_predict_layers = 2, +2 ratio entries)")
    out2 = tempfile.mkdtemp(prefix = "dspark_val_a2_")
    make_dsv4_checkpoint(out2, seed = 7, num_nextn_predict_layers = 2,
                         compress_ratios = list(DSV4_TINY["compress_ratios"]) + [0, 0])
    keys2 = scan_mtp_tensors(out2)
    n_mtp2 = sum(1 for k in keys2 if k.startswith("mtp."))
    config2 = Config.from_directory(out2)
    has_mtp2 = "mtp" in config2.model_classes
    print(f"  overridden tiny checkpoint: {len(keys2)} tensors, {n_mtp2} mtp.* tensors, "
          f"config num_mtp_layers = {config2.num_mtp_layers}")
    print(f"  Config model_classes: {sorted(config2.model_classes)}  -> mtp component: {has_mtp2}")
    if n_mtp2 == 0 and not has_mtp2:
        ok("override creates no head: config declares 2 MTP layers but no mtp.* tensors exist, "
           "so the loader drops the component (deepseek_v4.py gating). No tiny checkpoint with a "
           "VALID MTP head can be built from the existing testlib builders.")
    else:
        fail("override unexpectedly produced a usable MTP head")


def part_a_full():
    section("A3. Full converted DS4.1 (headers only, nothing loaded) — target for the two-node step")
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        print(f"  {model_dir}: not present, skipping probe")
        return
    keys = scan_mtp_tensors(model_dir)
    mtp = sorted(k for k in keys if k.startswith("mtp."))
    tc = json.load(open(os.path.join(model_dir, "config.json")))["text_config"]
    print(f"  {len(keys)} tensors, {len(mtp)} mtp.* tensors")
    print(f"  dspark_block_size = {tc.get('dspark_block_size')}, num_nextn_predict_layers = "
          f"{tc.get('num_nextn_predict_layers')}, dspark_target_layer_ids = {tc.get('dspark_target_layer_ids')}, "
          f"dspark_n_routed_experts = {tc.get('dspark_n_routed_experts')}")
    print(f"  head attention tensors present: {all(any(k == f'mtp.{i}.attn.wkv.{s}' for k in mtp) for i in range(3) for s in ('trellis', 'mul1'))}")
    if len(mtp) > 0 and tc.get("dspark_block_size"):
        ok("full DS4.1 checkpoint carries a DSpark head (3 blocks) — usable later on two nodes")
    else:
        fail("full DS4.1 checkpoint appears to lack the DSpark head")


# ---------------------------------------------------------------- part B: NC_BLOCK kernel

def derotate_ref(o_r, inv_freq, pos):
    """GPT-J pair rotation of (H, D_r) by pos * inv_freq, fp64 (tests/kernels/dsa reference)"""
    theta = inv_freq.double() * pos
    cos, sin = theta.cos(), theta.sin()
    e, o = o_r[..., 0::2], o_r[..., 1::2]
    return torch.stack((e * cos - o * sin, o * cos + e * sin), dim = -1).flatten(-2)


def ref_nc_block(q, main_kv, bt, kv_chunk, sinks, p0, hist, derot_inv_freq, groups, D_r, page_size = PAGE_SIZE):
    """fp64 reference of the NC_BLOCK semantics the DSpark drafter relies on: every query row sees
    the SAME keys — paged history rows [p0 - hist, p0) via the block table ++ the whole non-causal
    block [p0, p0 + s) — in one softmax with per-head sinks; eq.26 de-rotation at q_pos0 + row;
    group-major store."""
    s, H, D = q.shape
    D_c = D - D_r
    scale = D ** -0.5
    keys = []
    for a in range(p0 - hist, p0):
        phys = bt[0, a // page_size].item()
        keys.append(main_kv[phys * page_size + a % page_size].double())
    for j in range(s):
        keys.append(kv_chunk[j].double())
    keys = torch.stack(keys)                                  # (hist + s, D)
    out = torch.zeros((s, H, D), dtype = torch.float64, device = q.device)
    for r in range(s):
        sc = (q[r].double() @ keys.T) * scale                 # (H, n)
        if sinks is not None:
            sc = torch.cat([sc, sinks.double().unsqueeze(1)], dim = 1)
            p = torch.softmax(sc, dim = -1)[:, :-1]
        else:
            p = torch.softmax(sc, dim = -1)
        out[r] = p @ keys
        if derot_inv_freq is not None:
            out[r, :, D_c:] = derotate_ref(out[r, :, D_c:], derot_inv_freq, p0 + r)
    hpg = H // groups
    return out.view(s, groups, hpg * D).transpose(0, 1).contiguous().float()   # (groups, s, hpg*D)


class _KernelRecorder:
    """Record launches of a triton JITFunction without changing behavior"""
    def __init__(self, kern):
        self.kern, self.count = kern, 0

    def __getitem__(self, grid):
        self.count += 1
        return self.kern[grid]

    def run(self, *args, **kwargs):
        return self.kern.run(*args, **kwargs)   # pick_config's warmup probe


def run_case(name, seed, H, D_c, D_r, s, window, p0, groups, expect):
    """One DSpark-shaped dsa_attn(nc_block = True) call vs the fp64 reference"""
    import exllamav3.modules.attention_fn.dsa_triton as dsa_triton
    import exllamav3.modules.attention_fn.dsa_mqa as dsa_mqa
    from exllamav3.modules.attention_fn.dsa_triton import dsa_attn

    torch.manual_seed(seed)
    D = D_c + D_r
    hist = min(window, p0)
    pages = (p0 + PAGE_SIZE - 1) // PAGE_SIZE            # virtual pages covering [0, p0)
    perm = torch.randperm(pages, device = DEVICE, dtype = torch.int32)
    bt = perm.unsqueeze(0).contiguous()                  # (1, pages) row, as DSpark passes block_table[r:r+1]
    main_kv = torch.randn((pages * PAGE_SIZE, D), dtype = torch.half, device = DEVICE)   # kl.kv.view(-1, D)
    kv_chunk = torch.randn((s, D), dtype = torch.half, device = DEVICE)
    q = torch.randn((s, H, D), dtype = torch.half, device = DEVICE)
    sinks = torch.randn(H, device = DEVICE) * 2.0 + 4.0
    pc0 = q.new_empty((1, D_c))
    pr0 = q.new_empty((1, D_r))
    derot_inv_freq = -1.0 / (10000.0 ** (torch.arange(0, D_r, 2, dtype = torch.float, device = DEVICE) / D_r))
    hpg = H // groups

    def call():
        return dsa_attn(
            q, pc0, pr0, bt,
            sinks = sinks,
            ring = main_kv,                              # paged flat plane; ring_beg = 0
            kv_chunk = kv_chunk,
            win_len = hist + s,
            win_floor = p0 - hist,
            ring_beg = 0,
            indices = None, k_len = 0,
            pool_len = 0,
            q_pos0 = p0,
            compress_rate = 1,
            scale = D ** -0.5,
            derot_inv_freq = derot_inv_freq,
            groups = groups,
            page_size = PAGE_SIZE,
            nc_block = True,
            out = torch.empty((groups, s, hpg * D), dtype = torch.half, device = DEVICE),
        )

    triton_rec = _KernelRecorder(dsa_triton._dsa_attn_kernel)
    split_rec = _KernelRecorder(dsa_triton._dsa_attn_split_kernel)
    mqa_rec = _KernelRecorder(dsa_mqa._dsa_prefill_mqa_kernel)
    saved = (dsa_triton._dsa_attn_kernel, dsa_triton._dsa_attn_split_kernel, dsa_mqa._dsa_prefill_mqa_kernel)
    dsa_triton._dsa_attn_kernel, dsa_triton._dsa_attn_split_kernel = triton_rec, split_rec
    dsa_mqa._dsa_prefill_mqa_kernel = mqa_rec
    try:
        got = call().float().clone()
        got2 = call().float().clone()
    finally:
        dsa_triton._dsa_attn_kernel, dsa_triton._dsa_attn_split_kernel = saved[0], saved[1]
        dsa_mqa._dsa_prefill_mqa_kernel = saved[2]

    used = "mqa-prefill" if mqa_rec.count else ("triton-one-shot" if triton_rec.count else
                                                ("triton-split" if split_rec.count else "NONE"))
    ref = ref_nc_block(q, main_kv, bt, kv_chunk, sinks, p0, hist, derot_inv_freq, groups, D_r)
    scale = max(ref.abs().max().item(), 1e-6)
    rel = (got - ref).abs().max().item() / scale
    det = torch.equal(got, got2)

    print(f"  {name:34s} kernel: {used:16s} (expected {expect:16s}) rel-err {rel:.2e} deterministic: {det}")
    checks = [
        (used == expect, f"{name}: kernel dispatch was {used}, expected {expect}"),
        (rel < TOL, f"{name}: rel err {rel:.2e} >= {TOL}"),
        (det, f"{name}: non-deterministic output"),
        (torch.isfinite(got).all().item(), f"{name}: non-finite output"),
    ]
    for good, msg in checks:
        if good:
            ok(msg)
        else:
            fail(msg)


def part_b():
    section("B. dsa_attn NC_BLOCK (DSpark draft attention) vs fp64 reference, gfx1151")
    from exllamav3.util.backend import DSA_MQA, ROCM
    print(f"  host backend: ROCm = {ROCM}, DSA_MQA = {DSA_MQA} (MQA prefill takes NC_BLOCK shapes with H >= 16)")
    if not DSA_MQA:
        print("  NOTE: DSA_MQA off -> every case runs the Triton one-shot kernel; M* expectations would fail")
    cases = [
        # name, seed, H, D_c, D_r, s(block), window, p0, o_groups, expected kernel
        ("T1 tiny-dims multi-page hist", 11,  8,  48, 16, 6, 128,  600, 4, "triton-one-shot"),
        ("T2 tiny-dims floor-at-zero",   12,  8,  48, 16, 6,   8,    5, 4, "triton-one-shot"),
        ("T3 head-dims 448+64 few heads",13,  8, 448, 64, 6, 128,  600, 8, "triton-one-shot"),
        ("M1 tiny-dims MQA dispatch",    14, 32,  48, 16, 6, 128,  600, 4, "mqa-prefill"),
        ("M2 real head shape H64 D512",  15, 64, 448, 64, 6, 128, 1000, 8, "mqa-prefill"),
    ]
    for c in cases:
        run_case(*c)


def main():
    print(f"torch {torch.__version__}, device {DEVICE}: {torch.cuda.get_device_name(DEVICE)}")
    part_a_tiny()
    part_a_full()
    part_b()
    section("SUMMARY")
    if FAILURES:
        print(f"  {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"    - {f}")
        sys.exit(1)
    print("  all checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
