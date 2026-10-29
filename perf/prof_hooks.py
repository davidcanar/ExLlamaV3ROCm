"""Rank-0 timing instrumentation for the DS4.1 TP decode path (monkeypatching, no library edits).

PROF.on = True turns on synchronized, nested per-callable timing: every wrapped callable is timed
from a device-synchronized start to a device-synchronized end (total), and the host time spent
inside the Python call before the trailing sync is kept separately (cpu). Exclusive times subtract
the wrapped children. Off by default, so unprofiled steps run untouched (one attribute test per call).
"""
from __future__ import annotations

import collections
import functools
import time

import torch


class _Prof:
    def __init__(self):
        self.on = False
        self.stack = []
        self.stats = collections.defaultdict(lambda: [0, 0.0, 0.0, 0.0])  # n, total, excl, cpu
        self.device = None

    def reset(self):
        self.stats.clear()
        self.stack.clear()

    def report(self, steps: int, title: str = "") -> str:
        rows = sorted(self.stats.items(), key=lambda kv: -kv[1][2])
        tot_excl = sum(v[2] for v in self.stats.values())
        out = [f"== {title}: per-step exclusive time (ms), {steps} profiled steps; sum {tot_excl / steps * 1e3:.2f} ms/step"]
        out.append(f"{'label':<44} {'calls/step':>10} {'excl ms':>9} {'incl ms':>9} {'cpu ms':>9} {'excl %':>7}")
        for k, (n, tot, excl, cpu) in rows:
            out.append(f"{k:<44} {n / steps:>10.1f} {excl / steps * 1e3:>9.3f} {tot / steps * 1e3:>9.3f} "
                       f"{cpu / steps * 1e3:>9.3f} {100 * excl / max(tot_excl, 1e-12):>6.1f}%")
        return "\n".join(out)


PROF = _Prof()


def _sync():
    torch.cuda.synchronize()


def timed(label_fn):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not PROF.on:
                return fn(*args, **kwargs)
            label = label_fn(*args, **kwargs) if callable(label_fn) else label_fn
            _sync()
            t0 = time.perf_counter()
            PROF.stack.append(0.0)
            try:
                r = fn(*args, **kwargs)
                t1 = time.perf_counter()
                _sync()
                t2 = time.perf_counter()
            finally:
                child = PROF.stack.pop()
            total = t2 - t0
            s = PROF.stats[label]
            s[0] += 1
            s[1] += total
            s[2] += total - child
            s[3] += t1 - t0
            if PROF.stack:
                PROF.stack[-1] += total
            return r
        return wrapper
    return deco


def _attn_label(self, *a, **k):
    t = self.layer_type
    if self.compress_rate:
        t += f"-cr{self.compress_rate}"
    mode = self.indexer_mode or "-"
    if self.candidate_mode:
        mode += f"/{self.candidate_mode}"
    return f"attn[{t},{mode}]"


def install():
    """Patch the library classes in place (rank 0, in-process TP worker)."""
    from exllamav3.modules import dsv4 as dsv4_mod
    from exllamav3.modules.dsv4 import DSV4Attention, DSV4Compressor
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.hyperconnections import HyperConnection, HyperHead, ExpandStreams
    from exllamav3.modules.arch_specific.deepseek_v41 import EngramLayer
    from exllamav3.modules.embedding import Embedding
    from exllamav3.modules.linear import Linear
    from exllamav3.modules.rmsnorm import RMSNorm
    from exllamav3.model import model_tp_backend as be

    DSV4Attention.forward = timed(_attn_label)(DSV4Attention.forward)
    DSV4Attention._indexer_topk = timed(lambda self, *a, **k: f"  indexer_topk[{self.candidate_mode or 'plain'}]")(DSV4Attention._indexer_topk)
    DSV4Attention._project_qkv = timed("  attn.project_qkv")(DSV4Attention._project_qkv)
    DSV4Attention._project_o_grouped = timed("  attn.project_o")(DSV4Attention._project_o_grouped)
    DSV4Attention._indexer_keys = timed("  attn.indexer_keys")(DSV4Attention._indexer_keys)
    DSV4Compressor.forward_fused = timed("  attn.compressor")(DSV4Compressor.forward_fused)
    dsv4_mod.dsa_attn = timed("  attn.dsa_attn")(dsv4_mod.dsa_attn)
    dsv4_mod.dsa_indexer_scores = timed("    indexer.scores")(dsv4_mod.dsa_indexer_scores)

    from exllamav3.modules.transformer import TransformerBlock
    from exllamav3.model.model_tp import Model_TPMixin
    TransformerBlock.forward = timed("block(glue)")(TransformerBlock.forward)
    RMSNorm.forward = timed("  rmsnorm")(RMSNorm.forward)
    # forward_tp covers the TP dispatch + module walk; its exclusive time is the unattributed glue
    # (mp_model_forward itself is pickled by reference for child ranks, so it is left unwrapped)
    Model_TPMixin.forward_tp = timed("forward_tp(glue)")(Model_TPMixin.forward_tp)

    BlockSparseMLP.forward = timed("moe")(BlockSparseMLP.forward)
    HyperConnection.mix_norm = timed("hc.mix_norm")(HyperConnection.mix_norm)
    HyperConnection.apply_ = timed("hc.apply")(HyperConnection.apply_)
    HyperHead.forward = timed("hc.head")(HyperHead.forward)
    ExpandStreams.forward = timed("hc.expand")(ExpandStreams.forward)
    EngramLayer.forward = timed("engram")(EngramLayer.forward)
    Embedding.forward = timed("embed")(Embedding.forward)

    _lin_fwd = Linear.forward

    def lin_label(self, *a, **k):
        return "lm_head" if self.caps.get("logits_output") else "  linear(other)"
    Linear.forward = timed(lin_label)(_lin_fwd)

    # Print the TP split table (experts / heads per rank) when the plan is compiled
    from exllamav3.model import model_tp_alloc as alloc
    _compile = alloc.TPAllocator.compile_tp_plan

    def compile_tp_plan(self, *a, **k):
        try:
            self.print_split()
        except Exception as e:  # noqa: BLE001
            print(f"prof: print_split failed: {e}", flush=True)
        return _compile(self, *a, **k)
    alloc.TPAllocator.compile_tp_plan = compile_tp_plan

    be.TPBackendNCCL.all_reduce = timed(lambda self, t, *a, **k: f"    comm.all_reduce[{t.numel()}]")(be.TPBackendNCCL.all_reduce)
    be.TPBackendNCCL.broadcast = timed(lambda self, t, *a, **k: f"    comm.broadcast[{t.numel()}]")(be.TPBackendNCCL.broadcast)
    be.TPBackendNCCL.gather = timed("    comm.gather")(be.TPBackendNCCL.gather)
    be.TPBackendNCCL.gather_small = timed("    comm.gather_small")(be.TPBackendNCCL.gather_small)
    be.TPBackendNCCL.fwd_barrier = timed("    comm.fwd_barrier")(be.TPBackendNCCL.fwd_barrier)
