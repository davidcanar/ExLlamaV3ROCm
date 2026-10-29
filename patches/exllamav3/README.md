# patches/exllamav3 — the `multi-node-rocm` branch of ExLlamaV3

Eleven commits on top of the PR #449 (`dsv41f`) base of ExLlamaV3 that add the
cross-host tensor-parallel runtime this deployment runs on and its decode
performance work. Generated with:

    git format-patch d45e8961..multi-node-rocm

(`0008`–`0011` were exported with `git format-patch --start-number 8
0e259659..multi-node-rocm`.)

Upstream / base:

- Repository: <https://github.com/anoane/exllamav3-anemone> (fork of
  turboderp/ExLlamaV3, PR #449 "DeepseekV4.1: Tensor-parallel support")
- Base commit: `d45e89611738e9134f72f5c9679cc9eae113bba8` (branch `dsv41f`)
- Resulting branch head: `355bb55b` (`multi-node-rocm`); `0e259659` is the
  head after `0007`, which includes the DSV4 per-worker recurrent-`wshift`
  readback fix required for long-context serving.

Commits (in order):

1. `0001` TP: add cross-host NCCL collectives
2. `0002` TP: separate cross-host rank from CUDA device
3. `0003` TP: add symmetric cross-host model runtime
4. `0004` TP: Close an overflow segment's handle once its tensor is copied
5. `0005` Kernels: Never destroy the static per-device workspaces
6. `0006` TP: add scoped cross-host sync modes
7. `0007` TP: cross-host: read back recurrent wshift from the non-output rank
   worker (the long-context fix; source commit `0e259659` in the deployment
   README)
8. `0008` TP: decode all-reduces over the OdinLink stream API (odl_ar2;
   needs `odl_ar2/libodl_ar2.so`, enabled by `EXL3_TP_ODL_AR2=1`)
9. `0009` DeepseekV41: Engram reads host ids and stages its rows at pass start
10. `0010` DSV4Attention: candidate readers score per row without host syncs
11. `0011` TP: optional placement of whole-on-one-rank shared experts
    (`EXL3_TP_SHARED_EXPERT_DEVICE`)

Apply with `git am` to preserve authorship and commit identity:

    git clone https://github.com/anoane/exllamav3-anemone.git exllamav3-dsv41f
    cd exllamav3-dsv41f
    git fetch origin dsv41f
    git checkout -b multi-node-rocm d45e89611738e9134f72f5c9679cc9eae113bba8
    git am /path/to/patches/exllamav3/*.patch

`setup.sh` performs these steps automatically into `src/exllamav3-dsv41f`.

License: ExLlamaV3 is MIT (Copyright (c) 2025 Turboderp); these patches are
MIT (Copyright (c) 2026 David Canar), see the repository LICENSE/NOTICE.
