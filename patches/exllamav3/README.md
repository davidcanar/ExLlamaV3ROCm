# patches/exllamav3 — the `multi-node-rocm` branch of ExLlamaV3

Seven commits on top of the PR #449 (`dsv41f`) base of ExLlamaV3 that add the
cross-host tensor-parallel runtime this deployment runs on. Generated with:

    git format-patch d45e8961..multi-node-rocm

Upstream / base:

- Repository: <https://github.com/anoane/exllamav3-anemone> (fork of
  turboderp/ExLlamaV3, PR #449 "DeepseekV4.1: Tensor-parallel support")
- Base commit: `d45e89611738e9134f72f5c9679cc9eae113bba8` (branch `dsv41f`)
- Resulting branch head: `0e259659f09b47c79c04ac9157af284cc70a1604`
  (`multi-node-rocm`), which includes the DSV4 per-worker recurrent-`wshift`
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

Apply with `git am` to preserve authorship and commit identity:

    git clone https://github.com/anoane/exllamav3-anemone.git exllamav3-dsv41f
    cd exllamav3-dsv41f
    git fetch origin dsv41f
    git checkout -b multi-node-rocm d45e89611738e9134f72f5c9679cc9eae113bba8
    git am /path/to/patches/exllamav3/*.patch

`setup.sh` performs these steps automatically into `src/exllamav3-dsv41f`.

License: ExLlamaV3 is MIT (Copyright (c) 2025 Turboderp); these patches are
MIT (Copyright (c) 2026 David Canar), see the repository LICENSE/NOTICE.
