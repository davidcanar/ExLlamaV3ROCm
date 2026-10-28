# Vendored from Kyojin <https://github.com/Yamz-Labs/kyojin> commit e2f37564bbae01a6ea8ad4176dfd97db09c253aa (release 1.8.2).
# Source path in the Kyojin tree: tools/ds41/conftest.py. MIT License, Copyright (c) 2025 Turboderp -- see LICENSE.kyojin.
# Unmodified verbatim copy for the two-node EXL3 TP server; entry points here are
# tp_serve.py (rank 0, HTTP) and tp_rank1.py (rank 1, mirror worker).
"""pytest: the downloaded reference folders are not part of the suite."""
collect_ignore = ["_ref", "_scratch", "testdata"]

import os
os.environ.setdefault("DS41_PFX_MIN_FREE_GIB", "0")      # tests must not depend on the host's free memory
