#!/usr/bin/env python3
"""artifacts — idempotent artifact-reuse helper.

By default the pipeline reads the bundled `assets/{model,cache}` and runs without
recomputation; `--rep` switches it to `assets/{model_rep,cache_rep}`. In either mode an
artifact that already exists is not rebuilt: the same inputs give the same result.

    from utils.artifacts import skip_if_exists      # with pipeline/ on sys.path
    skip_if_exists(OUT, a.overwrite)               # exits 0 on the spot if OUT exists

`--overwrite` forces a rebuild. Call this before loading the model, since loading dominates
the wall-clock cost.
"""
import os
import sys


def skip_if_exists(path, overwrite=False, label=""):
    """Log a message and exit the process with status 0 if the artifact already exists.

    path      : final output path (None or empty means do nothing)
    overwrite : True skips the check and proceeds (usually the `--overwrite` flag)
    label     : name to show in the log (defaults to the file name)
    """
    if not path or overwrite or not os.path.exists(path):
        return False
    name = label or os.path.basename(path)
    size = os.path.getsize(path) / 1048576
    print(f"[skip] {name} already exists ({size:.0f} MB) -> {path}\n"
          f"       pass --overwrite to rebuild it", flush=True)
    sys.exit(0)


def add_overwrite_arg(ap):
    """The standard `--overwrite` flag, defined once so its wording stays consistent."""
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild the artifact even if it already exists (default: skip)")
    return ap
