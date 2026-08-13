#!/usr/bin/env python3
"""Single source of truth for the pipeline weights.

Default = the adopted weight set (`weights/final.json`).
  tools/ensemble/weights/final.json      default — the weights the pipeline actually uses
  tools/ensemble/weights/search/         search experiment outputs, used only when selected
                                         explicitly via `WEIGHTS`

To swap in a different set, point the `WEIGHTS` environment variable or `--set` at a JSON file
with the same schema (searches are produced by `comb_search.py` / `base.py`).

Every stage reads its weights through this module, so there is a single place to change them.

  pipeline/S1_base/build_base.py       ->  base()   encoder ensemble (anchor and members)
  run_submission.py (S3)               ->  comb()   reranker z-fusion
  pipeline/S4_tail/tail_refinement.py  ->  tail()   tail overlay, NN, R@5, cons6
                                           s4e()    final pass threshold

    python tools/ensemble/adopted.py --get comb --variant best
    python tools/ensemble/adopted.py --get tail_w
    python tools/ensemble/adopted.py --show
"""
from __future__ import annotations

import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT = os.path.join(HERE, "weights", "final.json")   

COMB_KEYS = ["sim", "internvl_r32", "pixtral", "qwen3vl_2b", "llama", "8b"]
# Engine keys are the encoder/reranker names as written in final.json; comb and base use no aliases.


def _path() -> str:
    return os.environ.get("WEIGHTS", DEFAULT)


def load(path: str | None = None) -> dict:
    """The full weight set. Defaults to the adopted values."""
    return json.load(open(path or _path()))


def _clean(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def base(path: str | None = None) -> dict:
    """Flat encoder ensemble member weights {name: w} (default = the `base` entry of final.json)."""
    return _clean(load(path)["base"])


def base_variant(name: str, path: str | None = None) -> dict:
    """`base` re-weighted by one entry of `base_variants`.
    """
    flat = base(path)
    spec = _clean(load(path).get("base_variants", {})).get(name)
    if spec is None:
        raise KeyError(f"base_variants has no `{name}` (have: "
                       f"{sorted(_clean(load(path).get('base_variants', {})))})")
    if spec == "all":
        return {k: 1.0 for k in flat}
    return {k: float(spec.get(k, 0.15)) for k in flat}


def base_variant_names(path: str | None = None) -> list:
    return sorted(_clean(load(path).get("base_variants", {})))


def comb(variant: str = "best", path: str | None = None) -> dict:
    """Reranker z-fusion weights. `variant` overrides them from the `variants` entry
    (adopted = best)."""
    c = load(path)
    w = _clean(c["comb"])
    if variant and variant in c.get("variants", {}):
        w.update(_clean(c["variants"][variant].get("comb", {})))
    return dict(w)


def tail(path: str | None = None) -> dict:
    return _clean(load(path)["tail"])


def tail_w(path: str | None = None) -> str:
    """S4a overlay weights as the comma-separated form taken by tail_refinement s4a_overlay(w=...)."""
    return ",".join(str(x) for x in tail(path)["overlay_w"])


def injective(path: str | None = None) -> dict:
    return _clean(load(path)["injective"])


def s4e(path: str | None = None) -> dict:
    c = load(path)
    t = _clean(c.get("tail", {}))
    if "final_pass_tau_px" in t:
        return {"tau_px": t["final_pass_tau_px"]}
    return _clean(c.get("s6", {}))


def _main():
    ap = argparse.ArgumentParser(description="Query the pipeline weights (defaults to the adopted set)")
    ap.add_argument("--get", choices=["base", "comb", "tail", "tail_w", "injective", "s4e", "all"],
                    default="all")
    ap.add_argument("--variant", default="best", help="comb variant (adopted = best)")
    ap.add_argument("--set", dest="path", default=None, help="path to a weights JSON (default tools/ensemble/weights/final.json)")
    ap.add_argument("--show", action="store_true", help="human-readable summary")
    a = ap.parse_args()

    if a.show:
        p = a.path or _path()
        print(f"[weights] source = {p}")
        print(f"  base      {base(a.path)}")
        print(f"  comb      {comb(a.variant, a.path)}   (variant={a.variant})")
        print(f"  tail      overlay_w {tail_w(a.path)} · {({k: v for k, v in tail(a.path).items() if k != 'overlay_w'})}")
        print(f"  injective {injective(a.path)} · final pass {s4e(a.path)}")
        return
    if a.get == "tail_w":
        print(tail_w(a.path)); return
    out = {"base": base, "comb": (lambda p=None: comb(a.variant, p)), "tail": tail,
           "injective": injective, "s4e": s4e}.get(a.get)
    print(json.dumps(out(a.path) if out else load(a.path)))


if __name__ == "__main__":
    _main()
