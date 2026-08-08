#!/usr/bin/env python3
"""S1 — build the base encoder ensemble score (flat 10 members).

base = normalize( Σₑ wₑ · normalize(scoreₑ) ) over the 10 members.
Weights come from the single source tools/ensemble/adopted.py (default = base.flat in final.json).
Equivalent to the nested form: the inner norm is a scalar scale plus a constant offset, so it does
not affect ranking.

  Members: anchor_tcap · anchor_filip · metaclip2 · mc2h378_peft · gme
           eva02_pre · beit3_v2 · beit3_helip · metaclip_v1 · siglip_maxsim

Output: base_score.pt ([1978, 36773]); use `--out` to change the path.
Usage: ENS_DEV=cuda:0 python pipeline/S1_base/build_base.py [--out PATH]
Environment: WEIGHTS (weight set) · S1_CACHE (member cache) · TRACK4_GALLERY · QUERY_INDEX · ENS_DEV
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402

sys.path[:0] = [os.path.join(ROOT, "tools", "ensemble"), os.path.join(ROOT, "pipeline"),
                os.environ.get("TRACK4_CODE", _REPO)]
from utils import gallery_norm as GN                      # noqa: E402
from ensemble import mmnorm                    # noqa: E402  tools/ensemble combination core
from adopted import base as base_weights          # noqa: E402  tools/ensemble single source

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=None, help="output .pt (default = <S1 cache>/base_score.pt)")
ap.add_argument("--overwrite", action="store_true",
                help="rebuild even if the artifact exists (default: skip)")
ap.add_argument("--rep", action="store_true",
                help="reproduction build: read members from and write output to assets/cache_rep/s1_base")
a = ap.parse_args()

DEV = os.environ.get("ENS_DEV", "cuda:0")
S1  = os.environ.get("S1_CACHE",
                     f"{_REPO}/assets/{'cache_rep' if a.rep else 'cache'}/s1_base")   # S1 cache root
MEM = f"{S1}/members"                                               # the 10 encoder member caches
OUT = a.out or f"{S1}/base_score.pt"
skip_if_exists(OUT, a.overwrite)
os.makedirs(S1, exist_ok=True)
PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
QIDX = os.environ.get("QUERY_INDEX", f"{PAB_TEST}/query_index.txt")

FLAT = base_weights()                             # flat member weights {name: w} (default = base in final.json)
if not FLAT:
    raise SystemExit("[build_base] base weights are empty (check the `base` entry in final.json).")

gal = sorted(os.listdir(GN.GALLERY_DIR))
Q = len([l for l in open(QIDX) if l.strip()])


def tta(fp, views):
    """TTA view dump -> cosine score reordered to the gallery order."""
    d = torch.load(fp, map_location="cpu")
    gb = [g[:-4] if g.endswith(".jpg") else g for g in d["G_base"]]
    g2r = {g: i for i, g in enumerate(gb)}
    perm = torch.tensor([g2r[g[:-4] if g.endswith(".jpg") else g] for g in gal])
    vv = [v for v in views if v in d["img"]]
    Qb = F.normalize(d["txt"]["base"].float(), dim=-1)
    Gc = F.normalize(torch.stack([F.normalize(d["img"][v].float(), dim=-1) for v in vv], 0).mean(0), dim=-1)
    return GN.normalize((Qb @ Gc.t())[:, perm][:Q]).to(DEV)


def score(fp):
    """A [Q, G] score matrix used as is."""
    return GN.normalize(torch.load(fp, map_location="cpu", weights_only=False)).float().to(DEV)[:Q]


def feats(fp):
    """{G, Q} feats -> cosine score, reordered to the gallery order when G_base is present."""
    d = torch.load(fp, map_location="cpu", weights_only=False)
    s = F.normalize(d["Q"].float(), dim=-1) @ F.normalize(d["G"].float(), dim=-1).t()
    if "G_base" in d:
        stem = lambda t: t[:-4] if str(t).endswith(".jpg") else str(t)          # noqa: E731
        src = {stem(x): i for i, x in enumerate(d["G_base"])}
        s = s[:, torch.tensor([src[stem(g)] for g in gal])]
    return GN.normalize(s)[:Q].to(DEV)


# Flat members: base = norm(Σ wₑ·norm(scoreₑ)); all read from the S1 member cache.
LOADERS = {  # member -> (loader, file name, TTA views)
    "anchor_tcap":   ("tta",   "anchor_tcap_tta_views.pt",  ("base", "hflip", "z090")),
    "anchor_filip":  ("tta",   "anchor_filip_tta_views.pt", ("base", "hflip", "z080")),
    "metaclip2":     ("feats", "metaclip2_feats.pt",        None),
    "mc2h378_peft":  ("feats", "mc2h378_peft_score.pt",     None),
    "gme":           ("feats", "gme_feats.pt",              None),
    "eva02_pre":     ("score", "eva02_pre_score.pt",        None),
    "beit3_v2":      ("score", "beit3_v2_score.pt",         None),
    "beit3_helip":   ("score", "beit3_helip_score.pt",      None),
    "metaclip_v1":   ("score", "metaclip_v1_score.pt",      None),
    "siglip_maxsim": ("score", "siglip_maxsim_score.pt",    None),
}
unknown = [k for k in FLAT if k not in LOADERS]
if unknown:
    raise SystemExit(f"[build_base] keys not in the flat member set: {unknown} — check that this is a base.flat weight set.")
acc = None
for k, w in FLAT.items():
    if not w:
        continue
    typ, fn, views = LOADERS[k]
    fp = f"{MEM}/{fn}"
    s = tta(fp, views) if typ == "tta" else feats(fp) if typ == "feats" else score(fp)
    acc = float(w) * s if acc is None else acc + float(w) * s
base = mmnorm(acc)

torch.save(base.cpu(), OUT)
print(f"[S1] base built from {sum(1 for w in FLAT.values() if w)} flat members shape={tuple(base.shape)} -> {OUT}")
