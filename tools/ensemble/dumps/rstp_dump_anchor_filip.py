"""RSTP v20 TTA-views dump (base/hflip/z080) — deployed_base anchor member.
Mirrors encode_anchor_filip.py (v20 l512, ep10) but encodes only the 3 views used by
build_base's deployed_base, with RSTP as the data source. The v20 train module is imported and
its checkpoint loaded read-only.

env: .venv_llm2clip  (SigLIP2-L512 DoRA inference)
run: CUDA_VISIBLE_DEVICES=6 $PY_llm2clip rstp_dump_anchor_filip.py
out: $ARTIFACTS/rstp_anchor_filip_feats.pt  {img:{base,hflip,z080}, txt:{base}, gal_paths, Q_idx}
"""
import os, importlib.util
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")   # overridable from the environment
import torch, torch.nn.functional as F
import torchvision.transforms.v2 as Tv2
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root; all default paths are relative to it

T4 = os.environ.get("TRACK4", f"{_REPO}/assets/data/benches")
POOL = f"{T4}/rstp_pool.pt"
MODULE_PATH = f"{_REPO}/train/encoders/anchor_filip_all/train.py"                                  # read-only import
BEST_CKPT = "ep10"
OUT = f"{T4}/rstp_anchor_filip_feats.pt"
GALLERY_BATCH = 48
VIEWS = ("base", "hflip", "z080")
device = "cuda:0"


def import_module_by_path(path):
    spec = importlib.util.spec_from_file_location("_v20mod", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


# ---- RSTP data ----
_p = torch.load(POOL, map_location="cpu", weights_only=False)
gal_paths = list(_p["gal_paths"]); caps = list(_p["caps"])
queries = [{"query_index": str(j), "caption": c, "change": ""} for j, c in enumerate(caps)]
print(f"[rstp-v20] gallery={len(gal_paths)} queries={len(caps)}", flush=True)

mod = import_module_by_path(MODULE_PATH); mod.EVAL_BATCH_SIZE = GALLERY_BATCH
S = mod.IMAGE_SIZE
ckpt_dir = os.environ.get("ENCODER_CKPT", f"{_REPO}/assets/model/encoder/anchor_filip")
print(f"[rstp-v20] ckpt={BEST_CKPT} IMG={S}", flush=True)

norm = [Tv2.ToDtype(torch.float32, scale=True), Tv2.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
def zoom(r): return [Tv2.CenterCrop(int(round(S * r))), Tv2.Resize(S, antialias=True)]
tfs = {
    "base":  Tv2.Compose(norm),
    "hflip": Tv2.Compose([Tv2.RandomHorizontalFlip(1.0), *norm]),
    "z080":  Tv2.Compose([*zoom(0.80), *norm]),
}

model, tok = mod.load_for_inference(mod.MODEL_NAME, ckpt_dir, device=device)
img_views = {}
for name in VIEWS:
    G, _ = mod.encode_test_gallery(model, gal_paths, tfs[name], device=device, batch_size=GALLERY_BATCH)
    img_views[name] = G.float().cpu()
    print(f"[img-view {name}] {tuple(img_views[name].shape)}", flush=True)
Q, Q_idx, _, _ = mod.encode_test_queries(model, queries, tok, device=device)

torch.save({"img": img_views, "txt": {"base": Q.float().cpu()},
            "gal_paths": gal_paths, "Q_idx": list(range(len(caps)))}, OUT)
print(f"[save] {OUT}  views={list(img_views)} txt=base", flush=True)
