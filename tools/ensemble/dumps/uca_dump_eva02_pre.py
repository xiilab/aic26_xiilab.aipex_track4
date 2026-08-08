"""UCA eva02_pre feats dump — EVA02-L-14 / merged2b_s4b_b131k zero-shot (open_clip).
Mirrors the backbone loading of encode_openclip_zs.py (eva02_pre member = EVA02-L-14 at 224px)
with UCA as the data source. Saves {G, Q} feats rather than a [Q, G] score matrix, so the
combine step normalizes it the same way as the other members.

env: .venv_beit3eval  (open_clip 3.3.0)
run: CUDA_VISIBLE_DEVICES=6 $PY_beit3eval uca_dump_eva02_pre.py
out: $ARTIFACTS/uca_eva02_pre_feats.pt  {G, Q, gal_paths, Q_idx}
"""
import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")   # overridable from the environment
import torch, torch.nn.functional as F
from PIL import Image

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repo root
T4 = os.environ.get("TRACK4", f"{_REPO}/assets/data/benches")
POOL = f"{T4}/uca_pool.pt"
MODEL_NAME = "EVA02-L-14"
PRETRAINED = "merged2b_s4b_b131k"
BATCH = 128
OUT = f"{T4}/uca_eva02_pre_feats.pt"
dev = "cuda:0"

import open_clip
def log(*x): print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)

# ---- UCA data ----
_p = torch.load(POOL, map_location="cpu", weights_only=False)
gal_paths = list(_p["gal_paths"]); caps = list(_p["caps"])
log(f"[uca-eva02_pre] gallery={len(gal_paths)} queries={len(caps)}")

log(f"[load] {MODEL_NAME} / {PRETRAINED} ...")
model, _, pp = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
tok = open_clip.get_tokenizer(MODEL_NAME)
model = model.to(dev).eval()


@torch.no_grad()
def enc_img():
    out = []
    for i in range(0, len(gal_paths), BATCH):
        ims = torch.stack([pp(Image.open(p).convert("RGB")) for p in gal_paths[i:i + BATCH]]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f_ = model.encode_image(ims)
        out.append(F.normalize(f_.float(), dim=-1).cpu())
        if (i // BATCH) % 30 == 0: log(f"  img {i}/{len(gal_paths)}")
    return torch.cat(out)


@torch.no_grad()
def enc_txt():
    out = []
    for i in range(0, len(caps), 256):
        tk = tok(caps[i:i + 256]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f_ = model.encode_text(tk)
        out.append(F.normalize(f_.float(), dim=-1).cpu())
    return torch.cat(out)


t0 = time.time(); G = enc_img(); log(f"img done {tuple(G.shape)} {time.time()-t0:.0f}s")
Q = enc_txt(); log(f"txt done {tuple(Q.shape)}")

torch.save({"G": G, "Q": Q, "gal_paths": gal_paths, "Q_idx": list(range(len(caps)))}, OUT)
log(f"[save] {OUT}  G={tuple(G.shape)} Q={tuple(Q.shape)}")
