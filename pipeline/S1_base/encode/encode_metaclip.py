"""MetaCLIP (v1) ViT-L-14-worldwide-xlmv fine-tuned ckpt -> score matrix (scoring only).

Produces the score for the build_base member `metaclip_v1` (weight 0.15), over a
36,773-image gallery and 1,978 queries.
Output: metaclip_v1_score.pt, the file name read by build_base LOADERS['metaclip_v1'].
Training: `../../train/encoders/metaclip_v1/train.py`.

Usage:
  python encode_metaclip.py --checkpoint assets/model/encoder/metaclip_v1/epoch_4.pt
"""
from __future__ import annotations
import argparse
import json
import os

import sys
import open_clip

import torch
from PIL import Image
from tqdm import tqdm
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import sys as _sys                                              # shared idempotent-skip helper
_sys.path.insert(0, os.path.join(_REPO, "pipeline"))
from utils.artifacts import skip_if_exists                      # noqa: E402


# Register the custom open_clip config (ViT-L-14-worldwide-xlmv) from the in-repo copy rather than
# relying on the venv's model_configs.
sys.path.insert(0, os.path.join(_REPO, "assets", "model", "vlm_models", "MetaCLIP-L14-worldwide"))
import register as _oc_register  # noqa: E402,F401

PAB_TEST = os.environ.get("PAB_TEST", f"{_REPO}/assets/data/raw/pab_test")
GALLERY_DIR = os.environ.get("GALLERY", f"{PAB_TEST}/gallery")
QUERY_FILE  = os.environ.get("QUERY_TEXT", f"{PAB_TEST}/query_text.json")
SCORE_DIR   = os.environ.get("SCORE_DIR", os.environ.get("S1_MEMBERS", f"{_REPO}/assets/cache/s1_base/members"))

MODEL_NAME   = "ViT-L-14-worldwide-xlmv"  # custom config (vocab 901629, xlm-v-base), same as train.py
PRETRAINED   = os.environ.get("VLM_MODELS", f"{_REPO}/assets/model/vlm_models") + "/MetaCLIP-L14-worldwide/l14_worldwide.pt"
DEFAULT_CKPT = os.environ.get("METACLIP_CKPT", f"{_REPO}/assets/model/encoder/metaclip_v1/epoch_4.pt")  # deployed checkpoint

BATCH_SIZE   = 128


def load_queries(path: str) -> list[str]:
    captions = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                captions.append(json.loads(line)["caption"])
    print(f"Queries loaded: {len(captions)}")
    return captions


@torch.no_grad()
def encode_images(paths, model, preprocess, device):
    feats = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Images"):
        batch = [preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+BATCH_SIZE]]
        batch = torch.stack(batch).to(device)
        with torch.amp.autocast("cuda"):
            f = model.encode_image(batch)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def encode_texts(captions, model, tokenizer, device):
    feats = []
    for i in tqdm(range(0, len(captions), BATCH_SIZE), desc="Texts"):
        tokens = tokenizer(captions[i:i+BATCH_SIZE]).to(device)
        with torch.amp.autocast("cuda"):
            f = model.encode_text(tokens)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu())
    return torch.cat(feats, dim=0)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {MODEL_NAME} (no pretrained yet)")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=None
    )
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    # Load the base weights first; this is the fallback when no fine-tuned checkpoint is given.
    if args.checkpoint is None or args.checkpoint == "":
        print(f"Loading base weight: {PRETRAINED}")
        state = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
        sd = state.get("state_dict", state)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

    ckpt_path = args.checkpoint
    if ckpt_path:
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected:
            print(f"  unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")

    model = model.to(device).eval()

    gallery_files = sorted(os.listdir(GALLERY_DIR))
    gallery_paths = [os.path.join(GALLERY_DIR, f) for f in gallery_files]
    print(f"Gallery: {len(gallery_files)} images")

    captions = load_queries(QUERY_FILE)

    print("Encoding gallery images...")
    img_feats = encode_images(gallery_paths, model, preprocess, device)

    print("Encoding query texts...")
    txt_feats = encode_texts(captions, model, tokenizer, device)

    sims = txt_feats @ img_feats.t()
    print(f"Score matrix: {tuple(sims.shape)}")

    tag = args.tag
    score_dir = f"{_REPO}/assets/cache_rep/s1_base/members" if args.rep else SCORE_DIR
    os.makedirs(score_dir, exist_ok=True)
    score_path = os.path.join(score_dir, f"{tag}_score.pt")   # build_base LOADERS['metaclip_v1'] = metaclip_v1_score.pt
    torch.save(sims, score_path)
    print(f"Saved score: {score_path}  {tuple(sims.shape)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--tag", type=str, default="metaclip_v1",
                        help="output file name {tag}_score.pt (default metaclip_v1 -> metaclip_v1_score.pt)")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild even if the artifact exists (default: skip)")
    parser.add_argument("--rep", action="store_true",
                        help="reproduction encoding with model_rep weights -> cache_rep/s1_base/members")
    args = parser.parse_args()
    if args.rep and args.checkpoint == DEFAULT_CKPT:          # rep: deployed weight source
        args.checkpoint = f"{_REPO}/assets/model_rep/encoder/metaclip_v1/epoch_4.pt"
    if args.checkpoint == "":
        args.checkpoint = None
    _sd = f"{_REPO}/assets/cache_rep/s1_base/members" if args.rep else SCORE_DIR
    skip_if_exists(os.path.join(_sd, f"{args.tag}_score.pt"), args.overwrite)
    main(args)
