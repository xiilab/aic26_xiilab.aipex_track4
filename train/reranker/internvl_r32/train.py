#!/usr/bin/env python3
"""internvl_r32/train — LoRA fine-tuning of InternVL3.5-30B as the main reranker.

Pairwise Bradley-Terry: minimise -logsigmoid so that score(chosen) > score(rejected).
score = logit(yes) - logit(no), i.e. **the same head as the zero-shot scorer**, so the scale is
unchanged by fine-tuning and the result is directly compatible with the deployment cache. The
transformers path is used rather than vLLM, which deadlocks here.

Input  --data = the merge of A_rescue (`build_rescue_pairs.py`) and B_antibreak
       (`build_antibreak_pairs.py`).
Run    Requires the **track4_vllm** conda env — bash requirements/setup_conda_envs.sh --only vllm,
       then PY_VLLM=$(conda info --base)/envs/track4_vllm/bin/python. On torch 2.8 the MoE layers
       dispatch to `torch._grouped_mm` (Hopper only), so every forward raises RuntimeError; the
       `except: continue` below swallows it and the run ends silently with 0 steps.
        CUDA_VISIBLE_DEVICES=6 $PY_VLLM -u train.py \
            --data assets/data/mining/dpo_train.jsonl --lora-r 32 --lr 1e-4 --grad-accum 8 --save-every 500
        smoke test: --max-steps 20 --heldout 500   (compares pair-acc before and after training)
"""
import argparse, json, os, time, random
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # repository root — all default paths are relative to it
T4 = os.environ.get("TRACK4", f"{_REPO}/assets/data/mining")
TRAIN_ROOT = os.environ.get("PAB_JPG", f"{_REPO}/assets/data/raw/pab_train/train_jpg_512")
PROMPT = ("You are judging whether an image matches a text description of a person.\n"
          "Description: \"{cap}\"\n"
          "Does this image show EXACTLY that person and situation — matching gender, clothing "
          "(color/type), the action/behavior, and the scene? Answer a single word: yes or no.")
YES = ["yes", "Yes", " yes", " Yes", "YES"]; NO = ["no", "No", " no", " No", "NO"]

def remap(p):
    parts = p.split("/")
    if parts and parts[0] == "train": parts = parts[1:]
    m = int(parts[0].split("_")[1])
    return os.path.join(TRAIN_ROOT, f"Part {m//8+1}", *parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("VLM_MODELS", f"{_REPO}/assets/model/vlm_models") + "/InternVL3_5-30B-A3B-HF")
    ap.add_argument("--data", default=f"{T4}/dpo_train.jsonl")
    ap.add_argument("--out", default=os.environ.get("OUTPUT_DIR", f"{_REPO}/assets/runs") + "/internvl_rerank_lora")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--dora", action="store_true", help="train with DoRA (use_dora=True)")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0, help=">0 runs a smoke test")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--heldout", type=int, default=0, help=">0 holds out N pairs to compare pair-acc before and after training")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    ap.add_argument("--run-note", default="", help="run directory suffix (default: none → <out>/{step|final})")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.run_note:
        args.out = f"{args.out}_{args.run_note}"
    os.makedirs(args.out, exist_ok=True)
    print(f"[run] gpu={args.gpu} · out={args.out}", flush=True)

    import torch, torch.nn.functional as F
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model

    proc = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    tok = proc.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    # LoRA targets the language-side attention projections (InternVL's inner Qwen3-MoE). Vision is untouched.
    lora = LoraConfig(r=args.lora_r, lora_alpha=2*args.lora_r, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM",
                      use_dora=args.dora)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    dev = model.device

    def ids_for(ws):
        s = set()
        for w in ws:
            e = tok.encode(w, add_special_tokens=False)
            if len(e) == 1: s.add(e[0])
        return list(s)
    yes_ids, no_ids = ids_for(YES), ids_for(NO)
    assert yes_ids and no_ids

    def score(cap, imgpath):
        img = Image.open(imgpath).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                 {"type": "text", "text": PROMPT.format(cap=cap)}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(dev)
        logit = model(**inp).logits[0, -1]           # differentiable through the LoRA weights
        ly = torch.stack([logit[t] for t in yes_ids]).max()
        ln = torch.stack([logit[t] for t in no_ids]).max()
        return ly - ln

    data = [json.loads(l) for l in open(args.data)]
    random.seed(0); random.shuffle(data)
    held = data[:args.heldout] if args.heldout else []
    if args.heldout:
        data = data[args.heldout:]

    @torch.no_grad()
    def eval_held(pairs, tag):
        if not pairs:
            return None
        was_train = model.training; model.eval()
        good = 0; msum = 0.0; n = 0
        for d in pairs:
            try:
                m = (score(d["query"], remap(d["chosen"])) - score(d["query"], remap(d["rejected"]))).item()
            except Exception:
                continue
            n += 1; msum += m; good += int(m > 0)
        if was_train: model.train()
        r = dict(n=n, acc=good / max(n, 1), mean_margin=msum / max(n, 1))
        print(f"[heldout {tag}] n={r['n']} pair_acc={r['acc']:.3f} mean_margin(ch-rej)={r['mean_margin']:+.4f}", flush=True)
        return r

    h0 = eval_held(held, "BEFORE")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    model.train()

    step = 0; micro = 0; t0 = time.time(); loss_acc = 0.0; opt.zero_grad()
    total = args.max_steps * args.grad_accum if args.max_steps > 0 else len(data) * args.epochs
    for ep in range(args.epochs):
        for d in data:
            try:
                sc = score(d["query"], remap(d["chosen"]))
                sr = score(d["query"], remap(d["rejected"]))
                loss = -F.logsigmoid(sc - sr) / args.grad_accum
                if not torch.isfinite(loss):
                    print(f"[skip] non-finite loss (sc={sc.item():.3f} sr={sr.item():.3f})", flush=True); continue
                loss.backward(); loss_acc += loss.item()
            except Exception as e:                   # on torch 2.8 every forward lands here, leaving 0 steps
                print(f"[skip] {e}", flush=True); continue
            micro += 1
            if micro % args.grad_accum == 0:
                gnorm = float(torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0))
                if not (gnorm == gnorm) or gnorm == float("inf"):
                    print(f"[WARN] grad norm is nan/inf: {gnorm}", flush=True)
                opt.step(); opt.zero_grad(); step += 1
                if step % 5 == 0:
                    el = time.time() - t0
                    print(f"[step {step}] loss={loss_acc:.4f} margin={(sc-sr).item():+.3f} "
                          f"gnorm={gnorm:.3f} {el:.0f}s ({el/step:.1f}s/step)", flush=True)
                loss_acc = 0.0
                if args.save_every and step % args.save_every == 0:
                    model.save_pretrained(f"{args.out}/step{step}"); print(f"[save] step{step}", flush=True)
                if args.max_steps and step >= args.max_steps:
                    h1 = eval_held(held, "AFTER")
                    if h0 and h1:
                        print(f"[heldout Δ] pair_acc {h0['acc']:.3f}→{h1['acc']:.3f} ({h1['acc']-h0['acc']:+.3f}) | "
                              f"mean_margin {h0['mean_margin']:+.4f}→{h1['mean_margin']:+.4f} "
                              f"({h1['mean_margin']-h0['mean_margin']:+.4f})", flush=True)
                    model.save_pretrained(f"{args.out}/final"); print(f"[done smoke] {step} steps, {time.time()-t0:.0f}s", flush=True); return
            if micro >= total and args.max_steps == 0: break
    model.save_pretrained(f"{args.out}/final")
    print(f"[done] {step} steps", flush=True)

if __name__ == "__main__":
    main()
