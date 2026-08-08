"""
MSR-axes — caption variants defined as coordinates over 8 control axes. Every axis and level is
defined either from prior work or as a quantity relative to the source caption.

  coverage      how many of the source facts to realise; search queries are partial.
  compression   length ratio. Not an absolute token count but a **ratio to the source**, following
                the sentence compression / simplification convention. At run time it is converted
                into an absolute budget as `source word count x ratio`, because the ratio wording
                alone is not obeyed.
  syntax        sententiality. sentential vs telegraphic (headlinese: articles, copulas and
                pronouns deleted, one facet per line). **Orthogonal** to compression: a short text
                can be a full sentence and a long one can be telegraphic.
  register      Biber (1988) multi-dimensional analysis: involved vs informational, and narrative.
                The involved level uses that paper's involved-production feature list. The g-dropping
                in the informal level is Labov (1966) (ING).
  epistemic     certainty marking (assertive / hedged / alternative), a property of an incomplete
                information need.
  specificity   term discriminativeness (IDF), the IR standard.
  paraphrase    kind of transformation. The two top branches of the Bhagat & Hovy (2013) typology —
                lexical (synonym substitution) vs structural (clause reordering, active/passive
                alternation). Neither changes meaning, so both are independent of the other axes.
  stance        speaker role. The recall-based setting follows the Belkin et al. (1982) ASK
                hypothesis: a query comes from an incomplete information need, not from a complete
                description of the scene.
────────────────────────────────────────────────────────────────────────────
Relation to LaCLIP
────────────────────────────────────────────────────────────────────────────
  · Text-side augmentation framing: caption rewriting as the counterpart of image augmentation,
    with the axes as the augmentation operators.
  · Multiple variants per image, one sampled at random each epoch, as in LaCLIP's training loop.
  · **Rewriter-source diversification**: LaCLIP mixes four sources. `--rewriter` names a model
    endpoint and records that value as a suffix in the style column, making the source itself an axis.
  · **In-context prompting**: exemplars are allowed but seeded **only from train-split captions**
    (`--exemplars`).
  · Meaning-preservation constraints, carried in SYSTEM.

The rewriter used here is Qwen3-VL-30B-A3B-Instruct, and it must be **vision-capable**: with
--img-max-side the source image is attached as a base64 data URI and the prompt switches to
SYSTEM_IMG, so a text-only server silently produces different captions.

The rewriter is not bundled with the repository — it is needed only for generation. Download it under
`assets/model/vlm_models/`, or point `VLM_MODELS` at wherever it lives.

  # 1) serve (env: track4_vllm)
  CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server \
    --model $VLM_MODELS/Qwen3-VL-30B-A3B-Instruct \
    --served-model-name qwen3vl-30b --port 8000 --tensor-parallel-size 2
  # 2) generate (--endpoint takes the base up to /v1; model_id is read from /v1/models)
  python msr_axes_generate.py --endpoint http://127.0.0.1:8000 [--limit N]
  python msr_axes_generate.py --dry-run          # print the prompts only
"""
from __future__ import annotations
import argparse, asyncio, csv, json, os, sys, time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
ANNOT_DIR = os.environ.get("PAB_ANNOT", f"{_REPO}/assets/data/raw/pab_train/annotation/train")
OUT_DIR   = os.environ.get("OUT_DIR", f"{_REPO}/assets/data/raw/recaption")
OUT_NAME  = os.environ.get("RECAP_NAME", "train_msr_v1.csv")   # same name the trainer reads

# ── Axis definitions (8 axes); one instruction fragment per level.
#    No measured numbers, observed vocabulary, or statistics derived from the evaluation set —
#    grammatical and sociolinguistic categories only.
AXES: dict[str, dict[str, str]] = {
"compression": {
  "x0.3": "Compress drastically, to roughly a third of the source length.",
  "x0.5": "Compress heavily, to roughly half of the source length.",
  "x0.6": "Compress substantially, to roughly two thirds of the source length.",
  "x0.8": "Compress moderately, to roughly four fifths of the source length.",
  "x0.9": "Compress a little, to roughly nine tenths of the source length.",
  "x1.0": "Keep the length close to the source.",
  "x1.1": "Expand slightly, to roughly one and a tenth times the source length.",
  "x1.3": "Expand moderately, to roughly one and a third times the source length.",
  "x1.5": "Expand noticeably, to roughly one and a half times the source length.",
},
"coverage": {
  "single":     "Mention only ONE facet of the scene and omit the rest.",
  "salient":    "Mention only the most salient facts; omit incidental detail.",
  "broad":      "Mention most of the facts present in the source.",
  "exhaustive": "Mention every fact present in the source, omitting nothing.",
},
"register": {
  "informational": "Write in a formal, informational register: precise diction, full "
                   "sentences, no contractions or colloquialisms.",
  "neutral":       "Write in a neutral descriptive register.",
  "informal":      "Write in an informal written register, as in a relaxed message or post. "
                   "Contract every auxiliary and copula that English allows to be contracted "
                   "(it is -> it's, he is -> he's, they are -> they're, is not -> isn't, "
                   "cannot -> can't). Choose everyday general-purpose words over formal or "
                   "technical ones, and colloquial spellings of common words are fine, "
                   "including dropping the final g of -ing forms where the register invites "
                   "it. The result must read as clearly informal, never neutral. Written "
                   "rather than transcribed: no filler sounds.",
  "involved":      "Write in an involved, spoken register, as if describing the scene aloud. "
                   "Use the features of unplanned speech: contract every auxiliary and copula "
                   "that English allows; open or link clauses with spoken discourse particles "
                   "and coordinators; use general emphatics and amplifiers; use the pro-verb "
                   "'do' and demonstrative pronouns in place of repeated nouns; leave "
                   "prepositions in final position where that is natural; drop the "
                   "complementiser 'that' where speech would. Everyday words only. The "
                   "result must read as clearly spoken, never neutral or written.",
  "narrative":     "Write in a literary, narrative register, as a passage of descriptive prose. "
                   "Evocative verbs and varied rhythm are welcome; do not invent facts.",
},
"epistemic": {
  "assertive":   "State everything directly, as fact.",
  "hedged":      "Mark uncertainty where a detail is not fully determinate "
                 "(e.g. seems, looks like, I think, maybe).",
  "alternative": "Mark uncertainty and, for at least one detail, offer an alternative "
                 "possibility instead of committing to a single reading.",
},
"specificity": {
  "free":     "",   # no constraint
  "distinct": "Prioritise the most distinctive, rarely-occurring details over generic ones.",
  "generic":  "Use only broad, commonly-occurring descriptors; avoid rare or highly specific terms.",
},
"syntax": {
  "sentential":  "",   # default: full sentences
  "telegraphic": "Write in telegraphic style (headlinese), not in sentences. Delete all "
                 "articles, all forms of 'be', and all pronouns. Keep only content words: "
                 "nouns, adjectives, colours, and verbs. Put ONE facet on each line, in this "
                 "order where applicable — who and what they are doing and where / what they "
                 "are wearing / notable objects and their attributes / background. Separate "
                 "lines with a newline. No full stops at the end of lines. Use between 3 and 5 "
                 "lines in total, never more. Do not leave blank lines between them.",
},
"paraphrase": {
  "free":       "",   # no constraint
  "phrasal":    "Re-phrase at the phrase level rather than word by word: swap multiword "
                "expressions for single words and single words for multiword expressions, "
                "turn verbs into nouns and nouns into verbs, and replace light-verb "
                "constructions with plain verbs or the reverse. Keep the clause order and "
                "keep individual content words where a phrase-level change is not available.",
  "diathesis":  "Re-organise by changing argument structure rather than clause order: "
                "alternate active and passive, swap which participant is the subject, and "
                "use converse constructions (X holds Y / Y is held by X, X above Y / Y "
                "below X). Reuse the source's own words; do not substitute synonyms.",
  "lexical":    "Re-word: replace many of the content words with synonyms or near-synonyms "
                "and vary the verbs. Leave concrete particulars exactly as they are — "
                "colours, counts, and the names of objects and garments. Keep the clause "
                "structure of the source essentially as it is. Re-word the opening too, "
                "not only the middle.",
  "structural": "Re-organise only. Reuse the source's own words — do not substitute "
                "synonyms. Every content word of the source that can be carried over must be "
                "carried over verbatim; nothing but the arrangement may change. Reorder the clauses, "
                "front something the source leaves for last, move modifiers, and alternate "
                "between active and passive where that is natural. Begin with a different "
                "element of the scene than the source begins with — the setting, the "
                "background, an object, or an action rather than whichever one the source "
                "starts from.",
  "both":       "Both re-word with synonyms and re-organise the clause order.",
},
"stance": {
  "observer":     "Write from the standpoint of a neutral third-person observer.",
  "photographer": "Write from the standpoint of the person who took the picture, "
                  "referring to framing and what is in view where natural.",
  "participant":  "Write from the standpoint of someone present in the scene.",
  "recaller":     "Write as someone recalling the scene from memory in order to search for it "
                  "later: partial, out of order, and unsure of some wording. Do not narrate the "
                  "act of remembering — produce the recalled description itself.",
},
}

# ── Two preset tables ──────────────────────────────────────────────────────
# `msr`  (default) the 11 styles of the shipped CSVs (train_msr_v{1,2,3}.csv). The trainers and the
#                  held-out scorer hardcode these keys, so a regenerated CSV drops straight in.
# `axes`           a table redesigned to cover the 8-axis space evenly: it adds two recall presets
#                  and drops the finer paraphrase levels.
# _validate() checks both tables for axis definitions, invalid combinations, and uniqueness of
# coordinates, slugs and instructions.
#
# TEMPERATURES_MSR mirrors the axes preset sharing the same coordinates, so regeneration does not
# reproduce the shipped CSVs byte for byte.
PRESETS_MSR: dict[str, dict[str, str]] = {
"p01_lexical":     {"compression":"x1.0","coverage":"broad", "register":"neutral",
                    "epistemic":"assertive","specificity":"free","stance":"observer",
                    "syntax":"sentential", "paraphrase":"lexical"},
"p02_phrasal":     {"compression":"x1.0","coverage":"broad", "register":"neutral",
                    "epistemic":"assertive","specificity":"free","stance":"observer",
                    "syntax":"sentential", "paraphrase":"phrasal"},
"p03_clausal":     {"compression":"x1.0","coverage":"exhaustive","register":"neutral",
                    "epistemic":"assertive","specificity":"free","stance":"observer",
                    "syntax":"sentential", "paraphrase":"structural"},
"p04_diathesis":   {"compression":"x1.0","coverage":"exhaustive","register":"neutral",
                    "epistemic":"assertive","specificity":"free","stance":"observer",
                    "syntax":"sentential", "paraphrase":"diathesis"},
"p05_involved":    {"compression":"x1.0","coverage":"broad", "register":"involved",
                    "epistemic":"assertive","specificity":"free","stance":"observer",
                    "syntax":"sentential", "paraphrase":"free"},
"p06_informal":    {"compression":"x1.0","coverage":"broad", "register":"informal",
                    "epistemic":"assertive","specificity":"free","stance":"observer",
                    "syntax":"sentential", "paraphrase":"free"},
"p07_telegraphic": {"compression":"x0.5","coverage":"broad", "register":"neutral",
                    "epistemic":"assertive","specificity":"distinct","stance":"observer",
                    "syntax":"telegraphic", "paraphrase":"free"},
"p08_compact":     {"compression":"x0.8","coverage":"broad", "register":"informational",
                    "epistemic":"assertive","specificity":"free","stance":"observer",
                    "syntax":"sentential", "paraphrase":"free"},
"p09_narrative":   {"compression":"x1.1","coverage":"broad", "register":"narrative",
                    "epistemic":"assertive","specificity":"free","stance":"observer",
                    "syntax":"sentential", "paraphrase":"free"},
"p10_formal":      {"compression":"x1.1","coverage":"exhaustive","register":"informational",
                    "epistemic":"assertive","specificity":"distinct","stance":"observer",
                    "syntax":"sentential", "paraphrase":"free"},
"p11_compound":    {"compression":"x1.0","coverage":"broad", "register":"informal",
                    "epistemic":"assertive","specificity":"free","stance":"observer",
                    "syntax":"sentential", "paraphrase":"lexical"},
}

# ── axes table, 11 presets (+ original = 12 rows per image) ────────────────
#    Hand-designed to cover each axis evenly while keeping every preset linguistically natural.
PRESETS_AXES: dict[str, dict[str, str]] = {
"p01_keyword":   {"compression":"x0.5","coverage":"broad", "register":"neutral",
                  "epistemic":"assertive","specificity":"distinct","stance":"observer",
                  "syntax":"telegraphic", "paraphrase":"free"},
"p02_terse":     {"compression":"x0.6","coverage":"salient", "register":"neutral",
                  "epistemic":"assertive","specificity":"free","stance":"observer",
                  "syntax":"sentential", "paraphrase":"free"},
"p03_compact":   {"compression":"x0.8","coverage":"broad", "register":"informational",
                  "epistemic":"assertive","specificity":"free","stance":"observer",
                  "syntax":"sentential", "paraphrase":"free"},
"p04_lexvar":    {"compression":"x1.0","coverage":"broad", "register":"neutral",
                  "epistemic":"assertive","specificity":"free","stance":"observer",
                  "syntax":"sentential", "paraphrase":"lexical"},
"p05_restruct":  {"compression":"x1.0","coverage":"exhaustive","register":"neutral",
                  "epistemic":"assertive","specificity":"free","stance":"photographer",
                  "syntax":"sentential", "paraphrase":"structural"},
"p06_informal":  {"compression":"x1.0","coverage":"broad", "register":"informal",
                  "epistemic":"assertive","specificity":"free","stance":"observer",
                  "syntax":"sentential", "paraphrase":"free"},
"p07_oral":      {"compression":"x1.0","coverage":"broad", "register":"involved",
                  "epistemic":"assertive","specificity":"free","stance":"observer",
                  "syntax":"sentential", "paraphrase":"free"},
"p08_formal":    {"compression":"x1.1","coverage":"exhaustive","register":"informational",
                  "epistemic":"assertive","specificity":"distinct","stance":"observer",
                  "syntax":"sentential", "paraphrase":"free"},
"p09_narrative": {"compression":"x1.1","coverage":"broad", "register":"narrative",
                  "epistemic":"assertive","specificity":"free","stance":"observer",
                  "syntax":"sentential", "paraphrase":"free"},
"p10_recall":    {"compression":"x0.6","coverage":"single", "register":"involved",
                  "epistemic":"hedged","specificity":"free","stance":"recaller",
                  "syntax":"sentential", "paraphrase":"free"},
"p11_recall_alt":{"compression":"x0.8","coverage":"salient", "register":"informal",
                  "epistemic":"alternative","specificity":"distinct","stance":"recaller",
                  "syntax":"sentential", "paraphrase":"free"},
}

# For hard negatives — **never mixed into the positive corpus**; reached only through --perturb.
PERTURB = {
  "fact_flip":        "Change exactly one stated attribute to a different plausible value.",
  "entity_swap":      "Replace exactly one entity with a different plausible entity.",
  "relation_reverse": "Reverse exactly one spatial or participant relation.",
  "count_change":     "Change exactly one stated count to a different plausible count.",
  "entity_add":       "Add exactly one plausible entity that is not in the source.",
}

TEMPERATURES_AXES = {  # higher only for the presets that need latitude
  "p01_keyword":0.4,"p02_terse":0.4,"p03_compact":0.5,"p04_lexvar":0.5,"p05_restruct":0.5,
  "p06_informal":0.8,"p07_oral":0.8,"p08_formal":0.5,"p09_narrative":0.8,
  "p10_recall":0.9,"p11_recall_alt":0.9,
}
TEMPERATURES_MSR = {  # follows the axes preset at the same coordinates (see the note above)
  "p01_lexical":0.5,"p02_phrasal":0.5,"p03_clausal":0.5,"p04_diathesis":0.5,
  "p05_involved":0.8,"p06_informal":0.8,"p07_telegraphic":0.4,"p08_compact":0.5,
  "p09_narrative":0.8,"p10_formal":0.5,"p11_compound":0.8,
}

# table -> (presets, temperatures, style label of the original row)
TABLES = {
  "msr":  (PRESETS_MSR,  TEMPERATURES_MSR,  "p00_original"),
  "axes": (PRESETS_AXES, TEMPERATURES_AXES, "original"),
}
PRESET_SET = os.environ.get("PRESET_SET", "msr")      # overridden by --preset-set
PRESETS, TEMPERATURES, ORIGINAL_STYLE = TABLES[PRESET_SET]

SYSTEM = """You are a caption-rewriting assistant. You will receive an English image caption and rewrite it under a set of stated constraints.

Rules that always apply:
- Never invent facts. Every entity, count, action, attribute, and spatial relation you state must be supported by the source caption.
- Omission is allowed when the constraints call for it; fabrication is not.
- Output only the rewritten caption. No preamble, no quotes, no explanation, no mention of the constraints."""

# SYSTEM for image conditioning: the evidence base widens from the caption to caption + image.
SYSTEM_IMG = """You are an image re-captioning assistant. You will receive an image and an English caption of that image, and produce a new caption under a set of stated constraints.

Rules that always apply:
- Ground every statement in what is visible in the image or stated in the caption. Never invent facts supported by neither.
- The caption may be incomplete. When the constraints ask for more detail, you may add visual detail that is clearly visible in the image but absent from the caption.
- When the image and the caption disagree, trust the image.
- Omission is allowed when the constraints call for it; fabrication is not.
- Output only the new caption. No preamble, no quotes, no explanation, no mention of the constraints."""

# compression level -> multiplier. The ratio wording alone is not obeyed, so it is multiplied by
# the source word count and injected as an **absolute cap**.
_RATIO = {k: float(k[1:]) for k in ("x0.3", "x0.5", "x0.6", "x0.8", "x0.9", "x1.0",
                                    "x1.1", "x1.3", "x1.5")}


def length_budget(cv: dict[str, str], src_words: int) -> tuple[int, int]:
    """(target word count, cap). The cap is 1.15x the target plus 4."""
    tgt = max(6, round(src_words * _RATIO[cv["compression"]]))
    return tgt, round(tgt * 1.15) + 4


def build_instruction(cv: dict[str, str], src_words: int | None = None) -> str:
    """control vector -> instruction: the axis fragments joined in order, with no exemplars."""
    parts = [AXES[a][cv[a]] for a in
             ("coverage", "compression", "syntax", "register", "epistemic",
              "specificity", "paraphrase", "stance")
             if AXES[a][cv[a]]]
    if src_words:
        tgt, cap = length_budget(cv, src_words)
        parts.append(f"Length: aim for about {tgt} words and do not exceed {cap} words. "
                     f"This limit overrides every other constraint, including any invitation "
                     f"to add detail from the image.")
    return "Rewrite the caption under all of the following constraints:\n" + \
           "\n".join(f"- {p}" for p in parts)


AXIS_CODES: dict[str, dict[str, str]] = {
  "compression": {k: "x" + k[1:].replace(".", "")   # x1.0 -> x10, so '.' stays free as the separator
                  for k in ("x0.3","x0.5","x0.6","x0.8","x0.9","x1.0","x1.1","x1.3","x1.5")},
  "coverage":    {"single":"one","salient":"sal","broad":"brd","exhaustive":"exh"},
  "syntax":      {"sentential":"sent","telegraphic":"tele"},
  "register":    {"informational":"infm","neutral":"neut","informal":"infl",
                  "involved":"invl","narrative":"narr"},
  "epistemic":   {"assertive":"asrt","hedged":"hedg","alternative":"altv"},
  "specificity": {"free":"spfr","distinct":"dstn","generic":"genr"},
  "paraphrase":  {"free":"phfr","phrasal":"phrs","diathesis":"diat","lexical":"lexi",
                  "structural":"strc","both":"both"},
  "stance":      {"observer":"obsv","photographer":"phtg","participant":"prtc",
                  "recaller":"recl"},
}
SLUG_ORDER = ("compression","coverage","syntax","register","epistemic","specificity",
              "paraphrase","stance")


def style_slug(cv: dict[str, str]) -> str:
    """8-axis coordinates -> style name, e.g. x0.8.sal.sent.infl.altv.dstn.phfr.recl"""
    return ".".join(AXIS_CODES[a][cv[a]] for a in SLUG_ORDER)


def slug_to_cv(slug: str) -> dict[str, str]:
    """style name -> 8-axis coordinates (a complete inverse)."""
    parts = slug.split("."); assert len(parts) == len(SLUG_ORDER), f"malformed slug: {slug}"
    cv = {}
    for a, code in zip(SLUG_ORDER, parts):
        inv = {v: k for k, v in AXIS_CODES[a].items()}
        assert code in inv, f"{a}: undefined code {code}"
        cv[a] = inv[code]
    return cv


_INVALID = [
    ("compression", "x0.3", "coverage",  "exhaustive"),
    ("compression", "x0.3", "coverage",  "broad"),
    ("compression", "x0.3", "epistemic", "alternative"),
    ("compression", "x0.3", "register",  "narrative"),
    ("syntax",      "telegraphic", "paraphrase", "structural"),
    ("syntax",      "telegraphic", "paraphrase", "diathesis"),  
    ("syntax",      "telegraphic", "paraphrase", "both"),
    ("compression", "x0.5", "coverage",  "exhaustive"),
    ("compression", "x0.6", "coverage",  "exhaustive"),
]


def _validate():
    for tag, (tbl, tmp, _orig) in TABLES.items():
        for name, cv in tbl.items():
            for a in AXES:
                assert a in cv, f"[{tag}] {name}: axis {a} missing"
                assert cv[a] in AXES[a], f"[{tag}] {name}: {a}={cv[a]} is not defined"
            for a1, l1, a2, l2 in _INVALID:
                assert not (cv.get(a1) == l1 and cv.get(a2) == l2), \
                    f"[{tag}] {name}: invalid combination {a1}={l1} × {a2}={l2}"
        assert len(tbl) == 11, f"[{tag}] {len(tbl)} presets (11 gives 12 rows per image including original)"
        assert set(tmp) == set(tbl), f"[{tag}] TEMPERATURES does not match the presets"

        seen = {}
        for name, cv in tbl.items():
            key = tuple(cv[a] for a in AXES)
            assert key not in seen, \
                f"[{tag}] coordinate collision: {name} and {seen[key]} share all 8 axes, so the axes alone cannot separate them"
            seen[key] = name
        # slugs must be unique and must round-trip
        slugs = {}
        for name, cv in tbl.items():
            sl = style_slug(cv)
            assert sl not in slugs, f"[{tag}] slug collision: {name} and {slugs[sl]} → {sl}"
            assert slug_to_cv(sl) == {a: cv[a] for a in AXES}, f"[{tag}] {name}: slug round-trip failed"
            slugs[sl] = name
        # instructions must differ too: an empty level lets two distinct coordinates share one instruction
        ins = {}
        for name, cv in tbl.items():
            t = build_instruction(cv)
            assert t not in ins, f"[{tag}] instruction collision: {name} and {ins[t]}"
            ins[t] = name


_validate()
PRESET_IDS = list(PRESETS)


def coord_to_style(cv: dict[str, str]) -> str:
    """8-axis coordinates -> style name. _validate guarantees uniqueness, so exactly one matches."""
    tbl = PRESETS
    key = tuple(cv[a] for a in AXES)
    for name, c in tbl.items():
        if tuple(c[a] for a in AXES) == key:
            return name
    raise KeyError(f"no style matches these coordinates: {cv}")


# In-context exemplars, allowed **only when seeded from the train split**.
# Format: {preset_id: [{"source": <train caption>, "rewrite": <rewrite for that preset>}, ...]}
EXEMPLARS: dict[str, list[dict]] = {}


def load_exemplars(path: str, annot_dir: str = ANNOT_DIR) -> None:
    """Load the exemplar file and **verify every source is a train caption**.

    A single sentence absent from train aborts the run, keeping evaluation-set text out of the prompts.
    """
    global EXEMPLARS
    data = json.load(open(path, encoding="utf-8"))
    train = {r["caption"].strip() for r in load_all_captions(annot_dir)}
    bad = [e["source"] for v in data.values() for e in v if e["source"].strip() not in train]
    if bad:
        raise SystemExit(f"[rejected] {len(bad)} exemplars whose source is not a train caption — e.g. {bad[0][:80]!r}")
    unknown = set(data) - set(PRESETS)
    if unknown:
        raise SystemExit(f"[rejected] unknown presets {sorted(unknown)}")
    EXEMPLARS = data
    n = sum(len(v) for v in data.values())
    print(f"[exemplars] loaded {n} · every source confirmed as a train caption", flush=True)



IMG_ROOT = os.environ.get("PAB_JPG", f"{_REPO}/assets/data/raw/pab_train/train_jpg_512")

def resolve_image(rel: str) -> str:
    rest = rel[6:] if rel.startswith("train/") else rel
    m = int(rest.split("/", 1)[0].removeprefix("imgs_"))
    return os.path.join(IMG_ROOT, f"Part {m // 8 + 1}", rest)


_IMG_CACHE: dict[str, str] = {}


def image_data_uri(path: str, max_side: int = 448) -> str:
    """Downscale the long side to max_side and return a base64 data URI. Cached: one image is reused by every preset."""
    key = f"{path}|{max_side}"
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    import base64, io
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))),
                       Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    uri = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    if len(_IMG_CACHE) > 4096:
        _IMG_CACHE.clear()
    _IMG_CACHE[key] = uri
    return uri


def build_messages(orig: str, preset_id: str, k_shot: int = 0,
                   img_uri: str | None = None) -> list[dict]:
    """k_shot=0 gives [system, user]; >0 prepends train-seeded exemplars.

    When img_uri is given, the final user turn becomes multimodal: [image, text].
    """
    sysmsg = SYSTEM_IMG if img_uri else SYSTEM
    msgs = [{"role": "system",
             "content": sysmsg + "\n\n"
             + build_instruction(PRESETS[preset_id], len(orig.split()))}]
    for ex in EXEMPLARS.get(preset_id, [])[:k_shot]:
        # exemplars stay text-only; adding their images would multiply the prompt by (k+1)
        msgs.append({"role": "user",      "content": f"Caption:\n{ex['source']}"})
        msgs.append({"role": "assistant", "content": ex["rewrite"]})
    if img_uri:
        msgs.append({"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": img_uri}},
            {"type": "text", "text": f"Caption:\n{orig}"}]})
    else:
        msgs.append({"role": "user", "content": f"Caption:\n{orig}"})
    return msgs


def load_all_captions(annot_dir):
    records = []
    for fn in sorted(os.listdir(annot_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(annot_dir, fn), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


_PUNCT = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',
                        "\u2026": "...", "\u00a0": " "})


def normalize(txt: str) -> str:
    """Map curly quotes and similar to ASCII, so one contraction does not split into two tokens."""
    return txt.translate(_PUNCT).strip()


def _safe_uri(rel: str, max_side: int):
    """Fall back to text-only when the image is missing, rather than failing the whole run."""
    try:
        return image_data_uri(resolve_image(rel), max_side)
    except Exception:
        return None


async def _call(session, endpoint, model_id, messages, temperature, max_tokens=400):
    payload = {"model": model_id, "messages": messages,
               "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens}
    import aiohttp
    async with session.post(f"{endpoint}/v1/chat/completions", json=payload,
                            timeout=aiohttp.ClientTimeout(total=120)) as r:
        return normalize((await r.json())["choices"][0]["message"]["content"])


async def gen_one(sem, session, endpoint, model_id, orig, preset_id, k_shot=0, img_uri=None):
    async with sem:
        try:
            # word cap -> token cap, so the limit holds even when the instruction is ignored
            _, cap = length_budget(PRESETS[preset_id], len(orig.split()))
            return await _call(session, endpoint, model_id,
                               build_messages(orig, preset_id, k_shot, img_uri),
                               TEMPERATURES[preset_id], max_tokens=int(cap * 2.2) + 32)
        except Exception as e:
            return f"<<ERROR: {type(e).__name__}: {e}>>"


async def process_chunk(sem, session, endpoint, model_id, chunk, presets, k_shot=0, tag="",
                        img_max_side=0, naming=None):
    # image encoding blocks, so it runs once per chunk in a thread; all presets share the URI
    if img_max_side:
        uris = await asyncio.gather(*[
            asyncio.to_thread(_safe_uri, rec.get("image", ""), img_max_side) for rec in chunk])
    else:
        uris = [None] * len(chunk)
    tasks = [gen_one(sem, session, endpoint, model_id, rec["caption"], p, k_shot, uris[i])
             for i, rec in enumerate(chunk) for p in presets]
    res = await asyncio.gather(*tasks)
    ns, out = len(presets), []
    for i, rec in enumerate(chunk):
        out.append([rec["image_id"], rec.get("image", ""), ORIGINAL_STYLE, rec.get("caption", ""),
                    rec.get("scene", ""), rec.get("normal", ""), rec.get("anomaly", "")])
        for j, p in enumerate(presets):
            out.append([rec["image_id"], rec.get("image", ""),
                        (naming(p) if naming else p) + tag, res[i * ns + j],
                        rec.get("scene", ""), rec.get("normal", ""), rec.get("anomaly", "")])
    return out


async def main_async(args):
    global OUT_DIR
    if getattr(args, "out_dir", ""):
        OUT_DIR = args.out_dir
    # a downloaded CSV needs neither the endpoint nor aiohttp, so return before touching them
    _csv = f"{OUT_DIR}/{OUT_NAME}"
    if os.path.exists(_csv) and not os.path.exists(f"{OUT_DIR}/done_image_ids.txt") and not args.force:
        print(f"  [recap] already present: {_csv}  ({os.path.getsize(_csv) / 2 ** 30:.1f} GB)\n        Treated as a downloaded CSV, so generation is skipped (no done_image_ids.txt means this was not an interrupted run).\n        Pass --force to rebuild, or --out-dir to write elsewhere", flush=True)
        return
    presets = args.presets.split(",") if args.presets else PRESET_IDS      # checked before connecting to the server
    if unknown := [p for p in presets if p not in PRESETS]:
        raise SystemExit(f"[preset] the {PRESET_SET} table has no preset {unknown}\n"
                         f"  available: {PRESET_IDS}\n"
                         f"  for the other table pass --preset-set {'axes' if PRESET_SET == 'msr' else 'msr'}")
    import aiohttp
    endpoint = args.endpoint.rstrip("/")
    async with aiohttp.ClientSession() as s:
        async with s.get(f"{endpoint}/v1/models",
                         timeout=aiohttp.ClientTimeout(total=10)) as r:
            model_id = (await r.json())["data"][0]["id"]
    print(f"endpoint={endpoint}  model={model_id}  concurrency={args.concurrency}  "
          f"chunk={args.chunk}  presets={PRESET_SET}:{len(presets)}(+{ORIGINAL_STYLE})  "
          f"image={'ON @'+str(args.img_max_side)+'px' if args.img_max_side else 'OFF (text-only)'}",
          flush=True)

    if args.exemplars:
        load_exemplars(args.exemplars)
    tag = f"@{args.rewriter}" if args.rewriter else ""
    naming = (lambda pid: style_slug(PRESETS[pid])) if args.style_naming == "axes" else (lambda pid: pid)
    records = load_all_captions(ANNOT_DIR)
    print(f"  loaded {len(records):,} captions  k_shot={args.k_shot}  tag={tag or '(none)'}",
          flush=True)
    if args.limit > 0:
        records = records[:args.limit]

    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump({"preset_set": PRESET_SET, "original_style": ORIGINAL_STYLE,
               "axes": AXES, "presets": PRESETS},
              open(f"{OUT_DIR}/control_vectors.json", "w"), indent=1, ensure_ascii=False)
    csv_path, done_path = f"{OUT_DIR}/{OUT_NAME}", f"{OUT_DIR}/done_image_ids.txt"
    done = ({l.strip() for l in open(done_path)}
            if os.path.exists(done_path) and not args.force else set())
    if done:
        print(f"  resume: {len(done):,} done", flush=True)
    pending = [r for r in records if r["image_id"] not in done]
    if not pending and not args.force:
        print(f"  [recap] already present: {csv_path}\n        {len(done):,} / {len(records):,} images done — nothing to generate.\n        Pass --force to rebuild", flush=True)
        return
    print(f"Pending: {len(pending):,} × {len(presets)} = {len(pending)*len(presets):,} generations",
          flush=True)

    new = not os.path.exists(csv_path)
    fo = open(csv_path, "a", newline="", encoding="utf-8"); w = csv.writer(fo)
    if new:
        w.writerow(["image_id", "image_path", "style", "caption", "scene", "normal", "anomaly"])
    fd = open(done_path, "a", encoding="utf-8")
    sem = asyncio.Semaphore(args.concurrency); t0 = time.time()
    from tqdm import tqdm
    async with aiohttp.ClientSession() as session:
        with tqdm(total=len(pending), desc="captions") as pbar:
            for i in range(0, len(pending), args.chunk):
                ch = pending[i:i + args.chunk]
                w.writerows(await process_chunk(sem, session, endpoint, model_id, ch, presets,
                                                args.k_shot, tag, args.img_max_side, naming))
                fo.flush()
                for rec in ch:
                    fd.write(rec["image_id"] + "\n")
                fd.flush(); pbar.update(len(ch))
    fo.close(); fd.close()
    print(f"done in {time.time()-t0:.0f}s → {csv_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MSR-axes recaptioning (8-axis control vector)")
    ap.add_argument("--endpoint", help="OpenAI-compatible /v1 endpoint")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preset-set", default=PRESET_SET, choices=sorted(TABLES),
                    help="preset table. msr=the 11 styles of the shipped CSVs (default) · axes=11 redesigned over the 8 axes")
    ap.add_argument("--style-naming", default="name", choices=["name", "axes"],
                    help="name=p01_lexical (default) · axes=8-axis slug, which identifies and restores the coordinates on its own")
    ap.add_argument("--presets", default="", help="comma separated (default = all 11 of the active table)")
    ap.add_argument("--exemplars", default="", help="train-seeded in-context exemplar JSON")
    ap.add_argument("--k-shot", type=int, default=0, help="number of exemplars (default 0)")
    ap.add_argument("--rewriter", default="", help="rewriter-source tag, recorded in the style column as '@tag'")
    ap.add_argument("--out-dir", default="", help="output directory (default %s)" % OUT_DIR)
    ap.add_argument("--img-max-side", type=int, default=0,
                    help="image conditioning: long-side pixels (0 = text only; 448 is the setting used here)")
    ap.add_argument("--force", action="store_true",
                    help="regenerate from the start, ignoring already-completed images")
    ap.add_argument("--dry-run", action="store_true", help="print the prompts and exit")
    a = ap.parse_args()
    if a.preset_set != PRESET_SET:                    # switching tables swaps the globals together
        PRESET_SET = a.preset_set
        PRESETS, TEMPERATURES, ORIGINAL_STYLE = TABLES[PRESET_SET]
        PRESET_IDS = list(PRESETS)
    if a.dry_run:
        print(f"[{PRESET_SET}] {len(PRESETS)} presets + {ORIGINAL_STYLE} = {len(PRESETS)+1} rows per image\n")
        for pid, cv in PRESETS.items():
            print(f"── {pid}  {style_slug(cv)}  {cv}")
            print(build_instruction(cv)); print()
        sys.exit(0)
    if not a.endpoint:
        ap.error("--endpoint is required (or --dry-run)")
    asyncio.run(main_async(a))
