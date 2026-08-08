#!/usr/bin/env python3
"""env — interpreter selection helper.

Model families need different environments (see `requirements/README.md`). In particular the
MoE reranker `internvl_r32` (InternVL3.5-30B-A3B) requires `torch._grouped_mm` to work on the
current GPU:

  · torch 2.8 defines `_grouped_mm` but only supports it on sm_90 (Hopper). On Blackwell
    (sm_100+) every forward raises RuntimeError.
  · A `hasattr(torch, "_grouped_mm")` check therefore does not detect this; the op has to be
    actually called.

This check runs before the scorer is started, so an unusable interpreter fails immediately
rather than after a long run.
"""
import os
import subprocess
import sys

# Models that are MoE and therefore need grouped_mm (matched against the path or name)
MOE_HINTS = ("InternVL3_5-30B-A3B", "A3B")

_PROBE = (
    "import torch,sys\n"
    "try:\n"
    "    if not torch.cuda.is_available(): sys.exit(3)\n"
    "    a=torch.randn(4,8,device='cuda',dtype=torch.bfloat16)\n"
    "    w=torch.randn(2,8,8,device='cuda',dtype=torch.bfloat16)\n"
    "    o=torch.tensor([2,4],device='cuda',dtype=torch.int32)\n"
    "    torch._grouped_mm(a,w,offs=o)\n"
    "    sys.exit(0)\n"
    "except Exception:\n"
    "    sys.exit(1)\n"
)


def needs_moe(model_or_name: str) -> bool:
    """Whether this model requires grouped_mm (MoE)."""
    return any(h in str(model_or_name) for h in MOE_HINTS)


def grouped_mm_ok(python: str = None, timeout: int = 180) -> bool:
    """Whether the given interpreter can actually run `_grouped_mm` on this GPU."""
    py = python or sys.executable
    try:
        return subprocess.run([py, "-c", _PROBE], capture_output=True, timeout=timeout).returncode == 0
    except Exception:
        return False


def resolve_python(model_or_name: str, env_var: str = "PY_VLLM", candidates=None) -> str:
    """Pick the interpreter that fits the model.

    Non-MoE models keep the current interpreter. For MoE models each candidate is verified by
    actually calling the op, and the first one that passes is returned. If none pass, the
    function aborts with instructions on how to fix it.
    """
    explicit = os.environ.get(env_var)
    if not needs_moe(model_or_name):
        return explicit or sys.executable

    # No absolute paths outside the repo are hard-coded; candidates come from PY_VLLM or from
    # the conda env created by setup_conda_envs.sh.
    cands = [c for c in ([explicit] + list(candidates or []) + [
        os.path.join(os.environ.get("CONDA_PREFIX_1", "/opt/conda"), "envs", "track4_vllm", "bin", "python"),
        "/opt/conda/envs/track4_vllm/bin/python",
        sys.executable,
    ]) if c and os.path.exists(c)]

    seen, tried = set(), []
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        if grouped_mm_ok(c):
            if explicit and c != explicit:
                print(f"  [env] {env_var}={explicit} fails grouped_mm on this GPU -> using {c} instead", flush=True)
            return c
        tried.append(c)

    raise SystemExit(
        f"[env] '{model_or_name}' is MoE and needs torch._grouped_mm, but every interpreter "
        f"tried failed on this GPU (sm_100+):\n    " + "\n    ".join(tried) +
        f"\n  _grouped_mm in torch 2.8 is sm_90 only; a torch 2.11+cu130 environment is required.\n"
        f"  create:  bash requirements/setup_conda_envs.sh --only vllm\n"
        f"  select:  {env_var}=<that env>/bin/python  (see requirements/README.md)")
