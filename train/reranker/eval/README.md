# `eval`

Two evaluators shared by the three fine-tuned rerankers. Neither opens the Track 4 test set, and
both read only precomputed caches, so neither needs a GPU for the measurement itself.

| | answers |
|---|---|
| `eval_step.py` | which **step** of one member to deploy |
| `eval_heldout_rerank.py` | how much each member **contributes** to the fused ranking |

## `eval_step.py` — step selection

```
t = argmax  ½·( z(acc_ucc) + z(acc_clean) )
```

Pair accuracy on Path B pairs, standardised along the step axis, averaged over two splits:

| split | negatives | measures |
|---|---|---|
| `ucc` | base top-20 candidates — the distribution deployment actually sees | generalisation |
| `clean_<member>` | the untrained tail of that member's own negcache (index ≥ SKIP) | fit to the training objective |

Neither split is used alone: `clean` alone rewards memorising the training distribution, and the
script prints its argmax separately as a warning.

Stages, each cached on disk:

```bash
# 1+2. build the pair bench and the candidate pool (once, shared by all members)
python eval_step.py --build-only

# 3+4. score a member's steps, then select
python eval_step.py --member dora --run <run> --steps ex006000,ex007000,ex008000 --gpu 7
```

| artifact | |
|---|---|
| `assets/data/benches/rerankstep/rerank_step_pairs.json` | pair bench, rebuilt with `--rebuild` |
| `<work>/gallery` · `query_text.json` · `union_pool.pt` | the scorer input contract |
| `<work>/<member>_<step>_n<N>_pairs.pt` | per-step scores |
| `<work>/steps_<member>.json` | the selection and every row behind it |

`--member` maps to the pipeline scorer, so scoring uses exactly the deployment code path:

| member | scorer | ckpt layout |
|---|---|---|
| `r32` | `score_union_hf_4b.py --adapter` | `<run>/step{N}` |
| `dora` | `score_union_qwen_4b.py --adapter --qwen 2b` | `<run>/checkpoints/ex{NNNNNN}` |
| `jina` | `score_union_jina.py --ckpt` | `<run>/checkpoints/ex{NNNNNN}` |

Useful flags: `--n` (queries per split, default 400), `--score-only` / `--measure-only` to run one
half, `--deploy-rep <NAME>` to promote the selection immediately, and `--adopted` to compare against
a specific step. Gallery filenames are zero-padded to 8 digits so `sorted(listdir)` equals the
candidate index; the script asserts this before scoring.

## `eval_heldout_rerank.py` — contribution

```
comb = SIM_W·z(sim) + Σ w_k·z(rerank_k) → injective assignment → mAP@10 / R@1
```

Assignment and softmax come from `pipeline/S3_assign/assign.py` unchanged, so the numbers reflect the
deployed ranking rather than a reimplementation. Sections printed:

| | |
|---|---|
| 1 solo | sim plus one member at `--solo-w` — the ckpt-selection axis |
| 2 comb | all members at `--uniform-w`, or explicit weights via `--comb-w` |
| 3 LOO | the comb with one member removed |
| 4 sweep | one member's weight across 0.0–0.7 (`--sweep`) |
| 5 noise floor | query bootstrap → SE, 95% CI, and the smallest resolvable difference (`--bootstrap N`) |

```bash
python eval_heldout_rerank.py --members internvl_r32,qwen3vl_2b,jina_m0 --bootstrap 1000
```

The bench is PAB rule-clean under `assets/data/benches/ruleclean/` (override with `RC_DIR`). Members
are discovered from `redump_<name>_ruleclean.pt`, so a newly dumped step appears without a code
change; `--members all` takes everything present.

Two caveats the output states as well: passing the deployed weights to `--comb-w` makes the result
test-derived and therefore not valid selection evidence, and the bootstrap resamples *after*
assignment, so the reported SE understates the true one.

## Files

| | |
|---|---|
| `eval_step.py` | pair-accuracy step selection |
| `eval_heldout_rerank.py` | per-member contribution on the rule-clean bench |
