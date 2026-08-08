# tools — weights and promotion

```
ensemble/adopted.py       single source of the shipped weights (reads weights/final.json)
ensemble/weights/         final.json = the adopted set · search/ = search outputs (opt-in via WEIGHTS)
ensemble/base.py          base-member weight search (held-out greedy)
ensemble/comb_search.py   reranker z-fusion weight search (per-axis grid)
ensemble/ensemble.py      combination core (normalize/mmnorm) + external-bench harness
ensemble/mass_pab.py      PAB-side weight-mass sweep
ensemble/dumps/           per-member score dumps for the external benches (uca · rstp)
promote.py                copy a chosen checkpoint into assets/model_rep/ (manual deploy)
```

Every pipeline stage imports `adopted.py` rather than reading JSON directly, so
`weights/final.json` is the only place the shipped numbers live. Point `WEIGHTS`
at another file of the same shape to experiment; the shipped file never changes.

```bash
python tools/ensemble/adopted.py --show          # print the adopted set
python tools/ensemble/adopted.py --get comb      # one block as JSON
```

Search scripts read the bench dumps under `assets/data/benches/{ucc,uca,rstp,ruleclean}`
(`ensemble/dumps/` regenerates the encoder ones; the reranker score dumps need the
VLMs and are shipped prebuilt).

## Note — qwen3vl_2b union coverage

The shipped `qwen3vl_2b_union_cache.pt` scores only each query's top-5 candidates
(23.8% of the union): the 2B reranker was run with K=5 (`recs_2b_dora_k5_p3.pt`),
so deeper candidates were never scored and impute as zero in S3. This is by
design, not missing data — the adopted submission was produced with exactly this
coverage, and `build_union.py` merges the cache as-is to reproduce it.
