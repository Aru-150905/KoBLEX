# Saved outputs

Every file here is the verbatim terminal output of one run, made from a fresh clone of this repository (commit `9192c46`) on 10 Nov 2026, on CPU (Python 3.13, bm25s 0.3.x). Each file starts with the exact command used.

| File | What it shows | Key numbers |
| --- | --- | --- |
| `01_score_all_saved_runs.txt` | All 46 saved runs scored with the repo's own metric functions | ParSeR GPT-4o: answer F1 46.1, retrieval F1 59.4 |
| `02_bm25_baseline_rerun.txt` | Standard-retrieval BM25 baseline re-run from scratch | F1 21.4 vs 21.5 saved; 218/226 identical |
| `03_bigram_bm25_experiment.txt` | Own experiment: Korean character bigrams vs the repo's tokenizer | F1 21.4 → 37.4; recall@100 51.8 → 75.6 |
| `04_by_hop_and_failure_cases.txt` | By-hop breakdown and example failures | ParSeR found no gold provision for 36/226 |
| `05_official_eval_script_gpt4o.txt` | The authors' `lf-eval/eval_pipeline.py`, now fixed, on all GPT-4o runs | ParSeR retrieval F1 0.5941, token F1 0.4614 |

Not here yet: the full ParSeR run with GPT-4o (needs a GPU and an OpenAI API key; see `../notebook/`) and LF-Eval (needs an OpenAI API key).
