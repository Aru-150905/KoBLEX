# Reproducing KoBLEX / ParSeR (EMNLP 2025)

A reproduction of [KoBLEX: Open Legal Question Answering with Multi-hop Reasoning](https://aclanthology.org/2025.emnlp-main.200/) from the authors' repository [daehuikim/KoBLEX](https://github.com/daehuikim/KoBLEX) (commit `1a2d225`), by Arihant Rakhecha (BITS Pilani, Hyderabad Campus).

## Headline result

The paper reports that ParSeR beats standard retrieval by **+37.91 F1** with GPT-4o. Scoring the authors' saved outputs gives:

| GPT-4o, 226 questions | Answer token F1 | Retrieval F1 |
| --- | --- | --- |
| ParSeR | 46.1 | **59.41** |
| No retrieval | 36.2 | – |
| Standard retrieval (BM25) | 26.8 | **21.50** |

59.41 − 21.50 = **37.90**, which matches the paper's figure. That is the *retrieval* F1 gap; the answer F1 gap is smaller (+19.4).

## What was reproduced

| Part | Result | How |
| --- | --- | --- |
| Score all 46 saved runs | Table in `scripts/scores.txt` | `scripts/score.py` (reuses the repo's metric functions) |
| Re-run the BM25 baseline from scratch | F1 21.4 vs 21.5; identical retrieval for 218/226 questions | `scripts/bm25_baseline.py` (CPU) |
| By-hop breakdown and failure cases | ParSeR's typical error is a near miss: right law, wrong neighbouring paragraph | `scripts/analyse.py` |
| Own experiment: Korean character-bigram BM25 | Baseline F1 21.4 → 37.4, recall@100 51.8 → 75.6 | `scripts/bm25_variants.py` |
| Full ParSeR run (GPT-4o + GPU reranker) | Notebook ready, not yet run | `notebook/KoBLEX_ParSeR_reproduction.ipynb` (Colab, T4) |

The raw terminal output of every run is saved in [`outputs/`](outputs/), so the numbers can be checked without running anything.

The full write-up, with the pipeline diagram, results, failure analysis and implications for Indian law, is in [`report/KoBLEX_ParSeR_Reproduction_Report.pdf`](report/KoBLEX_ParSeR_Reproduction_Report.pdf).

## Fixes to the original code

Each fix is a separate commit on top of the authors' code, so `git log` shows exactly what changed.

| Commit | Problem in the original | Effect |
| --- | --- | --- |
| Fix ParSeR selection | `parser/openai` and `parser/vllm` `selection_retrieval.py` used `contexts_list[0][choice]` for every sub-query; the original script uses `re[i]` | Multi-hop questions got provisions from the wrong candidate list |
| Eval: read result format | `lf-eval/eval_pipeline.py` read `answer`, but `results/*.jsonl` store `answers`; `"<\think>"` was a tab escape | `KeyError` on every saved file; Qwen reasoning never stripped |
| run_pipeline paths | `source/...` instead of `../../source/...` | Pipeline failed at step 1 |
| OpenAI client | 10 s total timeout; failed calls return `None` | Timeouts, then crashes |
| Compatibility | `default_activation_function=` rejected by sentence-transformers ≥ 4; backslash in an f-string expression | `TypeError` on current Colab; `SyntaxError` on Python < 3.12 |

With these fixes, the official evaluation script now runs on the saved outputs and gives the same numbers as `scripts/score.py` (ParSeR GPT-4o: token F1 0.4614, retrieval F1 0.5941).

Not changed, but flagged: the standard-retrieval baseline retrieves `k = n_hops` taken from the gold labels, and BM25 tokenizes Korean with English rules.

## Run it yourself

CPU scripts (any laptop with about 4 GB of free RAM), from the repository root:

```bash
python3 -m venv venv && source venv/bin/activate
pip install bm25s PyStemmer deepeval numpy

python3 reproduction/scripts/score.py .          # score all saved runs
python3 reproduction/scripts/bm25_baseline.py .  # BM25 baseline re-run
python3 reproduction/scripts/bm25_variants.py .  # bigram experiment
python3 reproduction/scripts/analyse.py .        # by-hop table and failure cases

# the official script, now fixed
cd experiments && python3 lf-eval/eval_pipeline.py --input_file ../results/parser_gpt4o.jsonl \
  --gold_file source/koblex.jsonl --collections_file source/statute.jsonl --eval_type retrieval --no_log
```

Full ParSeR run: open `notebook/KoBLEX_ParSeR_reproduction.ipynb` in Google Colab, choose a T4 GPU runtime, and run all cells. It asks for an OpenAI API key and starts with 20 questions so you can see the cost first.

## Limitations

- The full GPT-4o ParSeR run has not been completed yet; ParSeR numbers above come from the authors' saved outputs.
- LF-Eval (the paper's LLM-judge metric) was not re-run.
- All numbers are single runs.

## Licence and credit

The original code and data are by the KoBLEX authors and are licensed under [CC BY-NC 4.0](../LICENSE); this reproduction is shared under the same licence, for non-commercial use. Please cite the original paper (see the [main README](../README.md)).
