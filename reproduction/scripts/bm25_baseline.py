"""Re-run the KoBLEX standard-retrieval (sp_or / cot_or) BM25 step on CPU and compare with saved outputs."""
import json, ast, sys, re, time
import bm25s, Stemmer
sys.path.insert(0, sys.argv[1] + "/experiments/lf-eval")
from eval_pipeline import evaluate_retrieval
root = sys.argv[1]
src = f"{root}/experiments/source"
statutes = [json.loads(l) for l in open(f"{src}/statute.jsonl", encoding="utf-8")]
ctx = [s['hierarchy'] + s['content'] for s in statutes]
gold = [json.loads(l) for l in open(f"{src}/koblex.jsonl", encoding="utf-8")]
t = time.time()
stem = Stemmer.Stemmer("english")
ret = bm25s.BM25(); ret.index(bm25s.tokenize(ctx, stopwords="en", stemmer=stem, show_progress=False), show_progress=False)
print(f"indexed {len(ctx)} provisions in {time.time()-t:.0f}s")
norm = lambda s: re.sub(r'\s+', ' ', s).strip()
saved = [json.loads(l)['answers'][0]['comb']
         for l in open(f"{root}/results/sp_or_gpt4o.jsonl")]
mine, golds, same = [], [], 0
for g, sv in zip(gold, saved):
    k = int(g['n_hops'])
    res, _ = ret.retrieve(bm25s.tokenize([g['background'] + g['question']], stemmer=stem, show_progress=False), k=k, show_progress=False)
    m = [ctx[i] for i in res[0]]
    mine.append([norm(x) for x in m])
    gc = ast.literal_eval(g['contexts']) if isinstance(g['contexts'], str) else g['contexts']
    golds.append([norm(c['hierarchy'] + c['content']) for c in gc])
    same += set(map(norm, m)) == set(map(norm, sv))
r = evaluate_retrieval(mine, golds)
print(f"my BM25 run: P={r['precision']*100:.1f} R={r['recall']*100:.1f} F1={r['f1']*100:.1f} EM={r['em']*100:.1f}")
print(f"identical to saved retrieval for {same}/{len(gold)} questions")
# by hop count
from collections import defaultdict
by = defaultdict(lambda: ([], []))
for g, m, gg in zip(gold, mine, golds): by[g['n_hops']][0].append(m); by[g['n_hops']][1].append(gg)
for h in sorted(by): rr = evaluate_retrieval(*by[h]); print(f"  {h}-hop (n={len(by[h][0])}): F1={rr['f1']*100:.1f}")
