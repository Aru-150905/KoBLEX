"""Own experiment: does Korean-aware tokenization fix BM25? Compare whitespace/English tokenizer vs char bigrams; also k=gold hops vs fixed k."""
import json, ast, sys, re, time
import bm25s, Stemmer
sys.path.insert(0, sys.argv[1] + "/experiments/lf-eval")
from eval_pipeline import evaluate_retrieval
root = sys.argv[1]; src = f"{root}/experiments/source"
ctx = [s['hierarchy'] + s['content'] for s in map(json.loads, open(f"{src}/statute.jsonl", encoding="utf-8"))]
gold = [json.loads(l) for l in open(f"{src}/koblex.jsonl", encoding="utf-8")]
norm = lambda s: re.sub(r'\s+', ' ', s).strip()
golds = []
for g in gold:
    gc = ast.literal_eval(g['contexts']) if isinstance(g['contexts'], str) else g['contexts']
    golds.append({norm(c['hierarchy'] + c['content']) for c in gc})
def bigrams(t):
    out = []
    for w in re.findall(r'[가-힣]+|[a-zA-Z0-9]+', t):
        out += [w] if len(w) < 2 else [w[i:i+2] for i in range(len(w)-1)]
    return out
stem = Stemmer.Stemmer("english")
def recall_at(ids_list, k):
    hits = [len({norm(ctx[i]) for i in ids[:k]} & G) / len(G) for ids, G in zip(ids_list, golds)]
    return sum(hits) / len(hits) * 100
for name in ["repo (English tokenizer)", "Korean char bigrams"]:
    t = time.time()
    r = bm25s.BM25()
    if name.startswith("repo"):
        r.index(bm25s.tokenize(ctx, stopwords="en", stemmer=stem, show_progress=False), show_progress=False)
        q = lambda s: bm25s.tokenize([s], stemmer=stem, show_progress=False)
    else:
        r.index([bigrams(c) for c in ctx], show_progress=False)
        q = lambda s: [bigrams(s)]
    ids_all, preds = [], []
    for g in gold:
        res, _ = r.retrieve(q(g['background'] + g['question']), k=100, show_progress=False)
        ids = list(res[0]); ids_all.append(ids)
        preds.append([norm(ctx[i]) for i in ids[:int(g['n_hops'])]])
    f1 = evaluate_retrieval(preds, [list(G) for G in golds])['f1'] * 100
    print(f"{name:28s} F1@gold-k={f1:5.1f}  R@10={recall_at(ids_all,10):5.1f}  R@30={recall_at(ids_all,30):5.1f}  R@100={recall_at(ids_all,100):5.1f}  ({time.time()-t:.0f}s)")
