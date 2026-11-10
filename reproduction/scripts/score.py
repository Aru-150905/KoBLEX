import json, ast, sys, glob, os, re
sys.path.insert(0, sys.argv[1] + "/experiments/lf-eval")
from eval_pipeline import compute_token_f1, evaluate_retrieval
root = sys.argv[1]
gold = [json.loads(l) for l in open(f"{root}/experiments/source/koblex.jsonl")]
def norm(s): return re.sub(r'\s+', ' ', s).strip()
rows = []
for f in sorted(glob.glob(f"{root}/results/*.jsonl")):
    preds = [json.loads(l) for l in open(f)]
    if len(preds) != len(gold): rows.append((os.path.basename(f), len(preds), None, None, None)); continue
    f1s, pc, gc = [], [], []
    for p, g in zip(preds, gold):
        a = p.get('answers')
        try: a = ast.literal_eval(a) if isinstance(a, str) else a
        except Exception: a = [{'answer': a}]
        if isinstance(a, str): a = [{'answer': a}]
        top = a[0] if a else {}
        ans = str(top.get('answer', '')).split("</think>")[-1][:800]
        f1s.append(compute_token_f1(ans, g['answer'])['f1'])
        pc.append([norm(x) for x in top.get('comb', []) or []])
        gctx = ast.literal_eval(g['contexts']) if isinstance(g['contexts'], str) else g['contexts']
        gc.append([norm(c['hierarchy'] + c['content']) for c in gctx])
    r = evaluate_retrieval(pc, gc)
    rows.append((os.path.basename(f), len(preds), sum(f1s)/len(f1s)*100, r['recall']*100, r['f1']*100))
for n, c, f1, rr, rf in sorted(rows, key=lambda x: -(x[2] or -1)):
    print(f"{n:32s} n={c:4d}  " + ("misaligned" if f1 is None else f"ansF1={f1:5.1f}  ret_recall={rr:5.1f}  ret_F1={rf:5.1f}"))
