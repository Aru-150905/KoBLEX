"""By-hop breakdown of saved results and failure-case mining (ParSeR GPT-4o vs standard retrieval GPT-4o)."""
import json, ast, sys, re
from collections import defaultdict
sys.path.insert(0, sys.argv[1] + "/experiments/lf-eval")
from eval_pipeline import compute_token_f1, evaluate_retrieval
root = sys.argv[1]
gold = [json.loads(l) for l in open(f"{root}/experiments/source/koblex.jsonl", encoding="utf-8")]
norm = lambda s: re.sub(r'\s+', ' ', s).strip()
def load(name):
    out = []
    for l in open(f"{root}/results/{name}.jsonl"):
        a = json.loads(l)['answers']
        a = a[0] if isinstance(a, list) else {'answer': a, 'comb': []}
        out.append((str(a['answer']).split("</think>")[-1][:800], [norm(x) for x in a.get('comb') or []]))
    return out
G = []
for g in gold:
    gc = ast.literal_eval(g['contexts']) if isinstance(g['contexts'], str) else g['contexts']
    G.append([norm(c['hierarchy'] + c['content']) for c in gc])
runs = {k: load(k) for k in ["parser_gpt4o", "sp_or_gpt4o", "sp_gpt4o", "flare_gpt4o", "ircot_gpt4o"]}
print("hop  n   | " + " | ".join(f"{k:>12s} ansF1/retF1" for k in runs))
for h in ["1", "2", "3"]:
    idx = [i for i, g in enumerate(gold) if str(g['n_hops']) == h]
    cells = []
    for k, r in runs.items():
        af = sum(compute_token_f1(r[i][0], gold[i]['answer'])['f1'] for i in idx) / len(idx) * 100
        rf = evaluate_retrieval([r[i][1] for i in idx], [G[i] for i in idx])['f1'] * 100
        cells.append(f"{af:12.1f}/{rf:5.1f}")
    print(f"{h:3s} {len(idx):3d} | " + " | ".join(cells))
# refusals in standard retrieval
ref = sum(1 for a, _ in runs["sp_or_gpt4o"] if re.search(r"does not contain|cannot be (generated|determined|answered)|not (provided|included)", a, re.I))
print(f"\nsp_or_gpt4o answers that refuse / say context is irrelevant: {ref}/226")
ref2 = sum(1 for a, _ in runs["parser_gpt4o"] if re.search(r"does not contain|cannot be (generated|determined|answered)|not (provided|included)", a, re.I))
print(f"parser_gpt4o refusals: {ref2}/226")
# failure cases
print("\n=== cases: ParSeR right provision, standard retrieval wrong")
p, s = runs["parser_gpt4o"], runs["sp_or_gpt4o"]
cnt = 0
for i, g in enumerate(gold):
    if set(p[i][1]) == set(G[i]) and not set(s[i][1]) & set(G[i]) and str(g['n_hops']) in "12" and cnt < 3:
        cnt += 1
        print(f"\n[{g['id']}] Q: {g['question_eng']}\n  gold law: {[x[:40] for x in G[i]]}\n  std-retrieval got: {[x[:40] for x in s[i][1]]}\n  std answer: {s[i][0][:160]}\n  ParSeR F1={compute_token_f1(p[i][0], g['answer'])['f1']}")
print("\n=== cases: ParSeR retrieved wrong provision")
cnt = 0
for i, g in enumerate(gold):
    if not set(p[i][1]) & set(G[i]) and cnt < 3:
        cnt += 1
        print(f"\n[{g['id']}] Q: {g['question_eng']}\n  gold answer: {g['answer_eng'][:200]}\n  gold law: {[x[:45] for x in G[i]]}\n  ParSeR got: {[x[:45] for x in p[i][1]]}\n  ParSeR answer(ko): {p[i][0][:150]}")
none = sum(1 for i in range(226) if not set(p[i][1]) & set(G[i]))
print(f"\nParSeR retrieved none of the gold provisions for {none}/226 questions")
