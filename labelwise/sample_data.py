"""Deterministic synthetic support-request fixture."""
import csv
import io
from pathlib import Path
import numpy as np
from itertools import permutations

CLASSES = {
    "access": ["login", "password", "account", "sign in", "verification", "locked"],
    "billing": ["invoice", "charged", "payment", "refund", "card", "subscription"],
    "technical": ["crash", "error", "slow", "screen", "sync", "upload"],
}
TEMPLATES = ["{a} problem with {b}", "help: {b} and {a}", "customer reports {a} {b}", "unable to use {a}; {b} issue"]

def generate(seed=20261012):
    rng=np.random.default_rng(seed); rows=[]
    for label,words in CLASSES.items():
        candidates=[template.format(a=a,b=b) for template in TEMPLATES for a,b in permutations(words,2)]
        order=rng.permutation(len(candidates))[:80]
        for i,index in enumerate(order):
            text=candidates[index]
            if i%7==0:text += " urgent customer request"
            rows.append({"id":f"{label}-{i:03}","text":text,"label":label})
    # Fixed, stratified splits: 12 seed (4/class), 168 pool, 60 held-out test.
    seed_rows=[];pool=[];test=[]
    for label in CLASSES:
        group=[r for r in rows if r["label"]==label]
        seed_rows.extend(group[:4]);pool.extend(group[4:60]);test.extend(group[60:])
    return seed_rows,pool,test

def encode(rows, labelled=True):
    out=io.StringIO(newline="");fields=["id","text"] + (["label"] if labelled else [])
    writer=csv.DictWriter(out,fieldnames=fields,lineterminator="\n");writer.writeheader()
    for row in rows:writer.writerow({key:row[key] for key in fields})
    return out.getvalue()

def write(root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);seed,pool,test=generate()
    (root/"seed.csv").write_text(encode(seed));(root/"pool-labelled.csv").write_text(encode(pool))
    (root/"pool-unlabelled.csv").write_text(encode(pool,False));(root/"test.csv").write_text(encode(test))

if __name__=="__main__":write(Path(__file__).parent/"sample")
