#!/usr/bin/env python3
"""Write multi-cell variants of the benchmark instances to data/multicell/.
Usage: python scripts/prepare_benchmarks.py --cells 2 4 8 --sets brandimarte hurink/edata dauzere
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from mcfjsp.instances import read_fjsp_text
from mcfjsp.partition import multicell_variant

ap = argparse.ArgumentParser()
ap.add_argument("--cells", type=int, nargs="+", default=[2, 4, 8])
ap.add_argument("--sets", nargs="+", default=["brandimarte"])
ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
args = ap.parse_args()
out_dir = os.path.join(args.data, "multicell"); os.makedirs(out_dir, exist_ok=True)
meta = {m["name"]: m for m in json.load(open(os.path.join(args.data, "benchmarks", "instances.json")))}
index = []
for s in args.sets:
    for path in sorted(glob.glob(os.path.join(args.data, "benchmarks", s, "*.txt"))):
        name = os.path.splitext(os.path.basename(path))[0]
        base = read_fjsp_text(path, name)
        for C in args.cells:
            if C > base.n_machines:
                continue
            inst = multicell_variant(base, C)
            lit = meta.get(name, {})
            inst.meta["literature_lower_bound"] = lit.get("optimum") or (lit.get("bounds") or {}).get("lower")
            fn = os.path.join(out_dir, f"{name}_c{C}.json"); inst.save(fn)
            index.append(dict(name=inst.name, base=name, cells=C, ops=inst.n_ops, machines=inst.n_machines,
                              transitions=len(inst.transitions), file=os.path.relpath(fn, args.data)))
json.dump(index, open(os.path.join(out_dir, "index.json"), "w"), indent=1)
print(f"wrote {len(index)} instances to {out_dir}")
