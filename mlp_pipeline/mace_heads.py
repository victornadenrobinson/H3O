"""Print the heads of the multihead MACE foundation models (mh-0, mh-1) so they can be selected as  mace:mh-1@<head>.
usage: ~/work/envs/mace-ng/bin/python mlp_pipeline/mace_heads.py [mh-1 mh-0] [--names] [--pbe]   (--pbe: only PBE heads)"""
import sys
from mace.calculators import mace_mp
names_only = "--names" in sys.argv
for nm in ([a for a in sys.argv[1:] if not a.startswith("--")] or ["mh-1", "mh-0"]):
    try:
        c = mace_mp(model=nm, default_dtype="float32", device="cpu")
        heads = getattr(c.models[0], "heads", None)
        if names_only:
            print("\n".join(f"{nm}@{h}" for h in (heads or [])))
        else:
            print(nm, "heads:", heads, "| available:", getattr(c, "available_heads", "?"))
    except Exception as e:
        import re
        m = re.search(r"Available heads are: \[(.*?)\]", str(e))     # the model refuses to load without a head and lists them
        if m:
            heads = [h.strip().strip("'\"") for h in m.group(1).split(",")]
            heads = [h for h in heads if "--pbe" not in sys.argv or "pbe" in h.lower()]
            print("\n".join(f"{nm}@{h}" for h in heads) if names_only else f"{nm} heads: {heads}")
            continue
        import traceback; traceback.print_exc()
        print(nm, "failed:", type(e).__name__, str(e)[:300], file=sys.stderr if names_only else sys.stdout)
