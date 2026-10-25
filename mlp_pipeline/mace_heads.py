"""Print the heads of the multihead MACE foundation models (mh-0, mh-1) so they can be selected as  mace:mh-1@<head>.
usage: ~/work/envs/mace-ng/bin/python mlp_pipeline/mace_heads.py [mh-1 mh-0]"""
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
        print(nm, "failed:", str(e)[:200])
