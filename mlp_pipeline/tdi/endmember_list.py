"""List the TDI runs that are end members of the mixing / hull problem and can be extended in time: fcc and bcc H2O crystals, liquid H2O and pure H, all at dt 0.5 fs (dt 1.0 runs are left alone), as '<dir>/<tag>' base paths, one per line, sorted (a stable list: array tasks index into it).
usage (repo root):  python3 mlp_pipeline/tdi/endmember_list.py [dir ...]      default dirs: runs/tdi
Not listed: tags with _rep (small-cell tests), runs whose stage0.json no longer matches their node files (a node file with index >= the number of lambda values the json declares: the stale-json hazard; restore the json first),
runs without a stage0.json. The list does NOT depend on how far a run is, so the index of a tag never changes while a batch is running (extending a finished run is a no-op)."""
import glob, json, os, sys

def eligible(d):
    for f in sorted(glob.glob(os.path.join(d, "*_stage0.json"))):
        base = f[:-len("_stage0.json")]; tag = os.path.basename(base)
        if "_rep" in tag or "tdi_test" in base: continue
        try: a = json.load(open(f))["args"]
        except Exception: continue
        ph = a.get("phase"); h = float(a.get("hper", 2.0))
        if abs(float(a.get("dt", 0)) - 0.5) > 1e-9: continue
        if not (ph == "H" or (ph in ("fcc", "bcc", "liquid") and abs(h - 2.0) < 1e-9)): continue
        if os.path.exists(f"{base}_node{int(a['nodes'])}.npy"): continue   # more node files than the json declares: stale json
        yield base

if __name__ == "__main__":
    for d in (sys.argv[1:] or ["runs/tdi"]):
        for b in eligible(d): print(b)
