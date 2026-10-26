"""Build the young_tdi_par.sh command that EXTENDS a finished (or unfinished) TDI run in time, from the settings stored in its stage0 json (so dt, chunk, equil, k, nodes, rule, cell, composition all match the original run).
usage:  python3 mlp_pipeline/tdi/extend.py cmd runs/tdi/<tag> PS      -> one shell line:  env ... bash mlp_pipeline/young_tdi_par.sh <phase>   (PS = new ps per lambda; never shortened below the original)
        python3 mlp_pipeline/tdi/extend.py list [--below PS]           -> every run with fewer ps per lambda than PS (default 3), with its dir/tag, nodes, ps, N, and whether its files changed in the last 30 min ('ACTIVE': do not extend, another job is probably running it)
Extension in time works because every lambda keeps its state in <tag>_node<k>_state.npz with a fixed CHUNK: the same CHUNK and a larger PS continue from the last chunk (the RNG stream restarts from the seed, so the new part is not a bit-exact continuation)."""
import glob, json, os, sys, time

def load(base):
    j = json.load(open(base + "_stage0.json")); return j["args"], j

def cmd(base, ps):
    a, j = load(base); ph = a["phase"]; ps = max(float(ps), float(a["ps"])); nodes = int(a["nodes"])
    npar = nodes if nodes <= 9 else -(-nodes // -(-nodes // 9)); npar = min(npar, 8) if ph == "H" else npar
    env = dict(P=a["P"], T=a["T"], HPER=a.get("hper", 2.0), NODES=nodes, NPAR=npar, DT=a["dt"], EQUIL=a["equil_ps"], PS=ps, PS1=a["ps1"], CHUNK=a["chunk_ps"], POWER=a.get("power", 4.0),
               OUTDIR=os.path.dirname(base) or "runs/tdi")
    for k, key in (("NGHOST", "nghost"), ("NATOMS", "natoms"), ("NREP", "nrep"), ("K", "k"), ("MINRUNS", "min_runs"), ("NGAUSS", "ngauss")):
        if a.get(key) not in (None, 0): env[k] = a[key]
    if a.get("rule") == "gk": env["RULE"] = "gk"
    return " ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in env.items()) + f" bash mlp_pipeline/young_tdi_par.sh {ph}"

if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "cmd": print(cmd(sys.argv[2], sys.argv[3]))
    elif sys.argv[1] == "list":
        below = float(sys.argv[sys.argv.index("--below") + 1]) if "--below" in sys.argv else 3.0
        for f in sorted(glob.glob("runs/tdi*/*_stage0.json")):
            base = f[:-len("_stage0.json")]
            if "tdi_test" in base: continue
            try: a, j = load(base)
            except Exception: continue
            newest = max([os.path.getmtime(x) for x in glob.glob(base + "_node*")] or [0]); act = "ACTIVE" if time.time() - newest < 1800 else ""
            if float(a["ps"]) < below: print(f"{base}  {a['nodes']} lambda  {a['ps']} ps  N {j.get('N')}  {act}")
