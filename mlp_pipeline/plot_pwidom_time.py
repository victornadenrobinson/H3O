#!/usr/bin/env python3
"""Time-series diagnostics of finished PWidom runs (no GPU, no new MD). One figure per run dir, one column per host:
  row 1  MD trace T_kin and E_pot/atom vs time INCLUDING the equilibration (host MD and host + 1 H MD); only for runs made after the trace recording was added (<tag>_host_trace.npz), else 'no trace'
  row 2  E_pot/atom and MD pressure of the sampled host frames vs sampling time (always available), with the target pressure
  row 3  running insertion (Widom) and deletion estimates of mu_ex vs number of frames used, the BAR value (dashed) and the block values (dots): are the estimators settling, and do they bracket BAR?
   python3 mlp_pipeline/plot_pwidom_time.py runs/pwidom_endmem_new_500GPa_5000K [more run dirs] [-o prefix] [--blocks 4]"""
import argparse, glob, json, os
import numpy as np
KB = 8.617333262e-5
lse = lambda x: np.max(x) + np.log(np.sum(np.exp(x - np.max(x))))


def host_files(run):
    out = []
    for jf in sorted(glob.glob(os.path.join(run, "*.json"))):
        base = jf[:-5]
        if os.path.exists(base + "_work.npz") and os.path.exists(base + "_energies.npz"): out.append((jf, base))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("runs", nargs="+"); ap.add_argument("-o", default=None); ap.add_argument("--blocks", type=int, default=4); a = ap.parse_args()
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    for run in a.runs:
        hf = host_files(run)
        if not hf: print(f"{run}: no finished hosts"); continue
        fig, ax = plt.subplots(3, len(hf), figsize=(4.6 * len(hf), 9.5), squeeze=False, constrained_layout=True)
        for c, (jf, base) in enumerate(hf):
            r = json.load(open(jf)); kT = KB * r["T"]; tag = r["tag"]; W = np.load(base + "_work.npz", allow_pickle=True); n0 = r["n_frames_host"]
            wf = np.asarray(W["forward_work_eV"]); wr = -np.asarray(W["reverse_work_eV"]); dt_f = float(W["frame_interval_fs"]) / 1000
            E = np.load(base + "_energies.npz")["energies"]; e0 = E[:n0] / r["natoms"] * 1e3; fid = np.asarray(W["host_frame_ids"]).astype(int); t = (fid + 1) * dt_f
            ff = os.path.join(run, tag + "_frames.npz"); P = np.load(ff)["P_host"][fid] if os.path.exists(ff) else None
            ttl = f"{tag.replace('pw_', '').replace('_500GPa_5000K', '')}\nKish {r['kish_frac_ins']:.2f}, melted {r['melted']}"; ax[0, c].set_title(ttl, fontsize=9)
            # row 1: MD trace
            got = False
            for lab, col, ls in (("host", "#2a7fa8", "-"), ("host + 1 H", "#c98a1b", "-")):
                tf = os.path.join(run, tag + ("_host_trace.npz" if lab == "host" else "_full_trace.npz"))
                if not os.path.exists(tf): continue
                tr = np.load(tf)["trace"]
                if not len(tr): continue
                got = True; n_at = r["natoms"] + (0 if lab == "host" else 1); tp = tr[:, 0] / 1000
                ax[0, c].plot(tp, tr[:, 1], color=col, lw=0.7, label=f"T_kin {lab}"); a2 = ax[0, c].twinx() if lab == "host" else a2
                a2.plot(tp, tr[:, 2] / n_at * 1e3, color=col, lw=0.7, ls="--")
            if got:
                ax[0, c].axhline(r["T"], color="k", lw=0.8); ax[0, c].set_xlabel("MD time (ps), incl. equilibration"); ax[0, c].set_ylabel("T_kin (K, solid)   E_pot meV/atom (dashed, right)"); ax[0, c].legend(fontsize=7)
            else: ax[0, c].text(0.5, 0.5, "no MD trace\n(run predates trace recording)", ha="center", va="center", transform=ax[0, c].transAxes, color="0.5"); ax[0, c].set_xticks([]); ax[0, c].set_yticks([])
            # row 2: sampled frames
            ax[1, c].plot(t, e0, "o-", ms=3, color="#d55181", lw=0.7); ax[1, c].set_xlabel("sampling time (ps)"); ax[1, c].set_ylabel("E_pot meV/atom (pink)"); a3 = ax[1, c].twinx()
            if P is not None: a3.plot(t, P, "s-", ms=3, color="#5b8f3a", lw=0.7); a3.axhline(r["P_target"], color="#5b8f3a", lw=0.8, ls=":"); a3.set_ylabel("P host GPa (green)")
            # row 3: running estimators
            k = np.arange(1, n0 + 1); cw = [-kT * (lse(-wf[:i].ravel() / kT) - np.log(wf[:i].size)) * 1e3 for i in k]
            nd = len(wr); kd = np.arange(1, nd + 1); cd = [kT * (lse(wr[:i].ravel() / kT) - np.log(wr[:i].size)) * 1e3 for i in kd]
            ax[2, c].plot(k, cw, color="#d55181", label="Widom (running)"); ax[2, c].plot(kd, cd, color="#2a7fa8", label="deletion (running)"); ax[2, c].axhline(r["mu_ex_bar_meV"], color="k", ls="--", lw=1, label=f"BAR {r['mu_ex_bar_meV']:.0f}")
            for ii in np.array_split(np.arange(n0), a.blocks):
                w = wf[ii].ravel(); ax[2, c].plot([ii[-1] + 1], [-kT * (lse(-w / kT) - np.log(len(w))) * 1e3], "o", color="#d55181", mfc="none", ms=6)
            ax[2, c].set_xlabel("host / full frames used"); ax[2, c].set_ylabel("mu_ex (meV)"); ax[2, c].legend(fontsize=7); ax[2, c].grid(alpha=0.25)
        fig.suptitle(f"{os.path.basename(os.path.normpath(run))}", fontsize=11); out = (a.o or run.rstrip("/")) + "_time.png"; fig.savefig(out, dpi=120); print("figure ->", out)


if __name__ == "__main__":
    main()
