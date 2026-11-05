#!/bin/bash
# LOW-PRESSURE LIQUID snapshots (geometries only: DFT labels come later) from GRACE MD, in parallel on the GPUs of the current allocation. One short MD per (composition, T, pressure) volume.
#   bash mlp_pipeline/lowp_snapshots.sh              prints the plan and a time estimate (nothing runs)
#   RUN=1 bash mlp_pipeline/lowp_snapshots.sh        runs it (4-GPU interactive job; NG GPUs x PERGPU workers each)
# Volumes come from eos_low_p.py (extrapolated below the measured range: the DFT labels do not care, the geometries only need to be liquid-like at roughly the right density; check P in each thermo.csv).
# The liquid branch is forced (the O lattice is melted at 12000 K for 1 ps first, then cooled to T). Output: runs/eos_lowp3/<tag>_v<vpa>/{thermo.csv, traj.extxyz}; frames every 20 fs after equilibration.
# NOTE: at these P, T pure H is MOLECULAR (H2): keep small pure-H cells (the 500-atom rule is for metallic H), expect extrapolated volumes to be off for H, and remember single-H insertion (PWidom) is not meaningful there (H2 insertion would be).
# HPERS ("2 2.25 2.5 3 4 5 6 H": H2O, H2.25O, H2.5O, H3O, H4O, H5O, H6O, pure H; hper 6 is the most the fcc O cell can hold, one proton per O-O bond midpoint): compositions = H per O values, H = pure hydrogen with HATOMS atoms (686: even, see above); compositions without EOS-scan data get ideal-mixing volumes.
# COLD=1 (default): also a small COLD set, T = 1000 and 2000 K (COLDT) at 10, 30, 50 GPa (COLDP), compositions "2 4 6 H" (COLDHPERS), solid branch, 0.4 ps (COLDPS).
# env: PS ("10 25 50 100" GPa), MODEL (best fine-tune), MDPS (0.6 ps sampling + 0.3 equil), TS (only these temperatures, e.g. "3000 5000"), PERGPU (3), CMDMIN (minutes allowed per MD, 12), OUT (runs/lowp3), PY, PYRUN, EOSLOWP
cd "$(dirname "$0")/.."; PY=${PY:-$SCRATCH/grace-env/bin/python}; PYRUN=${PYRUN:-$PY}; PS=${PS:-"10 25 50 100"}; MODEL=${MODEL:-grace-ft:runs/grace/ft_mn_f_lr1e-2_e20/seed/1/final_model}; MDPS=${MDPS:-0.6}
PERGPU=${PERGPU:-3}; CMDMIN=${CMDMIN:-12}; OUT=${OUT:-runs/lowp3}; EOSLOWP=${EOSLOWP:-mlp_pipeline/eos_low_p.py}; mkdir -p "$OUT" logs_pw
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1; NW=$((NG * PERGPU))
$PY $EOSLOWP runs/eos_scan runs/eos_lowp runs/eos_lowp2 --P $PS --model "$MODEL" --ps "$MDPS" > "$OUT/cmds_raw.txt" 2> "$OUT/table.txt" || { echo "eos_low_p.py failed: see $OUT/table.txt"; tail -5 "$OUT/table.txt"; exit 1; }
grep '^python' "$OUT/cmds_raw.txt" | sed -E 's/--branch [a-z]+/--branch liquid/; s/_solid_/_liquid_/; s#--outdir [^ ]+#--outdir runs/eos_lowp3#' | sort -u > "$OUT/cmds_liquid.txt"
if [ -n "${TS:-}" ]; then pat=$(echo $TS | sed 's/ /|/g'); grep -E -- "--T ($pat) " "$OUT/cmds_liquid.txt" > "$OUT/cmds_T.txt" || true; mv "$OUT/cmds_T.txt" "$OUT/cmds_liquid.txt"; fi
$PY - "$OUT" "${HPERS:-2 2.25 2.5 3 4 5 6 H}" "${HATOMS:-686}" <<'PYEOF'
# composition selection for the liquid hull: keep the requested H-per-O values (and 'H' = pure H); a composition with no line from eos_low_p.py (no MLP EOS scan data) gets its volumes by IDEAL MIXING of the H2O and pure-H volumes at the same
# pressure (per O: 3 v_H2O + (hper-2) v_H over 1+hper atoms); cells: 108 O (nrep 3) up to H2.25O, 32 O (nrep 2) beyond; pure H = HATOMS atoms (default 686 = 2 x 7^3: an EVEN count, so molecular H2 has no unpaired atom; 729 is odd).
import re, sys
out, want, natoms = sys.argv[1], sys.argv[2].split(), int(sys.argv[3]); lines = [l.rstrip("\n") for l in open(f"{out}/cmds_liquid.txt") if l.strip()]
get = lambda l, k: re.search(rf"--{k} (\S+)", l).group(1)
vpas = lambda l: [float(x) for x in re.search(r"--vpa ((?:[0-9.]+ ?)+)", l).group(1).split()]
by = {}
for l in lines:
    ph = get(l, "phase"); key = ("H", 0.0, get(l, "T")) if ph == "H" else ("fcc", float(get(l, "hper")), get(l, "T")); by[key] = l
res = []; Ts = sorted({k[2] for k in by}, key=float)
for T in Ts:
    w, h = by.get(("fcc", 2.0, T)), by.get(("H", 0.0, T))
    for x in want:
        if x == "H":
            if h: res.append(h + f" --natoms {natoms}")
            continue
        X = float(x); l = by.get(("fcc", X, T))
        if l is None and w is not None and h is not None:
            vw, vh = vpas(w), vpas(h); v = [(3 * a + (X - 2) * b) / (1 + X) for a, b in zip(vw, vh)]
            l = re.sub(r"--hper \S+", f"--hper {X:g}", w); l = re.sub(r"--vpa (?:[0-9.]+ ?)+", "--vpa " + " ".join(f"{q:.4f}" for q in v) + " ", l); l = re.sub(r"--tag \S+", f"--tag lowP_fcc_h{X:g}_liquid_{T}K", l)
        if l is None: print(f"  no line for H/O {x} at {T} K (and no H2O / H lines to mix)"); continue
        res.append(re.sub(r"--nrep \d+", "--nrep 2" if X >= 2.5 else "--nrep 3", l))
open(f"{out}/cmds_liquid.txt", "w").write("\n".join(res) + "\n")
import collections; c = collections.Counter(("H" if "--phase H" in l else "H/O " + get(l, "hper")) for l in res); print("compositions: " + ", ".join(f"{k} x{v} T" for k, v in sorted(c.items())) + f";  {len(res)} commands")
PYEOF
# ---- COLD set (default on): 1000 and 2000 K at 10, 30, 50 GPa (a few compositions, solid branch = lattice start + short MD). The EOS scan has no data below 3000 K, so the volumes are those of the LOWEST scanned T at the same pressure
# (the cold pressure will come out lower than the target: see the pressure printout), copied to each cold T.  COLD=0 skips it; COLDP, COLDT, COLDHPERS ("2 4 6 H"), COLDPS (0.4 ps) change it.
if [ "${COLD:-1}" = 1 ]; then
  COLDP=${COLDP:-"10 30 50"}; COLDT=${COLDT:-"1000 2000"}; COLDHPERS=${COLDHPERS:-"2 4 6 H"}; COLDPS=${COLDPS:-0.4}
  if $PY $EOSLOWP runs/eos_scan runs/eos_lowp runs/eos_lowp2 --P $COLDP --model "$MODEL" --ps "$COLDPS" > "$OUT/cmds_raw_cold.txt" 2>> "$OUT/table.txt"; then
    grep '^python' "$OUT/cmds_raw_cold.txt" | sed -E 's#--outdir [^ ]+#--outdir runs/eos_lowp3#' | sort -u > "$OUT/cmds_cold_base.txt"
    $PY - "$OUT" "$COLDHPERS" "${HATOMS:-686}" "$COLDT" <<'PYEOF'
import re, sys
out, want, natoms, Tc = sys.argv[1], sys.argv[2].split(), int(sys.argv[3]), sys.argv[4].split(); lines = [l.rstrip("\n") for l in open(f"{out}/cmds_cold_base.txt") if l.strip()]
get = lambda l, k: re.search(rf"--{k} (\S+)", l).group(1)
vpas = lambda l: [float(x) for x in re.search(r"--vpa ((?:[0-9.]+ ?)+)", l).group(1).split()]
if not lines: print("cold set: no base commands from eos_low_p.py"); open(f"{out}/cmds_cold.txt", "w").write(""); sys.exit(0)
T0 = min({get(l, "T") for l in lines}, key=float); by = {}                          # volumes from the lowest scanned temperature
for l in lines:
    if get(l, "T") == T0: by[("H", 0.0) if get(l, "phase") == "H" else ("fcc", float(get(l, "hper")))] = l
w, h = by.get(("fcc", 2.0)), by.get(("H", 0.0)); res = []
for x in want:
    if x == "H": base = (h + f" --natoms {natoms}") if h else None
    else:
        X = float(x); base = by.get(("fcc", X))
        if base is None and w is not None and h is not None:
            v = [(3 * a + (X - 2) * b) / (1 + X) for a, b in zip(vpas(w), vpas(h))]
            base = re.sub(r"--hper \S+", f"--hper {X:g}", w); base = re.sub(r"--vpa (?:[0-9.]+ ?)+", "--vpa " + " ".join(f"{q:.4f}" for q in v) + " ", base); base = re.sub(r"--tag \S+", f"--tag lowP_fcc_h{X:g}_solid_{T0}K", base)
        if base is not None: base = re.sub(r"--nrep \d+", "--nrep 2" if X >= 2.5 else "--nrep 3", base)
    if base is None: print(f"cold set: no line for {x}"); continue
    for T in Tc: res.append(re.sub(r"--T \S+", f"--T {T}", re.sub(rf"_{T0}K\b", f"_{T}K", base)))
open(f"{out}/cmds_cold.txt", "w").write("\n".join(res) + "\n"); print(f"cold set: {len(res)} commands (compositions {' '.join(want)}, T {' '.join(Tc)} K, volumes from {T0} K)")
PYEOF
    cat "$OUT/cmds_cold.txt" >> "$OUT/cmds_liquid.txt"
  else echo "cold set skipped: eos_low_p.py failed for the cold pressures (see $OUT/table.txt)"; fi
fi
$PY - "$OUT" "$NW" "$PYRUN" <<'PYEOF'
import sys, re
out, nw, py = sys.argv[1], int(sys.argv[2]), sys.argv[3]; cmds = []
for l in open(f"{out}/cmds_liquid.txt"):
    t = l.split(); i = t.index("--vpa"); j = next((k for k in range(i + 1, len(t)) if t[k].startswith("--")), len(t)); nv = j - i - 1
    for k in range(nv): cmds.append(" ".join([py] + t[1:]) + f" --only {k}")
work = [[] for _ in range(nw)]
for n, c in enumerate(cmds): work[n % nw].append(c)
for w, cs in enumerate(work): open(f"{out}/worker_{w}.txt", "w").write("\n".join(cs) + ("\n" if cs else ""))
print(f"{len(cmds)} MD commands (liquid branch) -> {nw} workers, up to {max(len(c) for c in work)} each")
PYEOF
NC=$(cat "$OUT"/worker_*.txt | grep -c .); MAXW=$(wc -l "$OUT"/worker_*.txt | grep -v total | sort -n | tail -1 | awk '{print $1}')
echo "plan: $NC volumes, $NW workers ($NG GPU x $PERGPU), <= $MAXW MDs per worker; at ~4 min per MD with $PERGPU per GPU: ~$((MAXW * 4)) min   (table of volumes: $OUT/table.txt)"
if [ "${RUN:-0}" != 1 ]; then echo "(plan only: add RUN=1)"; head -3 "$OUT/cmds_liquid.txt" | cut -c1-230; exit 0; fi
for ((w = 0; w < NW; w++)); do
  ( g=$((w % NG)); while IFS= read -r c <&3; do [ -n "$c" ] || continue; CUDA_VISIBLE_DEVICES=$g timeout ${CMDMIN}m bash -c "$c" < /dev/null >> "$OUT/worker_$w.log" 2>&1 || echo "worker $w: FAILED or timed out: ${c:0:160}" >> "$OUT/failed.txt"; echo "[worker $w gpu $g] done $(echo "$c" | grep -o -- '--tag [^ ]*') $(echo "$c" | grep -o -- '--only [0-9]*')"; done 3< "$OUT/worker_$w.txt" ) &
done; wait
$PY - <<'PYEOF'
import glob, os
import pandas as pd
rows = []
for f in sorted(glob.glob("runs/eos_lowp3/*_v*/thermo.csv")):
    try:
        t = pd.read_csv(f); col = next((c for c in t.columns if c.lower() in ("p_gpa", "p", "pressure_gpa")), None)
        if col: rows.append((os.path.basename(os.path.dirname(f)), float(t[col].iloc[len(t) // 2:].mean())))
    except Exception: pass
print("mean pressure of each volume (second half of the MD), GPa -- compare with the 10/25/50/100 targets (EOS extrapolated; molecular H2 volumes in particular can be far off):")
for n, P in rows[:60]: print(f"   {P:8.1f}  {n}")
PYEOF
echo "finished. snapshots: $(ls -d runs/eos_lowp3/*_v* 2>/dev/null | wc -l) volume folders; failures: $(cat "$OUT/failed.txt" 2>/dev/null | wc -l) (see $OUT/failed.txt)"
