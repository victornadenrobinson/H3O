#!/bin/bash
# LOW-PRESSURE LIQUID snapshots (geometries only: DFT labels come later) from GRACE MD, in parallel on the GPUs of the current allocation. One short MD per (composition, T, pressure) volume.
#   bash mlp_pipeline/lowp_snapshots.sh              prints the plan and a time estimate (nothing runs)
#   RUN=1 bash mlp_pipeline/lowp_snapshots.sh        runs it (4-GPU interactive job; NG GPUs x PERGPU workers each)
# Volumes come from eos_low_p.py (extrapolated below the measured range: the DFT labels do not care, the geometries only need to be liquid-like at roughly the right density; check P in each thermo.csv).
# The liquid branch is forced (the O lattice is melted at 12000 K for 1 ps first, then cooled to T). Output: runs/eos_lowp3/<tag>_v<vpa>/{thermo.csv, traj.extxyz}; frames every 20 fs after equilibration.
# NOTE: at these P, T pure H is MOLECULAR (H2): keep small pure-H cells (the 500-atom rule is for metallic H), expect extrapolated volumes to be off for H, and remember single-H insertion (PWidom) is not meaningful there (H2 insertion would be).
# env: PS ("10 25 50 100" GPa), MODEL (best fine-tune), MDPS (0.6 ps sampling + 0.3 equil), TS (only these temperatures, e.g. "3000 5000"), PERGPU (3), CMDMIN (minutes allowed per MD, 12), OUT (runs/lowp3), PY, PYRUN, EOSLOWP
cd "$(dirname "$0")/.."; PY=${PY:-$SCRATCH/grace-env/bin/python}; PYRUN=${PYRUN:-$PY}; PS=${PS:-"10 25 50 100"}; MODEL=${MODEL:-grace-ft:runs/grace/ft_mn_f_lr1e-2_e20/seed/1/final_model}; MDPS=${MDPS:-0.6}
PERGPU=${PERGPU:-3}; CMDMIN=${CMDMIN:-12}; OUT=${OUT:-runs/lowp3}; EOSLOWP=${EOSLOWP:-mlp_pipeline/eos_low_p.py}; mkdir -p "$OUT" logs_pw
NG=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NG" -ge 1 ] || NG=1; NW=$((NG * PERGPU))
$PY $EOSLOWP runs/eos_scan runs/eos_lowp runs/eos_lowp2 --P $PS --model "$MODEL" --ps "$MDPS" > "$OUT/cmds_raw.txt" 2> "$OUT/table.txt" || { echo "eos_low_p.py failed: see $OUT/table.txt"; tail -5 "$OUT/table.txt"; exit 1; }
grep '^python' "$OUT/cmds_raw.txt" | sed -E 's/--branch [a-z]+/--branch liquid/; s/_solid_/_liquid_/; s#--outdir [^ ]+#--outdir runs/eos_lowp3#' | sort -u > "$OUT/cmds_liquid.txt"
if [ -n "${TS:-}" ]; then pat=$(echo $TS | sed 's/ /|/g'); grep -E -- "--T ($pat) " "$OUT/cmds_liquid.txt" > "$OUT/cmds_T.txt" || true; mv "$OUT/cmds_T.txt" "$OUT/cmds_liquid.txt"; fi
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
