#!/bin/bash
# Quick sign / trend scan of dg = mu_H(H2O ice) - G_H(pure H) over many (P, T) with the toy pair potential (CPU, no GPU): per state
#   1. toy_validate.py eos for H2O ice and pure H at (P, T), with a melt check (O mean-square displacement over 1 ps); a melted ice state is skipped
#   2. the production launcher young_tdi_par.sh: insertion of 1 H into the H2O crystal and into pure H (10 lambda values x PS ps)
#   3. dg = G(insert into H2O) - [G(insert into pure H) - fs], fs = the ideal-gas finite-size term of insertion into N = 128 H (T3 of toy_validate.py)
# usage (repo root, a python with ASE; TOYICE_FIT=docs/data/toyfit_t4.json for the fitted potential):
#   bash mlp_pipeline/tdi/toy_scan.sh "400:4000 400:5000 500:5000" [outdir]      (P:T pairs, GPa:K; finished states are skipped, so rerunning continues)
# Result lines: <outdir>/scan.csv (P_GPa,T_K,vpa_h2o,vpa_H,msd_O,dg_meV,err_meV,mu_ice_meV,mu_Hhost_meV) and "SCAN P T dg +- err" on stdout.
STATES=${1:?"P:T pairs, e.g. \"400:4000 500:5000\""}; R=${2:-runs/toyscan}; mkdir -p $R
export MODEL=toyice ALLOW_NOGPU=1 PY=${PY:-python3} NODES=${NODES:-10} PS=${PS:-1} EQUIL=${EQUIL:-0.25} CHUNK=${CHUNK:-0.25} DT=0.5 PS1=${PS1:-0.5} NPAR=${NPAR:-4}
[ -f $R/scan.csv ] || echo "P_GPa,T_K,vpa_h2o,vpa_H,msd_O,dg_meV,err_meV,mu_ice_meV,mu_Hhost_meV" > $R/scan.csv
for s in $STATES; do
  P=${s%%:*}; T=${s##*:}; D=$R/P${P}_T${T}; mkdir -p $D; export P T OUTDIR=$D/tdi
  grep -q "^$P,$T," $R/scan.csv && { echo "SCAN $P GPa $T K: already in $R/scan.csv"; continue; }
  echo "== $P GPa $T K: eos + melt check $(date +%H:%M)"
  [ -f $D/eos_${T}K.json ] || $PY mlp_pipeline/tdi/toy_validate.py eos --T $T --P $P --systems h2o,H --out $D > $D/eos.log 2>&1 || { echo "SCAN $P GPa $T K: eos FAILED (see $D/eos.log)"; continue; }
  read VW BW VH BH MSD SOLID < <(python3 -c "import json; d=json.load(open('$D/eos_${T}K.json')); w, h = d['h2o'], d['H']; print(round(w['vpa'], 5), round(w['B_GPa'], 2), round(h['vpa'], 5), round(h['B_GPa'], 2), round(w.get('msd_O_1ps', -1), 3), int(w.get('solid', 1)))")
  if [ "$SOLID" != 1 ]; then echo "SCAN $P GPa $T K: ice MELTED in the eos MD (O MSD $MSD A^2 over 1 ps): skipped"; echo "$P,$T,$VW,$VH,$MSD,melted,,," >> $R/scan.csv; continue; fi
  echo "   H2O vpa $VW (O MSD $MSD A^2: solid), H vpa $VH; insertion runs $(date +%H:%M)"
  env VPA=$VW NREP=2 BULK=$BW NGHOST=1 bash mlp_pipeline/young_tdi_par.sh insert > $D/launch_ins1.log 2>&1 || { echo "SCAN $P GPa $T K: insertion into H2O FAILED (see $D/launch_ins1.log)"; continue; }
  env VPA=$VH NATOMS=128 BULK=$BH NGHOST=1 HOSTPHASE=H bash mlp_pipeline/young_tdi_par.sh insert > $D/launch_insH.log 2>&1 || { echo "SCAN $P GPa $T K: insertion into pure H FAILED (see $D/launch_insH.log)"; continue; }
  python3 - $D/tdi $P $T $VW $VH $MSD $R/scan.csv <<'EOF'
import glob, json, sys
from math import lgamma, log, hypot
d, P, T, vw, vh, msd, csv = sys.argv[1:]; T = float(T)
j = {("H" if "_Hhost" in f else "w"): json.load(open(f)) for f in glob.glob(f"{d}/tdi_insert_*_n1*.json")}
w, h = j["w"], j["H"]; N = h["nH"] - 1; kT = 8.617333262e-5 * T * 1e3
fs = kT * (log(N + 1) - lgamma(N + 1) / N - 1)   # mu(N -> N+1) - G/N for an ideal gas, meV (toy_validate T3)
dg = w["G_meV_atom"] - (h["G_meV_atom"] - fs); e = hypot(w["G_err_meV_atom"], h["G_err_meV_atom"])
open(csv, "a").write(f"{P},{T:.0f},{vw},{vh},{msd},{dg:.1f},{e:.1f},{w['G_meV_atom']:.1f},{h['G_meV_atom']:.1f}\n")
print(f"SCAN {P} GPa {T:.0f} K: dg = mu_H(ice) - G_H = {dg:7.1f} +- {e:5.1f} meV  ({'favourable' if dg + 2 * e < 0 else 'unfavourable' if dg - 2 * e > 0 else 'not resolved'} at 2 sigma)", flush=True)
EOF
done
echo "== scan done $(date +%H:%M)"; sed "s/,/\t/g" $R/scan.csv
