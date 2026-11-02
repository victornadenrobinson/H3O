#!/usr/bin/env bash
# Run on YOUNG (repo root): pick all DFT-labelling candidates out of the Young GRACE data and pack everything Perlmutter needs into young_pack.tgz.
# usage:  bash mlp_pipeline/young_pack.sh            (INV=/home/mmm0037/young_inventory.csv  PYS=python3  can be overridden)
# Then on Perlmutter (repo root):  scp mmm0037@young.rc.ucl.ac.uk:/lustre/scratch/mmm0037/H3O/young_pack.tgz . && tar xzf young_pack.tgz     (or go through your laptop / Globus)
# Contents: sel_pwidom / sel_meltcomp / sel_eos .extxyz (DFT candidates), hosts_pool / hosts_pwidom .extxyz (insertion hosts), low_p_pass1.sh (17_eos_md commands at 50/100/200 GPa),
# runs/h3o_static/P*/ relaxed structures (Huang H3O supercells), runs/eos_scan + eos_map csvs (for the second pass of eos_low_p.py), the inventory csv.
set -eu
INV=${INV:-/home/mmm0037/young_inventory.csv}; PYS=${PYS:-python3}; SEL="$PYS mlp_pipeline/young_select.py $INV --runs runs"
$SEL --kind pwidom-host pwidom-full --P 400 600 --T 4000 6000 --ratio 1.99 3.0 --per-file 4 --out sel_pwidom.extxyz
$SEL --kind extxyz --path-re 'melt_comp/.*_(free|hold)_' --T 4000 6000 --P 300 600 --ratio 2.0 3.0 --per-file 2 --out sel_meltcomp.extxyz
$SEL --kind extxyz --path-re 'eos_scan/' --ratio 2.2 3.1 --max-vpa 2.4 --per-file 2 --out sel_eos.extxyz
$SEL --kind extxyz --path-re 'melt/melt_fcc_' --ratio 1.99 2.01 --per-file 4 --as-hosts --out hosts_pool.extxyz
$SEL --kind pwidom-host --ratio 1.99 2.01 --per-file 4 --as-hosts --out hosts_pwidom.extxyz
if [ ! -s low_p_pass1.sh ]; then $PYS mlp_pipeline/eos_low_p.py runs/eos_scan runs/eos_map 2>/dev/null | grep -v -- "--phase H\|--T 9000" > low_p_pass1.sh; fi
echo "low_p_pass1.sh: $(wc -l < low_p_pass1.sh) commands"
files=$(ls sel_*.extxyz hosts_*.extxyz low_p_pass1.sh); [ -f "$INV" ] && cp "$INV" young_inventory.csv && files="$files young_inventory.csv"
extra=$(find runs/h3o_static -name '*_relaxed.extxyz' 2>/dev/null; ls runs/eos_scan/eos_*.csv runs/eos_map/eos_*.csv 2>/dev/null)
tar czf young_pack.tgz $files $extra; ls -l young_pack.tgz; tar tzf young_pack.tgz | wc -l
