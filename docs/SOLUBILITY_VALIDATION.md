# Validating the solubility of H in superionic fcc H2O (GRACE TDI): results of the offline checks and the ordered Young checklist

Session 2026-10-29, branch `claude/eloquent-hopper-kfjdei`, follows the list in CURRENT_ISSUES.md section 0 in its order. The sandbox has no GPU, no Young data and no DFT, so this session did two things:
(1) everything about the existing numbers that can be decided by arithmetic, error propagation and toy models (section 1), and (2) wrote, toy-tested and documented the tools for the checks that need Young / VASP (section 2, exact commands).
Code review and flow diagram: docs/CODEBASE.md. Tools: `mlp_pipeline/tdi/solubility_check.py` (new), `dft_check.py prep-static` (new), `stability_npt.py` (fixed), `toy_solubility.sh` (new).

## 1. What the existing GRACE numbers say, and how far they can be trusted (no new MD)
Reproduce with `python3 mlp_pipeline/tdi/solubility_check.py` (sections A1-A5, ~1 s; `-o` writes the n* table to csv).

**Definition and algebra (checked).** a(n) = average dG per added H = mu0 + s n (n = added H per O, meV); G(n) = n a(n) per O; dG/dn = mu0 + 2 s n; solubility n* = -mu0/(2 s), minimum G = -mu0^2/(4 s). The zero of a(n) is at n0 = 2 n*
(the old "-mu0/s" was this n0). s is per unit H/O in `insertion_regsol_fit.csv` (divide by 108 for meV per added H in the 108-O cell). Closure: the fit reproduces the direct H2.25O crystals at n = 0.25 within 1.7 sigma
(400/5000: fit 236 vs 199-252; 500/5000: 92 vs 41-108; 500/5500: 146 vs 141; 500/6000: 46 vs 33-56 meV per H), as it must, because those points pin s.

**Result (regular solution, statistical errors only).** n* = 0.081 (400 GPa 5000 K), 0.107 / 0.097 / 0.114 (500 GPa 5000 / 5500 / 6000 K): H2.08O ... H2.11O, minimum of the formation free energy -6 ... -9 meV/atom, mu0/kT = -1.0 ... -1.2.
Four states (400/6000, 400/6500, 500/6500, 500/7000) have s_err of 1200-1600 meV (two usable points in the fit) and 5-10 % of their Monte-Carlo draws have no minimum at all: their n* (0.063, 0.044, 0.127, 0.099) has 16-84 % intervals of a factor 2-9 (500/6500: 0.07-0.60). 300 GPa 5000 K gives 0.029 (s 6500 +- 1300) but the shape range is 0.006-0.097.

**What the data do NOT fix: the shape of G(n).** mu0 is the n -> 0 limit (insertion n = 1, 2, 4), s is pinned by a(0.25). The minimum lies at 0.08-0.11, between them, and nothing has been measured there.
G(n) = mu0 n + c n^p pinned at the same two ends gives n* (p = 1.5 / 2 / 3 / 4): 500 GPa 6000 K 0.093 / 0.114 / 0.138 / 0.153; 500/5000 0.081 / 0.107 / 0.133 / 0.149; **400/5000 0.047 / 0.081 / 0.116 / 0.136**; 300 GPa 0.006 / 0.029 / 0.069 / 0.097.
So within GRACE the solubility is 5-15 % extra H at 400-500 GPa, not better than a factor 1.5 (500 GPa) to 3 (400 GPa) until the curvature is measured. This is the largest uncertainty that does not need DFT.

**Model error (GRACE vs DFT), bracketed.** dn*/d(delta) = -1/(2 s) ~ -2e-4 per meV of mu0.
* the bias proxy from DFT frames (GRACE mu_H - G_H too high by +40 +- 15 meV; delta = -40 meV): n* +7-8 %;
* the STATIC offset of H3O (GRACE dH_f 90-100 meV/atom above Huang's static DFT = 380 meV per extra H), if it were a constant error per extra H (reading A): n* x1.7-2.2; if it only appears where the lattice is full (reading B, shifts s by -380): x1.1-1.2; half and half: x1.5. All three increase n* (GRACE under-binds extra H);
* H stays soluble (n* > 0.01) for any shift of mu0 below +250 (300-400 GPa) ... +490 (500 GPa) meV: the SIGN of the answer is not at risk, the size is.
* numerics (quadrature of insertion up to 20-46 meV per H in the toy, pressure-correction noise ~10-30 meV): a few % of n*, far below the shape and model-error items.
Order of importance for n*: shape of G(n) (x1.4-3) > model error of mu0 / s (x1.1-2) > statistics of mu0, s at the four tight states (+-5-10 %) > numerics (few %). ZPE / NQE (not in the model) is a further item of unknown size (CURRENT_ISSUES P5).

**Planning (what a new run resolves, 5 ps per lambda, nominal).** Insertion of n_add = 8 and 16 separates p = 2 from p = 3 and 1.5 by 3-6 sigma at 400-500 GPa; per-added-H error of insertion does not grow with n (assumed 40 meV at 2 ps), the crystal route has ~1150 meV / n_add at 1.5 ps and only competes at n_add >= 22.
These are the cheapest decisive runs, and they bracket n* from both sides (n_add = 8: 0.074, 16: 0.148 vs n* ~ 0.11).

## 2. Ordered checklist for Young (section-0 order). Every block starts with the pull; paste outputs back
Placeholders: `<TEMPLATE_DIR>` = a training VASP run directory with INCAR, KPOINTS, POTCAR and POSCAR (the one used for the GRACE training data; same for prep-fep); VASP runs on Young's SGE CPU nodes with `young_dft_check.sge` (templates on Young: `/lustre/scratch/mmm0037/dft_templates/{9.8,10.1}`, PAW_PBE H_h + O_h).
Never commit DFT frames (the repo is public): only the small csv tables named below go to git.

### Step 0. State of the repo, the queue and the existing numbers (login node)
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && bash mlp_pipeline/jobs_status.sh
```
Refit with everything that exists (login node; numpy + pandas only, ~1 min):
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && mkdir -p runs/checks && ~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py runs/tdi runs/tdi_long runs/tdi_prep runs/tdi_dil runs/tdi_ins_rep2 runs/tdi_ins_rep4 -o docs/data/tdi_results.csv > runs/checks/summarize_$(date +%F).txt 2>&1; ~/grace-venv/bin/python mlp_pipeline/tdi/solubility_check.py --mix docs/data/tdi_results_mix.csv -o runs/checks/solubility_mix.csv 2>&1 | tee runs/checks/solubility_$(date +%F).txt
```
Read in the output: which (P, T) have insertion n = 8, 16, 27 and crystals n = 11-27, whether "data below / above n*" says yes / yes, chi2 of the linear fit (a chi2/dof >> 1 = the regular solution is wrong there), n* from the free-curvature fit, rows that were dropped for a missing pressure correction.
Commit the small tables (not runs/): `git add docs/data/tdi_results*.csv && git commit -m "TDI summary tables 2026-10-29" && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && git push origin HEAD:claude/eloquent-hopper-kfjdei`.

### Step 1. Model error for extra H in ice (CURRENT_ISSUES 0.1, largest first)
**(a) CASTEP (you).** The H3O cell (`docs/data/h3o_huang.cell`) and ice X are already in `docs/data/*_castep_enthalpy.csv`. Missing and needed: H reference enthalpies per cell with the same CASTEP settings, at 300 / 400 / 500 / 600 GPa: Cs-IV (I41/amd, 4 atoms/cell, c/a ~ 2.5), C2/c-24, molecular hcp H2; and ice X / Pbcm above 400 GPa.
Put them in `docs/data/h_castep_enthalpy.csv` (`structure,P_GPa,natoms,H_eV_cell`). With them, dH_f(H3O) is a DFT number of your own, to be compared with GRACE on the same structures (step 1b prints the GRACE side).

**(b) GRACE vs VASP on the same relaxed structures (new tool, toy-tested).** First the GRACE relaxations (the relaxed structures are saved as `*_relaxed.extxyz`, the input of the DFT step): interactive GPU job
```bash
srun -p intergpu --qos=intergpu -A allusers --gres=gpu:1 --cpus-per-task=8 --mem=120G --time=04:00:00 --pty bash
```
inside it:
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && TF_FORCE_GPU_ALLOW_GROWTH=true ~/grace-venv/bin/python -u mlp_pipeline/h3o_static.py --P 400 500 --variants 3 2.75 2.5 2.25 --parallel 2 --redo -o runs/h3o_static 2>&1 | tee runs/h3o_static_$(date +%F).out | grep --line-buffered -v _handle_getgr
```
(all pressures of the earlier plan: `PS="300 350 400 450 500 600 700 800" VARIANTS=3 REDO=1 sbatch mlp_pipeline/young_h3o_static.sbatch`.) Then, login node, the single points (3 volumes each: the E(V) fit gives the enthalpy at the target pressure for DFT and GRACE alike; 30 single points):
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && ~/grace-venv/bin/python mlp_pipeline/tdi/dft_check.py prep-static runs/h3o_static --template <TEMPLATE_DIR> --P 400 500 --scales 0.98 1.0 1.02 --only iceX Href_H2_hcp Href_H_Cs-IV H3O H2.5O -o runs/dft_check/static
```
It prints each directory with its k-mesh (Gamma-centred, spacing 0.25 1/A, not the MD template mesh) and ends with the sbatch line; the settings are the template's (ENCUT, PREC, POTCAR) with ISMEAR = -1 and SIGMA = kB x 1000 K (`--tel`). Look at one INCAR and KPOINTS by eye before submitting:
```bash
cd /lustre/scratch/mmm0037/H3O && cat runs/dft_check/static/static_H3O_400GPa_s1.000/INCAR runs/dft_check/static/static_H3O_400GPa_s1.000/KPOINTS && qsub -V -P Gold -A UKCP_ED_P -pe smp 16 -l mem=4G,h_rt=6:00:00 -t 1-30 -tc 6 mlp_pipeline/young_dft_check.sge runs/dft_check/static
```
(adjust 1-30 to `wc -l < runs/dft_check/static/dirs.txt`; first run `dft_check.py prep-selftest --template <TEMPLATE_DIR> -o runs/dft_check/selftest` the same way to get the cost of one template-sized point; progress: `dft_check.py status <dir>`; Free queue: `-P Free -A UKCP_ED_P`). When done, in an interactive GPU job:
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && ~/grace-venv/bin/python mlp_pipeline/tdi/dft_check.py analyze runs/dft_check/static 2>&1 | tee runs/dft_check/static_analysis_$(date +%F).txt | grep --line-buffered -v _handle_getgr && cp runs/dft_check/static/static_results.csv docs/data/static_dft_results.csv
```
Reading (printed at the end): VASP minus GRACE dH_f of H3O and of the partly filled variants, per extra H. Decision: if VASP agrees with Huang's static curve and sits 90 meV/atom below GRACE, the static error is real (then FEP on hot frames tells how much of it is in mu0 and s, below); a VASP value that also sits above Huang means settings differ (ENCUT, smearing, PAW) and the comparison is void. The partly filled variants show whether the error per extra H is constant (reading A of section 1) or grows with filling (reading B).

**(c) The decisive test: FEP MLP -> DFT of mu_H - G_H at one state point (hot frames, VASP).** First 10 frames per ensemble (40 single points of 250 atoms, ~10-40 node-hours) to see the width of Delta = E_DFT - E_MLP; go to 20+ only if the exponential average is usable (sd < 2 kT per cell):
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && ~/grace-venv/bin/python mlp_pipeline/tdi/dft_check.py prep-fep runs/tdi_long/insert_500GPa_6000K_n1 --h-run runs/tdi_long/H_500GPa_6000K_N250 --template <TEMPLATE_DIR> --frames 10 -o runs/dft_check/fep
```
then `qsub -V -P Gold -A UKCP_ED_P -pe smp 16 -l mem=4G,h_rt=6:00:00 -t 1-40 -tc 8 mlp_pipeline/young_dft_check.sge runs/dft_check/fep` and the analyze line as above with `runs/dft_check/fep` (prints "DFT correction to mu_H - G_H").
Feed the result back: `python3 mlp_pipeline/tdi/solubility_check.py --bias <delta> <err>` puts the measured shift in the delta column of table A3 (n* with the correction); a delta below -100 meV moves n* by more than 20 %.
**(d)** If the correction is large and mu0 and s are affected differently: refit / fine-tune with these DFT frames (`grace_make_input.py --wE`, energies are nearly absent from the loss) and rerun mu0 and s.

### Step 2. The slope s and the shape of G(n) (CURRENT_ISSUES 0.2)
**(a) Insertion n = 8, 16, 27 (cheapest, decisive).** First see what exists (Step 0 prints it); then submit only the missing ones. Tags carry no run length, so an existing tag with other settings is refused loudly; use the original PS or extend. Example for the three best states, 5 ps, 12 lambda (login node):
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && ls runs/tdi/insert_*_n8_stage0.json runs/tdi/insert_*_n16_stage0.json runs/tdi/insert_*_n27_stage0.json 2>&1 | sed 's/_stage0.json//'
```
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && PS=5 NODES=12 JOBS="500 6000 8;500 6000 16;500 6000 27;500 5000 8;500 5000 16;400 5000 8;400 5000 16;400 5000 27" bash mlp_pipeline/submit.sh "insertion n=8,16,27 for the shape of G(n)" --array=0-7 --time=08:00:00 mlp_pipeline/young_tdi_insert.sbatch
```
(drop the entries that exist). Their pressure correction is made in stage 0.
**(b) Direct crystals at n_add = 16 and 22 (independent route; also tests the bookkeeping at the composition where n* lives).** Partners (H2O, pure H, 10 ps) exist at 500/6000 (tdi_long) and, after the end-member extension, 400/5000:
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && PS=10 JOBS="500 6000 16;500 6000 22;400 5000 16;400 5000 22" bash mlp_pipeline/submit.sh "direct crystals n=16,22" --array=0-3%4 mlp_pipeline/young_tdi_dilute.sbatch
```
**(c) Long NPT with a flexible cell (is H2.1O / H2.2O one stable superionic fcc phase over tens of ps?).** The script was never run before this session; on the toy it needed three fixes (docs/CODEBASE.md). 256 O + ~540 H = ~800 atoms is slow (~0.4 s/step -> 30 ps ~ 7 h): look at 108 O first (4 tasks, ~3 h):
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && NREP=3 PS=20 JOBS="2.1 400 4500;2.2 400 4500;2.1 500 5000;2.2 500 5000" bash mlp_pipeline/submit.sh "NPT stability 108 O" --array=0-3 --time=06:00:00 mlp_pipeline/young_stability_npt.sbatch
```
then `grep VERDICT runs/slurm_stabnpt_*.out` and the per-ps table (`less runs/slurm_stabnpt_<job>_<task>.out`): look for the O sublattice label, the systematic shear, O MSD growth and D_H. The full 256-O table: `sbatch mlp_pipeline/young_stability_npt.sbatch` (6 tasks).
**(d) Refit** (login node, as Step 0). Decision rule: if insertion n = 8, 16 fall on the quadratic line within 1.5 sigma, take n* = -mu0/(2 s); if not, take the free-curvature n* of the output and replace the (mu0, s) rows of `docs/data/insertion_regsol_fit.csv` by `report.py`'s `<out>_table_mu0.csv` (renamed columns are accepted) for tx_diagram / huang_hull.

### Step 3. Method and numerics (0.3): collect what is already queued, add only what is missing
```bash
cd /lustre/scratch/mmm0037/H3O && git pull --no-rebase --no-edit origin claude/eloquent-hopper-kfjdei && bash mlp_pipeline/jobs_status.sh && ~/grace-venv/bin/python mlp_pipeline/tdi/collapse.py runs/tdi runs/tdi_long runs/tdi_prep 2>&1 | tail -40
```
* wall test (117562 / 117563): `grep -h "FINAL\|G =" runs/slurm_*117562* runs/slurm_*117563* | tail`, then the pure-H rows of summarize with and without the wall (`H_500GPa_*`), expected difference of G_H within errors (toy: 4099 vs 4084 +- 12 meV). A wall shifts G_H, hence mu_H - G_H and n*, one-to-one: it must not.
* finite size: insertion n = 4 in 32 O and 256 O (`runs/tdi_ins_rep2`, `runs/tdi_ins_rep4`, queued 115860) enter Step 0 automatically at a = 4/32, 4/256; compare with the 108-O line, they are not fitted when flagged.
* quadrature (P3): Kronrod extension of one insertion run, the command in CURRENT_ISSUES section 5 (`RULE=gk NGAUSS=10 NODES=21 ONLYNODES="10 ... 20" JOBS="500 5500 1" sbatch --array=0 mlp_pipeline/young_tdi_insert.sbatch`); `~/grace-venv/bin/python mlp_pipeline/tdi/bar_lambda.py runs/tdi --glob 'insert_*'` is the quadrature-free cross-check.
* pressure correction (P4): `insp0` output lists |P_host - P_target|; keep hosts within ~5 GPa.
* electronic entropy: closed (CURRENT_ISSUES P1c: -29 ... -30 meV per H, flat over 5000-7000 K).

### Steps 4 and 5
Item 4 (ZPE / NQE, He, other ices) and item 5 (phase-diagram side: liquid points at n = 6, 12, melting-line finite size, 300 GPa two-liquid artefact, Uranus / Neptune profiles) change the phase diagram, not the solid-solution solubility itself; do them only after steps 1-2 changed or confirmed (mu0, s).
Cheap pre-check for 4: a PIMD-free estimate is not available here; the Huang static vs ZPE curves differ by 10-20 meV/atom for H3O, i.e. 40-80 meV per extra H, the size of the proxy bias: carry it as a systematic until a quantum-nuclei correction exists.

## 3. Verdict so far (to be updated after Step 1-2)
* Bookkeeping of the method: validated (toy T0 exact, insertion vs Widom/BAR, pure-H insertion check, closure of the fit on its crystal anchors).
* GRACE answer: n* ~ 0.08-0.11 at 400-500 GPa for the quadratic shape, 0.05-0.15 for plausible shapes; direction of every known model error: larger n*.
* NOT validated: the shape of G(n) (Step 2a, cheap), the DFT accuracy of mu0 and s (Step 1b-c), the stability of H2.1-2.2O as one phase (Step 2c), quantum nuclei.

## 4. Toy test of the whole chain (2026-10-29, `bash mlp_pipeline/tdi/toy_solubility.sh 450 5000`, 8 lambda x 1.5 ps, 4 cores, ~20 min)
Launcher -> `summarize.py` -> `report.py` -> `solubility_check.py --solid` ran on 9 toy runs (fitted toy potential, 32-O host). Results: insertion n_add = 2, 4, 8 and direct crystals n_add = 4, 8, 16 give average dG per added H of +34 / +52 / +50 meV (insertion) and -206 +- 141 / -43 +- 73 / +29 +- 37 meV (crystals); at the two compositions measured by both routes the routes differ by 1.8 and 1.2 sigma (short runs: agreement within errors, not a precision test).
The toy has mu0 > 0 (H insoluble), so the fit correctly reports no minimum (n* = nan for the linear fit) and the report's QA correctly excluded the n_add = 16 insertion (host + 16 H sits at 552 GPa vs 450 target). This exercises the mechanics (file formats, caveat / pressure-correction handling, both fits, the bracketing flag); the quantitative checks of the free energies remain the toy T0-T5 tests in CURRENT_ISSUES section 2b.
Do not edit `toy_solubility.sh` while it runs (bash reads scripts incrementally; an edit during this session killed the first attempt).
