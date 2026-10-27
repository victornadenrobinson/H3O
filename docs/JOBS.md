# Jobs on Young: what each batch script is, and what has been submitted

Purpose: one place where a new chat (or the user) can see (1) what each Slurm script really does and (2) what has been submitted. The repository cannot see Young: the running state comes from `bash mlp_pipeline/jobs_status.sh` (paste its output to the chat; the chat then refreshes the "Submission log" below) and from `runs/JOBS_LOG.tsv` on Lustre, which `mlp_pipeline/submit.sh` appends to on every submission.
**Submit with `bash mlp_pipeline/submit.sh "note" [sbatch args] mlp_pipeline/<script>.sbatch`** (env knobs such as `PS=10 JOBS="..."` in front as usual): it runs `sbatch --parsable`, and records date, job id, script, sbatch arguments, the run knobs that were set, git commit and the note.
Conventions for every TDI script: GRACE `ft_m2`, dt 0.5 fs, energies in meV, array task = one GPU, restartable by resubmitting the same command (a changed dt / lambda grid / cell is refused), results under `runs/<dir>`, table by `~/grace-venv/bin/python mlp_pipeline/tdi/summarize.py runs/tdi runs/tdi_long runs/tdi_prep ...`, figures by `tdi/report.py`. All cost numbers below are ESTIMATES from the script headers unless stated.

## Registry of batch scripts (job name = what `squeue` shows)

| job name | script | what it does | array / time (default) | output dir | notes |
|---|---|---|---|---|---|
| tdi | young_tdi.sbatch | first TDI: bcc (task 0) and fcc (1) H2O at 400 GPa, 3000 K | 0-1 / 8 h | runs/tdi | old, dt 1.0 results exist |
| tdiscan | young_tdi_scan.sbatch | job table (phase P T H_per_O): fcc/bcc H2O scan at 200-500 GPa, 3000/5000 K | 0-5 / 6 h | runs/tdi | PS default now 8 |
| tdihull | young_tdi_hull.sbatch | liquid hull at 400 GPa, 7000/9000 K (H2O ... H5O, pure H) | 0-9 / 6 h | runs/tdi | |
| tdiliq | young_tdi_liquid.sbatch | liquid hull on a (P,T) grid, 7 compositions (2, 2.25, 2.5, 3, 4, 5, pure H 250 atoms); hot melt step (HOT=0.5 ps at 12000 K) | 0-83%8 / 12 h | runs/tdi | PLIST/TLIST override; PS default 8; `--array=6,13` with PLIST="400" TLIST="5500 6000" = the pure-H tasks only |
| tdimelt | young_tdi_melt.sbatch | liquid vs fcc vs bcc H2O TDI at 300/400/500 GPa x 5000-6500 K for DeltaG_melt(T) | 0-35%6 / 14 h | runs/tdi | needs NODES=17 for Kronrod |
| tdiprep | young_tdi_prep.sbatch | H2.25O / H2.5O with the start protocol (O held, H thermalises, O released) plus H2O and pure-H partners, 5 ps, 17 lambda (GK); P 300-600 x T 4000/5000/6000 | 0-47%8 / 8 h | runs/tdi_prep | task = (P,T,composition), composition fastest (2.25, 2.5, 2, H); 300 GPa >= 5000 K protocol runs melted: their rows are invalid |
| tdilong | young_tdi_long.sbatch | 10 ps per lambda of 7 system types (fcc H2O, fcc H2.25O, pure H, bcc H2O, 2 liquids, insertion n=1) | 0-6 / 5 h | runs/tdi_long | the time-convergence reference; `tdi/convergence.py runs/tdi_long` |
| tdiins | young_tdi_insert.sbatch | ghost-proton insertion into fcc H2O: table "P T n [NREP]" (mu_H per added H, fixed volume, pressure correction in stage 0) | 0-10 / 12 h | runs/tdi (runs/tdi_ins_rep<R> for other host sizes) | PS default 8; JOBS="500 6000 4;..." |
| insp0 | young_tdi_insert_p0.sbatch | adds the stage-0 pressure correction to EXISTING insertion runs (no lambda sampling redone); skips tags active in the last 45 min | 0-3 / 4 h | runs/tdi, runs/tdi_long | ONLYTAG=<tag> tests one; interactive: `SLURM_SUBMIT_DIR=$PWD SLURM_ARRAY_TASK_ID=0 ONLYTAG=... bash ...` |
| tdiext | young_tdi_extend.sbatch | extend runs listed in mlp_pipeline/extend_list.txt (EXT / NEW lines) | 0-12%6 / 8 h | runs/tdi | OLD extend.py bug (before 9ed6ec1) started NEW `_rep` runs: check `ls runs/tdi/*_rep[0-9]_stage0.json` |
| tdiend | young_tdi_endmembers.sbatch | extend all END MEMBERS (fcc/bcc H2O, liquid H2O, pure H at dt 0.5) to PS ps per lambda (default 8) | 0-59%6 (N = number of lines printed by tdi/endmember_list.py) / 8 h | runs/tdi | skips stale-json runs |
| tdifb | young_tdi_fccbcc.sbatch | extensive fcc vs bcc H2O: P 100-600 step 50 x T 2000.. step 500 up to guessed TMAX(P); 95 points, fcc then bcc per task, 10 lambda, PS 8 | 0-94%8 / 12 h | runs/tdi_fb | volumes: DFT EOS, else the other phase, else scaled; check coverage with `tdi/fcc_bcc_grid.py coverage` |
| tdidil | young_tdi_dilute.sbatch | DIRECT crystal route at H2.02/2.05/2.10O (n = 2, 5, 11) at 500/6000, 500/5500, 400/5000 for comparison with insertion; 10 ps, 9 lambda | 0-6%4 / 8 h | runs/tdi_dil | partners at 500/5500 and 400/5000 must be extended first (tdiend) |
| tdifs | young_tdi_finite_size.sbatch | finite size: pure H 250/500/750 atoms (500/6000, 400/5000), fcc H2O 32/108/256 O, liquid H2O 32/108 O | 0-10%6 / 14 h | runs/tdi_fs | 750-atom H and 256-O tasks may need resubmission or more memory |
| mixscan | young_mix_scan.sbatch | quick mixing scan at one (P,T): fcc H2O, fcc H2.25O, pure H | 0-2 / 2 h | runs/tdi, runs/mix | wrote the 17-lambda Kronrod pure-H runs at 500 GPa (the stale-json incident) |
| meltscan | young_melt_scan.sbatch | free NVT stability of bcc and fcc H2O vs T per pressure | 0-5 / 2.5 h | runs/melt | `21_melt_summary.py` |
| meltcomp | young_melt_comp.sbatch | free NVT melting of fcc H-rich crystals, O held then released (HOLD, SEED); table HLIST x PLIST, all TLIST at once (max 9 runs per GPU) | 0-20%6 / 6 h | runs/melt_comp | `tdi/melt_table.py`; env HLIST PLIST TLIST PS HOLD SEED |
| eosscan | young_eos_scan.sbatch | MLP EOS at DFT-EOS volumes (100-1200 GPa), 5 compositions solid/liquid + pure H | 0-32 / 2.5 h | runs/eos_scan | `22_eos_summary.py` |
| eosmap | young_eos_map.sbatch | BIG MLP EOS map on a V/atom grid (about 20-1200 GPa): compositions 2, 2.02, 2.05, 2.1, 2.25, 2.5, 3, solid 2000-7000 K, liquid 5000-9000 K, pure H 3000-9000 K | 0-76%8 / 4 h | runs/eos_map | for MLP-native volumes, bulk moduli, partial molar volume of H |
| stab | young_stability.sbatch | 20 ps NVT stability of bcc/fcc at 400/500 GPa, 3000 K | 0-3 / 6 h | | old |
| grace_sweep | young_ft_sweep.sbatch | GRACE fine-tune sweep (ft_m4/ft_m3/ft_l3/ft_m5) | 0-1 / 6 h | runs/grace | |
| mace_ft_h3o / smoke | 05_finetune_young.sbatch / smoke_test.sbatch | MACE fine-tune / one-snapshot smoke test | | | |

Analysis tools: `tdi/summarize.py` (tables), `tdi/report.py` (figures, also runs `plot_all.py`), `tdi/convergence.py` (G(t) table), `tdi/liquid_check.py` (did the liquids stay liquid: O diffusion), `tdi/plot_all.py` (G vs time svg), `tdi/melt_table.py`, `tdi/fcc_bcc_grid.py`, `tdi/endmember_list.py`, `22_eos_summary.py`, `21_melt_summary.py`.

## Submission log (maintained by the chats; refresh from `bash mlp_pipeline/jobs_status.sh` output)

Status words: DONE (results seen), RUNNING, PENDING, UNKNOWN (never reported back), PROPOSED (the chat suggested a command; the user may not have submitted it).

| date (2026) | job id | script / what | status (last seen) |
|---|---|---|---|
| 10-26 | 115302 | tdiscan | DONE (task 2 failed) |
| 10-26 | 115312 | tdihull (400 GPa 7000/9000 K) | DONE |
| 10-26 | 115317 | meltscan | tasks 0-3 DONE, 4-5 were running |
| 10-26 | 115387 | tdiins n = 1,2,4 at four (P,T) | tasks 0-2 running, rest pending; later results in tables |
| 10-26 | 115445, 115456, 115457 | mixscan 500 GPa 5000/6000/7000 K | results in runs/tdi (500 GPa H references were corrupted afterwards by tdiliq: stale json) |
| 10-26 | 115446, 115451 | tdihull (500 GPa liquid hull) | results seen (500 GPa liquid rows) |
| 10-26 | 115452 | eosscan (33 tasks) | UNKNOWN (results not seen) |
| 10-26 | 115463 | tdiins (second table, 18 tasks) | UNKNOWN |
| 10-26 | 115466, 115467 | meltcomp full 21-task scans | UNKNOWN |
| 10-26 | 115468 | tdiext (13 tasks) | UNKNOWN; old extend.py bug: check `*_rep[0-9]_stage0.json` |
| 10-26 | 115470-115475 | meltcomp hold-matrix (HOLD 0/1/3 x seeds 0/1, H2.25O 400/500 GPa) | UNKNOWN |
| 10-26 | 115476 | tdiliq 84 tasks (300-600 GPa x 4000-6000 K) | DONE for 5000/6000 K rows; no 4000 K liquid rows exist (hull EOS covers 5000-9000 K only: those tasks were refused) |
| 10-26 | 115478 | tdilong 7 tasks | DONE (task 6 insertion n=1 reached 6.5 ps at last look) |
| 10-26/27 | 115480 | tdiprep 48 tasks | tasks 0-7 DONE; 8-14 running at 11:00 on 10-27; tasks 8, 9 (300 GPa 6000 K H2.25O/H2.5O) have molten starts: scancel recommended |
| 10-27 | (interactive 115455) | bcc 400 GPa 3000 K dt 0.5 | DONE: 2003.7 +- 1.8 |
| 10-27 | (interactive) | insp0 test on insert_400GPa_5000K_n2 | DONE: correction -40.6 +- 9.1 meV per H |
| 10-27 | ? | tdiins JOBS="500 6000 4;500 6000 11;500 6000 27" PS=10 NODES=12 | SUBMITTED by the user (job id not reported) |
| 10-27 | - | tdiend PS=12; tdiliq PLIST=400 TLIST="5500 6000" --array=6,13 (pure H); tdiliq PLIST="400 500" TLIST="5500 6500" PS=4; tdiins JOBS with n = 2, 8, 16, 54 and NREP 2/4 at 500/6000; tdiins T/P table (n = 4, 12); insp0 full set; tdifs; tdifb; tdidil; eosmap; meltcomp 9-composition tables | PROPOSED (status UNKNOWN: ask the user / read JOBS_LOG.tsv) |

## Things to check first in a new chat
1. `bash mlp_pipeline/jobs_status.sh` (what is running / finished / failed, GPU hours per job name).
2. `git log --oneline | head` (other chats push to other branches too: `git branch -r`; the Opus review fixes live on claude/eloquent-hopper-kfjdei and are merged here).
3. HANDOFF.md "Session 4" blocks for known data hazards (stale stage-0 json, uncorrected insertion rows, molten protocol starts).
