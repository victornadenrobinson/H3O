# Inline numerical evidence not contained in uploaded logs

## Training-label examples (verbatim scientific fields)
train_m first: 384 atoms; cell diagonal 9.1 A; source=tdi/H3O/5000K-liquid/bcc-128mol-liquid-start/9.1/out10/OUTCAR.gz; frame=389; frame_rank=0; energy_sigma0=-541.15184903 eV; free_energy=energy=-542.98418743 eV; config_type=h2o_5000K; T_K=5000; P_GPa=636.134500018345; box_A=9.1; run_rank=2; phase_tag=all.
Stress (full tensor eV/A^3): [-3.660281272280565,0.010321677264110032,0.02835014530253218,0.010321677264110032,-3.8289851944892397,-0.016717370389929986,0.02835014530253218,-0.016717370389929986,-3.7685090155573437].
valid_m first: 512 atoms; cell diagonal 10.1 A; source=tdi/H3O/9000K/h3o-128mol-9000K/10.1/out30/OUTCAR.gz; frame=92; frame_rank=0; energy_sigma0=-717.71411316 eV; free_energy=energy=-748.31581584 eV; config_type=h3o_9000K; T_K=9000; P_GPa=506.9534075741652; box_A=10.1; run_rank=1; phase_tag=all.
Stress: [-2.6784882417377993,0.01579900097865665,-0.029325518416652208,0.01579900097865665,-2.806811840603785,-0.010082196800459017,-0.029325518416652208,-0.010082196800459017,-2.7760385230057656].

## VASP template and generated static settings
Template copied from ../tdi/H3O/7000K/bcc-128mol-7000K/backup/ to vasp_inputs/.
VASP_MODULE=vasp/5.4.4-cpu, executable /global/common/software/nersc9/vasp/vasp/5.4.4-cpu/bin/vasp_std (or vasp_gam for Gamma).
Template tags: SYSTEM=H3NO4-P212121; NWRITE=1; ISTART=0; ICHARG=2; LWAVE=FALSE; LCHARG=FALSE; NBLOCK=1; KBLOCK=1; NCORE=16; NSIM=1; LPLANE=TRUE; ENCUT=1100 eV; EDIFF=1e-6 eV; IALGO=48; LREAL=TRUE; NELMIN=6; NELM=80; ISMEAR=-1; IBRION=0; ISYM=0; NSW=5000; ISIF=2; SMASS=0; POTIM=0.5 fs; TEBEG=7000 K; SIGMA=0.60321312 eV.
Commented alternatives are not active: NPAR=8, BMIX=2.0, MAXMIX=50, PREC=High, WC=1000, NBANDS=360, EDIFFG=-1e-3, PSTRESS=3000 kbar.
KPOINTS: automatic Gamma, 1 1 1, shift 0 0 0.
Generated first 3000-K static INCAR: SYSTEM=H3NO4-P212121; NCORE=16; NSIM=1; LPLANE=TRUE; ENCUT=1100; EDIFF=1e-6; IALGO=48; LREAL=TRUE; NELMIN=6; NELM=80; ISMEAR=-1; IBRION=-1; NSW=0; ISYM=0; LWAVE=FALSE; LCHARG=FALSE; SIGMA=0.258520. ISIF removed in that first writer (later explicitly set 2).
User's sigma table, K:eV: 20000:1.72346606,10000:.86173303,9500:.81864638,9000:.77555973,8500:.73247308,8000:.68938643,7500:.64629978,7000:.60321312,6500:.56012647,6000:.51703982,5500:.47395317,5000:.43086652,4500:.38777987,4000:.34469321,3500:.30160656,3000:.25851991,2500:.21543326,2000:.17234661,1500:.12925996,1000:.086173303.
Available modules reported: vasp-tpc/5.4.4-cpu (default), vasp-tpc/6.4.2-cpu, vasp-tpc/6.4.2-gpu, vasp/5.4.4-cpu (default), vasp/6.4.2-cpu,gpu; vasp/6.4.3-cpu,gpu; vasp/6.6.1-cpu,gpu.
Historical native VASP LOOP+ cpu/real seconds: 18.1578/18.6506,9.1340/9.3091,7.8391/7.9944,7.4318/7.5604,7.2683/7.4004,7.2280/7.3703,7.1883/7.3226,7.2013/7.3329,7.2093/7.3407. Core count of these historical MD timings not established by that paste.

## First VASP point and paired queue retries
Single cached H2O 300 GPa/3000 K snapshot: 324 atoms, TOTEN=-965.297829640 eV, 87.1 s, 32 ranks, one attempted/completed, zero incomplete.
Saved 40-structure queue: four concurrent x32 ranks on 128-task node; 20 pairs. First pass 38/40 completed; p018_without_H and p019_without_H exit=-9 at cap, not accepted.
Retry: p019_without_H TOTEN=-934.770057750 eV,102.3 s; p018_without_H TOTEN=-949.519682960 eV,103.1 s. 40 selected points complete,2 attempted on retry,0 failed. No failed TOTEN used.
Saved candidate example: request_6220f9e6b0c3c45ee451e75f.json: K=2000,P=300 GPa,T=5000 K,dU_mlp=2.408374071121216 eV,direction=deletion,frame=31,gap_kT=-9.752482026158667,log_weight=6.8266885427661705,model_fp=d810867382d55724,phi=6.610393458674928 eV,reason=dominant_estimator_weight,trial=3,tag=pw_H2p083O_300GPa_5000K. This was a queue-layout example; the later 40-point batch was explicitly filtered to the 300 GPa/3000 K reference run.

## Early MD cancellation / diagnostics
Allocation 60628367,600 GPa/3000 K,30-minute requested wall cap; displayed elapsed05:01/wallcapleft24:59: H2O40% [02:12/03:18 ETA],H2p083O36%[02:10/03:51],H2p25O30%[01:59/04:35],pureH27%[01:40/04:31]. GPU0/1 2.5/40GiB, GPU2/3 4.5/40GiB,util24/30/33/41%.
Slurm revoked allocation for time limit. Last dashboard elapsed05:31/wallcapleft24:29: H2O50%[02:45/02:46],H2p083O45%[02:43/03:19],H2p25O39%[02:35/04:01],pureH36%[02:14/03:58],GPUutil43/30/33/42%. Dashboard wall-cap estimate was not the allocation's actual remaining time.
User subsequently requested salloc -N1 -Cgpu -qinteractive -t04:00:00 --gpus1 -A m3349_g.
Time plots requested at 0.1 ps increments. Screenshot cumulative production range0.1–1.0 ps; upward drift observed; no digitized point arrays supplied. First-hull plot H-fraction zoom approximately .667–.692; full H endpoint x=1.0; no additional exact values recoverable from image alone.

## Resource and software milestones
Original added files: build_eos_2000.py 373 lines,plot_pwidom_diagnostics.py224,pwidom_auto.sh361; 3 files/958 insertions.
User first pull e90715a:19 files,780 insertions,23 deletions; next checkpoint branch pull26c3715 initially blocked by the same three untracked files.
Python path /pscratch/sd/v/victornr/grace-env/bin/python; first VASP CPU allocation60643090. Subsequent kmesh allocation60643730. CPU interactive requests -N1 -Ccpu -qinteractive -t01:00:00 -n128 -A m3349; mistaken two-node request -N2 -n128 still totals128 tasks, not256. Correct two-node allocation256 tasks,128/node.
Model training recalled by user: approximately700 snapshots in medium set;1–2 hours fine-tuning. These are recollections, not a recovered training log.
Checkpoint interval300 wall seconds, dt0.5 fs; original user discussion1 vs0.5 fs, possible0.25 fs; no quantitative matched timestep free-energy results supplied. Earlier eq0.2/0.4 ps questioned; default friction.01 inverse ASE time corresponds approximately1.018 ps damping. This is distinct from run_real.py's explicit tau conversion.
Original template benchmark first GPU dashboard40GiB/device; prior repository benchmark80GiB/device, not identical machines. Full GPU resource numbers remain in linked historical benchmark CSVs.
Pure-H size criterion recalled by user: roughly700 atoms for Gamma. First DFT plan erroneously inherited250; new builder-compatible size729 (=9^3),full730,at unchanged V/atom. 686 (=2x7^3) would also be roughly700 but was not selected. A 750-atom cell is not accepted by the current n^3/2n^3/4n^3 builder.
Original backend INCAR observed LWAVE.FALSE at line16,ISTART0 line20,LWAVE.TRUE line22; WAVECAR0 bytes at Nov1 08:16; checkout290c3ca. New unique-key fix6f532a7. Four-worker plan55fe2de; parser bugfixed50f6413.
Initial inline cold force calls: H2O324 atoms call1=72.9s E=-507.91550002;call2=73.5s E=-519.42372927;call3=68.2s E=-541.61172732. PureH250 call1=19.2s E=-239.28472402;call2=22.5s E=-238.45629365. All warm=False. Full subsequent cold logs archived separately.
First later pureH tail: calls31–49 and full-MD checkpoint19/24,3/5 frames; all251 atoms,warmFalse. Same values are retained in smoke_cold_tail_all_workers.txt.
VASP single-point wrappers defaultNCORE16,OMP1; original four32-rank packing,then8x32(two replicas of four hosts),then4x64(one replica). Requested20-minute test stop at18min,120sgrace,hardcap20min. Smoke2fs eq+10fs sampling perensemble,dt.5fs =>24steps each,5frames every2fs,sel=ndel=1 =>10trials. Production1ps eq+.5ps sampling perensemble =>3000steps each,6000total;20fs frame interval25frames/ensemble,sel=ndel=4 =>200trial differences.
48h batch allowance,47h runner budget;2nodes256ranksfourworkers64each; jobs60649379/60649380/60649381 correspond500/400/600GPa,5000K. Submission only confirmed. Three2-node48h jobs reserve up to288node-hours (96each); earlier24h equivalents48node-hours/job,144total.

## Explanatory estimates, NOT measurements
At3000K,kBT~.259eV. 1.4eV~5.4kBT,exponential factor~225; .1eV factor~1.47;1eV factor~48. These are individual Boltzmann-weight sensitivities, not chemical-potential corrections. 1.4eV/400atoms=3.5meV/atom is a numerical normalization only; the measured pair error remains per added H.
20meV/atom x324atoms=6.48eV whole-cell scale, NOT an independent bound on paired-energy error. Example total errors+5 and+6.4eV produce1.4eV pair error; illustrative,not data.
Working goals discussed:20–50meV/H systematic transfer/addition bias,50–100meV/H paired scatter well belowkBT; these are provisional targets,not universal sufficient accuracy conditions.
Ideal throughput estimate:8concurrent points at110s/point,24h =>~6300points/~3100pairs; ignores MD,failures,SCF variation,correlations and trial overlap. Not a measured production result.
For6000 MD steps:9/20/60/90s perstep =>15/33.3/100/150h,beforetrial calculations. Earlier optimistic9s extrapolation from nativeVASP does not apply automatically to ASEsinglepoints.
NativeMD thought experiment24h,node128cores,dt.5fs:9–18s/step =>2.4–4.8ps forone lambda;ten sequentiallambda =>.24–.48ps each;ten nodesonewindoweach=>2.4–4.8ps each,all includingequilibration. Corecountassumedinestimate,not verifiedforhistoricaltimingpaste.

## Last ordinary-snapshot preparation predictions supplied inline
normal_037_pureH frame44,T3000K,E_GRACE=-369.874787807eV;
normal_038_pureH frame46,T3000K,E_GRACE=-369.609475076eV;
normal_039_pureH frame49,T3000K,E_GRACE=-369.133857131eV.
Preparation reported40 ordinary host snapshots,no MD or VASP executed at that stage.
