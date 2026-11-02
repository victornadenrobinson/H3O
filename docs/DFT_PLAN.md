# DFT labelling plan (insertion / defect / low-P / validation snapshots)

Numbers marked ~ are plan targets or estimates, not measurements. DFT cost: stock VASP, Gamma only, mean 105 s/point (max 181 s) for ~330 atoms on 32 ranks; about 450 points per 4-node hour.
729-atom pure-H points: estimated 5-10x that (~10-15 min each), to be measured with the first one.

## Tree of what feeds the training / validation sets

```
DFT labels
├─ HAVE
│  ├─ training tiers s/m/l + valid_*   (MD runs mined from old OUTCARs; Perlmutter train_l/valid_l = 324-atom H2O..H2.75O, H729 on Young)
│  ├─ lambda=1 frames of the old custom-VASP TDI campaigns (frame_batch.py --count): evaluation only, no insertion pairs
│  └─ 82 stock-VASP single points (Perlmutter timing run: insertion/audit tests)
└─ TO ADD (GRACE or toy geometry -> prepare_points.py -> perlmutter_vasp_check.sh)
   ├─ A  GRACE-MD + insertion      build_insertion_set.py / run_insertion_batch.sh   (running)  H2O host + n=1..27 H, 4 frames/run, train hosts from train_l, valid from valid_l
   ├─ B  raw insertion chains      --equil-ps 0 --chain: exact n -> n+1 pairs from one growing cell
   ├─ C  Young pwidom caches       host/full ensembles, H/O 2.009..2.5, 400/500/600 GPa, 4000-6000 K  (dilute H; GRACE-sampled = reverse test)
   ├─ D  Young EOS / melt_comp     H2.25..H3O at 5000/7000 K, H2.1O, volumes in 300-700 GPa
   ├─ E  low-P EOS states          eos_low_p.py -> 17_eos_md.py at 50/100/200 GPa x {H2O,H2.25O,H2.5O,H2.75O,H3O} x {3000,5000,7000} K
   ├─ F  toy-potential cells       build_toy_cells.py: shake / vacancy / void / insertion / cramp / expand / close contact / hot, 324-432 atoms and 729 H
   └─ G  pure-H 729 GRACE-MD       build_insertion_set.py --comp h --natoms 729 --n-list 0
```

## Table

| id | what | geometry source | atoms | states | DFT points ~ | split |
|----|------|-----------------|-------|--------|--------------|-------|
| A  | H2O + n H, n = 1,2,3,4,6,8,12,16,20,24,27 | GRACE-MD, 4 frames/run (label 2 of 4) | 325-351 | 3000/500, 5000/400, 5000/500, 7000/500, 5000/600 GPa; ~19 train hosts | ~420 train | train |
| A  | same, validation | valid_l hosts: only 5000 K / 500 GPa has any (4 hosts) | 325-351 | 1 | ~90 | valid |
| B  | n -> n+1 chains, raw | build_insertion_set --chain | 324-351 | 5000 K, 500 GPa, 4 chains x n=0..8 | ~40 | valid + train |
| C  | pwidom host/full, dilute H | Young caches, ~4 frames per composition/state | 324-379 | 16 compositions x (500, 600 GPa) x 5000 K (+4000/6000 K at 2.25) | ~150 | 2/3 train, 1/3 valid by cache |
| D  | EOS / melt_comp H-rich | Young eos_scan, melt_comp | 351-432 | H2.25-H3O, 5000/7000 K, v <= 2.4 A^3/atom | ~150 | train |
| E  | low-P states | GRACE NVT at guessed volumes | 324-432 | 5 comps x 3 T x 3 P = 45 states x 4 frames | ~180 | train; 1 of 4 per state valid |
| F  | toy defect cells | build_toy_cells.py | 310-432 | 4 comps x 8 kinds x ~6 | ~190 | train/valid by base |
| F  | toy defect cells, pure H | same, sc 9^3 | 714-756 | 8 kinds x ~4 | ~32 | train |
| G  | pure H, GRACE-MD | 729-atom hosts | 729 | 3 hosts x ~10 frames | ~30 | train |
|    | **total new** | | | | **~1280** (about 3 four-node hours; 729-atom points ~60 of them cost ~5-10x each) | |

Existing: 82 stock-VASP points + the tier frames (counts per tier: `dataset_inventory.py`); lambda=1 frames per campaign: `frame_batch.py --count`.

## Rules
- Whole MD runs / host frames stay on one side of the train / valid split (run_id, base_id, host_source in the frame info).
- Toy and close-contact frames: never in the validation set that decides the model (they are stress tests); keep a separate toy-valid set.
- Label cap: frames with DFT max force > 50 eV/A get the Huber loss or are flagged `stress_test`.
- Validation hosts for 3000/7000 K and 5000 K/400 GPa are missing in valid_l: take them from pool runs outside the train tier (Young melt_comp, tdi_fb) instead.

## Update: free-energy-grade sets (heavy) and gap fill

Free-energy methods (PWidom, TDI) need the DFT-minus-GRACE offset D at exactly their states and at every n, so these sets are labelled densely (all Young GRACE snapshots; select with `young_select.py`, copy the small extxyz / npz subsets to Perlmutter).

| id | what | how | DFT points ~ |
|----|------|-----|--------------|
| B+ | paired n -> n+1 insertion chains (20 chains x n = 0..8) | build_insertion_set --equil-ps 0 --chain | 180 (was 40) |
| C2 | PWidom reverse test: host + 1 H at 2 proxy-chosen sites per cached frame | `tdi/dft_check.py prep-widom <pw_*_frames.npz> --template ... --proxy docs/data/toyfit_t4.json --frames-n 20 --sel 2` (3 DFT per frame), ~7 states: 400/500/600 GPa x 4000/5000/6000 K, H2O, H2.25O, pure H | 420 |
| C3 | TDI-state D statistics: >= 40 independent GRACE frames per end state (pure H, H2O, H2.25O) x 6 states | young_select.py on pwidom host caches / tdi lambda=1 nodes | 240 (some exist: frame_batch --count) |
| D2 | T-gap fill 4000 / 6000 K for H2O+dilute, H2.25O, H2.5O, H2.75O, H3O (training data: H2.5O / H2.75O only at 3000 K, nothing at 4000 / 6000) | young_select.py on melt_comp 300-600 GPa {free, hold} and eos_scan | ~120 |

New grand total ~2,200 points (~5 four-node hours, plus the 729-atom points). Priority order if capacity is short: C2 and B+ (they decide whether the mixing free energy can be trusted), then A, C, E, D, D2, F, G.
Known issue: eos_scan H3O 5000 K starts at ~460 GPa (all four lowest-volume runs repeat one volume): its low-P guess is an extrapolation over a decade of pressure, so pass 2 of eos_low_p.py matters most there.

## Update 2: A and B+ widened to a grid
A: T = 3000, 4000, 5000, 6000, 7000 K x P = 300, 400, 500, 600 GPa (+-50 GPa windows), H2O host + n = 1,2,4,8,16,27 H (full n list at 5000 K / 500 GPa), dv per H = 1.26 (500/P)^0.75 A^3, 40 jobs (20 states x train/valid). ~770 points.
B+: chains at 4000 / 5000 / 6000 K x 300-600 GPa, 2 chains per state, ~210 points.
Hosts: tiers (train_l / valid_l) where they exist; elsewhere pick host frames from the Young pool with `young_select.py --as-hosts` and pass HOSTS_TRAIN / HOSTS_VALID to run_insertion_batch.sh (valid hosts must come from runs that are not in train_l).
Below 300 GPa (50-200): E only (GRACE NVT at guessed volumes); insertions there come after E has produced hosts (build_insertion_set --hosts on the E trajectories).
New total ~2,540 points (~5.6 four-node hours + 729-atom points).

## Update 3: H  Huang H3O supercells (+ H2.75O / H2.5O / H2.25O channel variants)
`build_h3o_supercells.py` from the static relaxations (runs/h3o_static/P<P>, H3O at 300-800 GPa, variants at 300-800): supercell of 360-392 atoms with the best-conditioned cell (min width ~7.5 A, not a skinny 3x2x1), frames: static, rattle x3, strain x4, defects x8, GRACE-MD at 1000/2000/3000/4000 K x3.
27 relaxed structures x ~28 frames = ~750 candidates; label ~8 per structure = ~220 points. One pressure (--valid-P 450) goes to the validation side whole. Plan total ~2,760 points.
