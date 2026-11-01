# GRACE / PWidom / DFT audit: results and concerns, 2026-11-01

**Status:** user-reported Perlmutter measurements and local code audits from this
chat. This is a record of evidence, not validation of equilibrium solubility.
No production VASP executable or remote cluster filesystem is available in this
cloud checkout. Job submission is confirmed where IDs were pasted; the later three-job tails
also confirm production force calls and checkpoints at all three states.

## Main concern and what is established

The scientific concern is H uptake into superionic ice: dilute H chemical
potentials, composition-dependent transfer costs, and shallow formation-free-energy
hull minima. **H2.25O is not a dilute-defect limit**, and sizeable errors appeared
on its *ordinary* GRACE MD configurations, not only on perturbed Widom probes.

1. Seven supplied original-label VASP reproduction rows agree very closely in
   energy, forces and stress. This supports the reference-energy convention and
   DFT setup for those samples; it is not evidence that every original label was
   checked. The remaining 33 numerical VASP reproduction rows were not supplied.
2. All 20 saved paired GRACE addition energies reproduce in independent single
   evaluations (maximum discrepancy below 0.07 meV/H). Large paired VASP–GRACE
   discrepancies therefore survive the cached/batched-energy sanity check.
3. Ordinary 300-GPa/3000-K H-rich GRACE snapshots have large raw energy, force and
   stress discrepancies versus VASP. A constant elemental energy shift alone
   cannot explain the force differences or configuration-dependent energy scatter.
4. The historical ~20-meV/atom benchmark is reproducible as **21.684725 meV/atom
   MAE after fitted per-element shifts**, not raw energy RMSE. These metrics must
   not be compared as if identical. Force RMSE is 0.717290 eV/Angstrom.
5. The small held-out breakdown suggests worse low-T energy accuracy in this
   selection; it does not isolate temperature from composition/pressure/phase.
   Liquid-data dominance as the cause is a hypothesis, not a demonstrated result.
6. VASP-driven MD, checkpoint resume, larger H sizes and warm starts are observed.
   A completed physical VASP PWidom/BAR result has **not** been supplied yet.

The measured selected pair errors are NOT additive corrections to the chemical
potential. Host and reservoir ensembles, estimator weights, reference convention,
finite size, and overlap all matter. The current evidence neither proves nor
rules out favourable DFT H uptake at the most promising 500–600 GPa states.
Same-GRACE agreement among TDI/insertion/PWidom demonstrates internal consistency,
not independent DFT accuracy.

User reports bcc–fcc agreement with literature, plausible EOS and plausible TDI
melting temperatures. Those can survive stronger cancellation for same-composition
comparisons; none independently establishes solubility accuracy. Their new
quantitative validation outputs were not pasted in this portion of the chat.
Earlier numerical TDI/EOS results are linked below rather than reinvented.

## Scope and preservation of all supplied numbers

The tables below cover every supplied numerical result in the current validation
campaign. Complete uploaded logs, inline numerical transcriptions, preliminary
partial analyses, earlier 44-result PWidom export, and full-precision collector
CSVs are retained in [the evidence directory](grace_dft_audit_2026-11-01/).
Repetitions are kept in source logs. Module/progress timestamps and cold force
call histories are preserved there rather than omitted from the archive.
Screenshots showed time-dependent curves but did not supply their underlying
arrays: exact curve values cannot be recovered from the chat images without
inventing precision. The earlier result tables preserve the available numerical
endpoints. No claimed-complete inventory of unseen remote files is implied.

## Units, signs and selection

- Absolute energies and paired differences: eV; chemical potentials: meV/H;
  formation energies: meV/atom; force component RMSE: eV/Angstrom.
- DFT label is force-consistent electronic free energy **TOTEN**, not energy
  extrapolated to zero smearing. ISMEAR=-1; SIGMA=kB*T of the target state.
- Both insertion- and deletion-selected pairs use dU=E(with H)-E(without H).
  Normal/replay logs print DFT−GRACE; baseline/benchmark energy errors print
  GRACE−DFT. Reversing the log sign does not imply a scientific disagreement.
- Static stress uses ASE tensile convention, eV/Angstrom^3 or converted GPa,
  without ionic kinetic stress. Do not double-count NkBT/V in MD pressure.
- A neutral H addition adds one nucleus and one electron. These are not explicitly
  charged OH-minus calculations. Local OH-like/H-rich/short-contact environments
  can appear, but no charge or bond analysis was supplied to identify them.
- Original saved queue: per direction, up to two largest estimator weights, two
  largest absolute centered (dU_GRACE−phi_proxy)/kBT residuals, and one middle-list
  trial; overlaps deduplicated, default limit 10/host. “Representative” is a list
  midpoint, not random ensemble sampling. Proxy disagreement is not known DFT error.
- VASP queue import: exact 300-GPa/3000-K reference run, matching model/state,
  geometry hashes and one-H differences, round-robin across host/direction groups;
  within each group request-filename order. 20 pairs/40 structures. This is a
  targeted diagnostic selection, unsuitable for unbiased free-energy correction.
- Normal set: 10 evenly spaced host-only frames/composition after discarding the
  first half of cached production frames. No inserted/deleted atoms. Discarding
  frames does not establish equilibration or independent samples.

## Original label convention and VASP reproducibility

First train label: 384-atom H2O, 5000 K, 636.134500018345 GPa, 9.1-A cubic cell;
TOTEN=-542.98418743 eV; sigma-zero energy=-541.15184903 eV (difference 1.83233840 eV).
First validation label: 512-atom H3O, 9000 K, 506.9534075741652 GPa, 10.1-A cube;
TOTEN=-748.31581584 eV; sigma-zero=-717.71411316 eV (difference 30.60170268 eV).
Original source/frame/stress metadata and the complete numerical INCAR/smearing
settings are in [inline settings](grace_dft_audit_2026-11-01/raw/inline_settings_and_milestones.md).

The 7000-K template SIGMA=.60321312 eV was correctly replaced by .258520 eV for
3000-K single points; 5000 K gives .430867 eV, 9000 K .775560 eV. ENCUT=1100 eV,
EDIFF=1e-6, IALGO=48, LREAL=TRUE, NELMIN=6, NELM=80, NCORE=16, NSIM=1, ISYM=0.
KPOINTS was Gamma-centered 1x1x1, zero shift. Static IBRION=-1, NSW=0. The first
static writer removed ISIF; later writers explicitly request ISIF=2. Intended
species/pseudopotentials still require matching the original per-source inputs;
no POTCAR is committed here.

First 324-atom host test: TOTEN=-965.297829640 eV, 87.1 s on 32 ranks, exit accepted.
Original-label reproduction rows supplied (printed per-atom rounding retained):

| id | TOTEN_eV | DFT_minus_original_eV | printed_meV_atom | force_RMSE_eV_A | stress_RMSE_GPa | wall_s |
|---|---|---|---|---|---|---|
| 004 | -655.653610590 | +0.000023 | +0.000 | 0.000177 | 0.0005 | 138.8 |
| 000 | -542.983598390 | +0.000589 | +0.002 | 0.000245 | 0.0005 | 145.5 |
| 006 | -908.739697180 | +0.000033 | +0.000 | 0.000132 | 0.0006 | 149.3 |
| 007 | -415.378372280 | -0.000149 | -0.000 | 0.000147 | 0.0008 | 163.3 |
| 005 | -727.446427520 | +0.000069 | +0.000 | 0.000259 | 0.0008 | 167.9 |
| 001 | -711.800247480 | -0.000005 | -0.000 | 0.000140 | 0.0002 | 171.2 |
| 003 | -63.942604260 | +0.000638 | +0.001 | 0.000124 | 0.0002 | 172.5 |

## Ordinary GRACE-MD configurations vs VASP: all 40, 300 GPa / 3000 K

All 40 VASP structures completed, zero failed/incomplete. Ten configurations per
host. H2O=324 atoms; H2.083O=333; H2.25O=351; pure H=250. The reservoir in THIS
validation set remains the small Gamma cell: subsequent 729-atom changes do not
retroactively make it k-point converged.

Derived summary from the printed values (full precision in CSV; rounding below):

| host | n | mean_error_eV | sample_std_error_eV | energy_bias_meV_atom | energy_RMSE_meV_atom | force_RMSE_eV_A | stress_RMSE_GPa |
|---|---|---|---|---|---|---|---|
| H2O | 10 | 7.634995 | 0.730063 | 23.5649 | 23.661653 | 0.406826 | 0.870357 |
| H2p083O | 10 | 26.302909 | 3.597493 | 78.9879 | 79.650045 | 0.652012 | 3.676795 |
| H2p25O | 10 | 48.87858 | 7.86349 | 139.2553 | 140.867831 | 1.163267 | 8.471202 |
| pureH | 10 | 21.181985 | 1.328046 | 84.728 | 84.877748 | 0.49318 | 6.37304 |

Energy bias and RMSE here are raw, not element-shifted. Sample standard deviation
of whole-cell DFT−GRACE errors is shown to separate bias from fluctuation. Force
and stress summaries are RMS of equal-size per-frame component RMSE values;
no independent-sample uncertainty or relative force error is inferred.

Every supplied ordinary-snapshot row:

| index | host | DFT_minus_GRACE_eV | DFT_minus_GRACE_meV_atom | force_component_RMSE_eV_A | stress_component_RMSE_GPa |
|---|---|---|---|---|---|
| 0 | H2O | 6.871009 | 21.207 | 0.394191 | 0.7453 |
| 1 | H2O | 7.706638 | 23.786 | 0.391179 | 1.0037 |
| 2 | H2O | 7.271278 | 22.442 | 0.389493 | 0.7615 |
| 3 | H2O | 8.435288 | 26.035 | 0.451561 | 1.208 |
| 4 | H2O | 7.694969 | 23.75 | 0.421547 | 0.6938 |
| 5 | H2O | 8.760566 | 27.039 | 0.449355 | 1.1242 |
| 6 | H2O | 7.386466 | 22.798 | 0.368883 | 0.9811 |
| 7 | H2O | 8.224395 | 25.384 | 0.383794 | 0.6825 |
| 8 | H2O | 7.688295 | 23.729 | 0.428429 | 0.6559 |
| 9 | H2O | 6.311043 | 19.479 | 0.380142 | 0.6078 |
| 10 | H2p083O | 26.683929 | 80.132 | 0.637813 | 4.084 |
| 11 | H2p083O | 29.222923 | 87.757 | 0.672994 | 3.289 |
| 12 | H2p083O | 19.45947 | 58.437 | 0.695055 | 3.1958 |
| 13 | H2p083O | 28.611265 | 85.92 | 0.581896 | 3.4687 |
| 14 | H2p083O | 28.750458 | 86.338 | 0.675586 | 3.9202 |
| 15 | H2p083O | 21.81067 | 65.498 | 0.64267 | 3.0454 |
| 16 | H2p083O | 26.424188 | 79.352 | 0.643972 | 3.6104 |
| 17 | H2p083O | 24.173555 | 72.593 | 0.709452 | 4.409 |
| 18 | H2p083O | 31.286192 | 93.953 | 0.69193 | 4.8371 |
| 19 | H2p083O | 26.606443 | 79.899 | 0.55094 | 2.2209 |
| 20 | H2p25O | 44.193991 | 125.909 | 1.073787 | 6.875 |
| 21 | H2p25O | 46.46536 | 132.38 | 1.085321 | 6.9053 |
| 22 | H2p25O | 52.039949 | 148.262 | 1.203958 | 7.9509 |
| 23 | H2p25O | 49.190105 | 140.143 | 1.103885 | 7.4624 |
| 24 | H2p25O | 38.248196 | 108.969 | 1.157356 | 7.0298 |
| 25 | H2p25O | 35.507725 | 101.162 | 1.12239 | 6.2599 |
| 26 | H2p25O | 59.263461 | 168.842 | 1.228563 | 9.9378 |
| 27 | H2p25O | 57.880404 | 164.901 | 1.212016 | 9.6876 |
| 28 | H2p25O | 54.419981 | 155.043 | 1.193663 | 9.3902 |
| 29 | H2p25O | 51.576628 | 146.942 | 1.237121 | 11.5918 |
| 30 | pureH | 23.313944 | 93.256 | 0.477561 | 6.959 |
| 31 | pureH | 22.240012 | 88.96 | 0.483503 | 6.6011 |
| 32 | pureH | 19.741162 | 78.965 | 0.481705 | 6.0463 |
| 33 | pureH | 20.026603 | 80.106 | 0.483526 | 5.8079 |
| 34 | pureH | 22.415706 | 89.663 | 0.510981 | 5.7352 |
| 35 | pureH | 22.353921 | 89.416 | 0.487507 | 6.7969 |
| 36 | pureH | 20.401753 | 81.607 | 0.5147 | 6.0813 |
| 37 | pureH | 20.886497 | 83.546 | 0.501062 | 5.8107 |
| 38 | pureH | 19.396752 | 77.587 | 0.490258 | 6.0502 |
| 39 | pureH | 21.043503 | 84.174 | 0.499495 | 7.5778 |

## Saved insertion/deletion pairs: all 20 independent replays

All 20 match saved GRACE dU within the configured 1-meV replay tolerance; the
largest actual printed difference is 0.069857 meV/H.
All 40 completed OUTCARs supplied parsed force/stress data, zero unavailable.
**The numerical force/stress comparison CSV was not pasted**, so its accuracy
cannot be summarized from this chat. It exists remotely in
runs/plots/vasp_edge_replay_1/force_stress.csv.

Full original/single-replayed absolute energies, saved/recomputed dU and residuals:

| pair | host | direction | E_DFT_without_H_eV | E_DFT_with_H_eV | E_GRACE_without_H_eV | E_GRACE_with_H_eV | dU_GRACE_saved_eV | dU_GRACE_recomputed_eV | dU_DFT_eV | paired_DFT_minus_GRACE_eV | replay_difference_meV | match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | H2O | deletion | -936.20992879 | -933.13569054 | -944.635668278 | -943.202962279 | 1.432775855064392 | 1.432705998 | 3.0742382500000076 | 1.6415322520000075 | -0.06985706439199824 | True |
| 1 | H2O | insertion | -954.87816202 | -950.96962727 | -963.313448489 | -960.764041245 | 2.54941987991333 | 2.549407244 | 3.9085347499999443 | 1.3591275059999441 | -0.012635913329894777 | True |
| 2 | H2p083O | deletion | -930.85248911 | -931.62442844 | -954.952019036 | -957.054952085 | -2.10292911529541 | -2.102933049 | -0.7719393300000092 | 1.330993718999991 | -0.00393370459006448 | True |
| 3 | H2p083O | insertion | -942.38168397 | -938.23972641 | -963.860220492 | -961.705195367 | 2.1550601720809937 | 2.155025125 | 4.141957560000037 | 1.9869324350000372 | -0.03504708099377751 | True |
| 4 | H2p25O | deletion | -929.83168446 | -928.04969427 | -970.225629807 | -968.824246883 | 1.4013631343841553 | 1.401382923 | 1.7819901899999877 | 0.3806072669999878 | 0.019788615844618107 | True |
| 5 | H2p25O | insertion | -929.2842636 | -927.02284508 | -978.474361062 | -978.789426625 | -0.3150586485862732 | -0.315065563 | 2.2614185200000065 | 2.5764840830000066 | -0.006914413726799573 | True |
| 6 | pureH | deletion | -353.42824365 | -352.29973516 | -376.114580393 | -375.189633131 | 0.9249526262283325 | 0.924947262 | 1.1285084900000015 | 0.20356122800000154 | -0.005364228332527254 | True |
| 7 | pureH | insertion | -346.87379612 | -346.60655269 | -367.555118084 | -367.659036994 | -0.10392212867736816 | -0.10391891 | 0.2672434300000077 | 0.3711623400000077 | 0.0032186773681608383 | True |
| 8 | H2O | deletion | -927.37967435 | -923.5009287 | -935.547026038 | -933.597831726 | 1.9491826295852661 | 1.949194312 | 3.878745649999928 | 1.929551337999928 | 0.011682414733815705 | True |
| 9 | H2O | insertion | -957.87916417 | -954.33158433 | -964.750172019 | -962.697647572 | 2.0525401830673218 | 2.052524447 | 3.547579839999912 | 1.495055392999912 | -0.015736067321636682 | True |
| 10 | H2p083O | deletion | -931.08724652 | -926.88389257 | -957.025753379 | -954.675386906 | 2.350361943244934 | 2.350366473 | 4.203353950000064 | 1.8529874770000645 | 0.004529755065707519 | True |
| 11 | H2p083O | insertion | -946.65366268 | -941.29403438 | -973.337599516 | -971.015389323 | 2.3222110271453857 | 2.322210193 | 5.359628300000054 | 3.037418107000054 | -0.0008341453856530734 | True |
| 12 | H2p25O | deletion | -948.97929715 | -945.13987989 | -992.693866849 | -990.38290298 | 2.3109620809555054 | 2.310963869 | 3.8394172600000047 | 1.5284533910000047 | 0.001788044494688279 | True |
| 13 | H2p25O | insertion | -934.53377023 | -930.53426396 | -978.727744743 | -977.208210036 | 1.5195622444152832 | 1.519534707 | 3.9995062699999835 | 2.4799715629999834 | -0.027537415283163114 | True |
| 14 | pureH | deletion | -349.99866939 | -349.76366467 | -370.644316256 | -370.229085267 | 0.41521573066711426 | 0.415230989 | 0.23500471999994943 | -0.18022626900005057 | 0.01525833288573697 | True |
| 15 | pureH | insertion | -345.16394986 | -345.40787796 | -368.477890849 | -368.637208462 | -0.1593151092529297 | -0.159317613 | -0.2439281000000051 | -0.08461048700000509 | -0.0025037470703093145 | True |
| 16 | H2O | deletion | -959.65328251 | -957.82314575 | -966.357700884 | -966.167240441 | 0.19044137001037598 | 0.190460443 | 1.830136760000073 | 1.639676317000073 | 0.01907298962403048 | True |
| 17 | H2O | insertion | -943.11780365 | -943.32395792 | -952.938281238 | -954.207374394 | -1.2690843343734741 | -1.269093156 | -0.20615426999995634 | 1.0629388860000437 | -0.008821626525934434 | True |
| 18 | H2p083O | deletion | -949.51968296 | -946.5830415 | -977.091112196 | -976.721596897 | 0.36950671672821045 | 0.3695153 | 2.936641459999919 | 2.567126159999919 | 0.008583271789541769 | True |
| 19 | H2p083O | insertion | -934.77005775 | -932.83041429 | -959.097166896 | -959.641383529 | -0.5441958904266357 | -0.544216633 | 1.9396434599999566 | 2.4838600929999566 | -0.020742573364262995 | True |

Final group summaries, exactly as printed:

| host | direction | pairs_complete | mean_DFT_minus_MLP_eV | rms_DFT_minus_MLP_eV |
|---|---|---|---|---|
| H2O | insertion | 3 | 1.3057072615636873 | 1.3181119426661945 |
| H2O | deletion | 3 | 1.7369199686623158 | 1.7422528444024474 |
| H2p083O | insertion | 3 | 2.5027368785376134 | 2.539249903845321 |
| H2p083O | deletion | 3 | 1.9170357854023148 | 1.9828635515451263 |
| H2p25O | insertion | 2 | 2.52822782282783 | 2.5286883142728405 |
| H2p25O | deletion | 2 | 0.9545303288894615 | 1.1137844629385412 |
| pureH | insertion | 2 | 0.1432759263372816 | 0.2691843394626927 |
| pureH | deletion | 2 | 0.011667479366735733 | 0.19224812329930205 |

All partial analyses (one pair, intermediate, and 38/40 structures complete) are
retained in [paired_partial_analysis_history.csv](grace_dft_audit_2026-11-01/data/paired_partial_analysis_history.csv).
These are successive views of overlapping data, not independent measurements.
The final raw-GRACE means before replay differ slightly from replay by numerical
precision; the exact later source log retains both absolute energies.
Two initial failed points had exit=-9 and were excluded; retries completed in
102.3 and 103.1 s, bringing all 40 to completion. Details are in inline milestones.

## Pure-H Gamma vs 2x2x2: two matched pairs

```text
Exact matched Gamma vs 2x2x2 test, 32 ranks x 4 parallel VASP structures, vasp_std.
p000_without_H TOTEN=-371.189277810 eV wall=198.6 s
p000_with_H TOTEN=-370.211255850 eV wall=199.0 s
PAIR 000 pureH deletion: dU_VASP=0.978022 eV; dU_GRACE=0.924953 eV; VASP-GRACE=+53.07 meV/H; k-mesh shift=-150.49 meV/H
p001_without_H TOTEN=-360.885802000 eV wall=200.2 s
p001_with_H TOTEN=-361.021631520 eV wall=200.8 s
PAIR 001 pureH insertion: dU_VASP=-0.135830 eV; dU_GRACE=-0.103922 eV; VASP-GRACE=-31.91 meV/H; k-mesh shift=-403.07 meV/H
4 selected points complete; 4 attempted; 0 failed/incomplete.
Original Gamma corresponding live values:
p006 pureH deletion dU_VASP=1.128508 dU_GRACE=0.924953 residual=+203.56 meV/H
p007 pureH insertion dU_VASP=0.267243 dU_GRACE=-0.103922 residual=+371.17 meV/H
```

Mesh corrections of -150.49 and -403.07 meV/H are consequential. Residuals happen
to become +53.07 and -31.91 meV/H for these two cases; this does not establish
reservoir convergence or correct the water-host errors. User reports H2O training
was Gamma-only and roughly700 H atoms were needed for pure-H Gamma convergence.
Comparing Gamma-trained H2O labels remains a valid model reproduction test;
denser-mesh physical convergence is a separate target. No quantitative forces
for the two k-mesh pairs were supplied.

## GRACE on original DFT labels: 40 supplied baseline rows

The labels mix training and validation sources; these are not 40 exclusively
held-out configurations. Baseline indices are independently selected and do not
identify the same geometry as normal_000 etc. Temperature/composition metadata
for all these indices remains in the remote baseline.csv, not in the pasted log.

| index | GRACE_minus_DFT_meV_atom | force_component_RMSE_eV_A |
|---|---|---|
| 0 | -3.253 | 0.6032 |
| 1 | -8.483 | 0.7115 |
| 2 | 90.114 | 0.676 |
| 3 | 55.852 | 0.4628 |
| 4 | 62.809 | 0.8225 |
| 5 | -20.747 | 0.6703 |
| 6 | -22.496 | 0.6244 |
| 7 | -26.631 | 0.7495 |
| 8 | 90.877 | 0.7101 |
| 9 | -16.76 | 0.5748 |
| 10 | -3.33 | 0.7036 |
| 11 | 0.276 | 0.8922 |
| 12 | -2.491 | 0.6031 |
| 13 | -1.21 | 0.4098 |
| 14 | 4.319 | 1.0524 |
| 15 | 90.493 | 0.7496 |
| 16 | -14.579 | 0.7295 |
| 17 | -0.892 | 0.7627 |
| 18 | -18.918 | 0.6419 |
| 19 | -5.604 | 0.3961 |
| 20 | -2.112 | 1.0711 |
| 21 | 33.537 | 0.6141 |
| 22 | 7.42 | 0.745 |
| 23 | -7.001 | 0.9632 |
| 24 | 27.092 | 0.6794 |
| 25 | -35.601 | 0.5612 |
| 26 | 13.55 | 0.4104 |
| 27 | 72.874 | 0.8367 |
| 28 | -15.696 | 0.7564 |
| 29 | -10.636 | 0.5952 |
| 30 | 48.223 | 0.6681 |
| 31 | 14.068 | 1.0621 |
| 32 | -16.767 | 0.3735 |
| 33 | -2.361 | 0.6595 |
| 34 | -5.441 | 0.7072 |
| 35 | -18.0 | 0.3977 |
| 36 | -3.673 | 0.767 |
| 37 | -1.573 | 0.7293 |
| 38 | -2.259 | 0.7934 |
| 39 | -10.516 | 0.3961 |

Derived from the printed 40 rows: raw MAE=22.213350 meV/atom, raw
RMSE=34.156097 meV/atom, signed bias=+8.361850 meV/atom. These are summaries
of this mixed original-data selection, not the separate held-out benchmark.

## 5000 K, 600 +/-100 GPa filtered original-label baseline: 24 rows

Requested up to40 distinct original source OUTCARs; 24 printed results supplied.
This is a new subset with new indices, not a matched-temperature perturbation
of the preceding 40. Lower raw energy errors alone cannot isolate temperature.

| index | GRACE_minus_DFT_meV_atom | force_component_RMSE_eV_A |
|---|---|---|
| 0 | -3.253 | 0.6032 |
| 1 | 10.539 | 0.7559 |
| 2 | 3.21 | 0.8182 |
| 3 | 4.183 | 0.7928 |
| 4 | 12.35 | 0.608 |
| 5 | 1.069 | 0.806 |
| 6 | -3.33 | 0.7036 |
| 7 | -2.491 | 0.6031 |
| 8 | -4.152 | 0.7386 |
| 9 | -15.35 | 0.4494 |
| 10 | -19.552 | 0.4531 |
| 11 | -10.434 | 0.556 |
| 12 | -11.355 | 0.3902 |
| 13 | 5.357 | 0.709 |
| 14 | -1.006 | 0.7904 |
| 15 | -13.453 | 0.3969 |
| 16 | -7.586 | 0.7662 |
| 17 | -9.379 | 0.4614 |
| 18 | 19.887 | 0.8325 |
| 19 | 10.19 | 0.7479 |
| 20 | -11.596 | 0.3936 |
| 21 | -29.374 | 0.7893 |
| 22 | -10.684 | 0.53 |
| 23 | -2.259 | 0.7934 |

Derived from these printed 24 rows: raw MAE=9.251625 meV/atom, raw
RMSE=11.484491 meV/atom, signed bias=-3.686208 meV/atom. The selection differs
from the 40-source baseline, so this does not isolate temperature.

## Original held-out valid_m benchmark: 40 configurations

Overall: **shifted E_MAE=21.684725 meV/atom**, F_RMSE=.717290 eV/Angstrom,
S_MAE=2.020483 GPa, signed P_err=-.671949 GPa, N=40.
The code fits H/O reference shifts by least squares on the evaluated subset and
then reports MAE. Those fitted offsets were not printed in the user log.
This is the historical diagnostic convention, not a fixed independently
calibrated energy alignment or raw energy RMSE. Force/stress errors are absolute.

Per composition/temperature (all rows):

| group | shifted_E_MAE_meV_atom | force_RMSE_eV_A | stress_MAE_GPa | signed_P_error_GPa | N |
|---|---|---|---|---|---|
| h2.25o_5000K | 26.264001 | 0.833571 | 2.466153 | -3.461949 | 1.0 |
| h2.25o_7000K | 9.343843 | 0.963281 | 2.972743 | -1.268972 | 2.0 |
| h2.25o_9000K | 14.390073 | 0.707463 | 0.787314 | -1.209123 | 1.0 |
| h2.5o_3000K | 43.750352 | 0.63305 | 1.537263 | 3.368755 | 1.0 |
| h2.75o_3000K | 52.857791 | 0.735608 | 2.672146 | 2.534268 | 1.0 |
| h2o_11000K | 26.434051 | 0.731916 | 1.259913 | -0.452543 | 2.0 |
| h2o_2000K | 35.765376 | 0.283868 | 0.591855 | 0.731597 | 1.0 |
| h2o_3000K | 18.885763 | 0.397755 | 1.580167 | -1.726053 | 2.0 |
| h2o_5000K | 14.181741 | 0.504127 | 1.178963 | -0.412936 | 1.0 |
| h2o_6000K | 25.482975 | 0.613976 | 1.419562 | 1.281989 | 2.0 |
| h2o_7000K | 29.131308 | 0.638917 | 2.060443 | -0.156029 | 2.0 |
| h2o_9000K | 11.995411 | 0.766986 | 2.21477 | -1.018116 | 1.0 |
| h3o_13000K | 42.31131 | 1.092671 | 4.844006 | -9.6071 | 1.0 |
| h3o_2000K | 83.237073 | 0.888858 | 2.342483 | 3.070633 | 1.0 |
| h3o_3000K | 48.705177 | 0.701892 | 2.780001 | -5.932459 | 2.0 |
| h3o_5000K | 14.702633 | 0.80268 | 1.932543 | -0.010229 | 2.0 |
| h3o_7000K | 17.96441 | 0.820311 | 1.793659 | -2.78133 | 2.0 |
| h3o_9000K | 1.869904 | 0.908843 | 2.232529 | 0.360527 | 2.0 |
| h5o_5000K | 3.125054 | 0.954622 | 2.422872 | -1.767859 | 1.0 |
| h5o_7000K | 4.146627 | 0.900481 | 2.021207 | 3.135893 | 1.0 |
| h5o_9000K | 5.629504 | 0.974492 | 2.030947 | -0.172943 | 1.0 |
| h7o_5000K | 11.596561 | 0.674467 | 1.847204 | 1.687438 | 1.0 |
| h7o_7000K | 19.881659 | 0.934434 | 2.997174 | -0.4108 | 1.0 |
| h_11000K | 34.713554 | 0.413868 | 1.181451 | -0.86398 | 1.0 |
| h_3000K | 18.955958 | 0.399022 | 1.877496 | -0.247414 | 2.0 |
| h_5000K | 14.230704 | 0.393591 | 1.499694 | 0.437045 | 1.0 |
| h_7000K | 3.247227 | 0.420144 | 1.791303 | 0.577188 | 2.0 |
| h_9000K | 9.932863 | 0.475774 | 2.39155 | -1.104077 | 2.0 |

By temperature (all rows):

| group | shifted_E_MAE_meV_atom | force_RMSE_eV_A | stress_MAE_GPa | signed_P_error_GPa | N |
|---|---|---|---|---|---|
| 2000 | 59.501224 | 0.659791 | 1.467169 | 1.901115 | 2.0 |
| 3000 | 33.712742 | 0.56591 | 2.085592 | -1.238604 | 8.0 |
| 5000 | 14.114761 | 0.732773 | 1.897139 | -0.505531 | 7.0 |
| 6000 | 25.482975 | 0.613976 | 1.419562 | 1.281989 | 2.0 |
| 7000 | 14.340186 | 0.778148 | 2.225468 | -0.453319 | 10.0 |
| 9000 | 7.945789 | 0.769333 | 2.04017 | -0.555326 | 7.0 |
| 11000 | 29.193886 | 0.643607 | 1.233759 | -0.589689 | 3.0 |
| 13000 | 42.31131 | 1.092671 | 4.844006 | -9.6071 | 1.0 |

Wall timing: 40/40 in04:32,6.80s/iteration; real283.09s,user1102.01s,sys69.06s.
CPU timing used OMP/TF intra-op8,TF inter-op1 in the proposed command. Runtime
includes initialization/compilation; mixed atom counts and no matched VASP
per-step benchmark mean it must not be read as a GRACE/VASP speed ratio.
XLA compilation and pandas groupby FutureWarnings were logging/deprecation
messages, not evidence of failed inference.

## Earlier PWidom mu, transfer dG and provisional hulls

The uploaded export contains 44 finished results:37 host and7 reservoir results,
11 complete manifests and3 manifests reporting running without contributing new
final results. All supplied final result fingerprints d810867382d55724. Prior
and extension runs must not be pooled as independent observations. The collector
reported quality flags for 30/37 host calculations. The original uploaded text is retained byte-for-byte, with its parsed JSON
also retained in the archive, including all stored errors, pressures, frame counts,
run status, seeds, sizes and metadata.

Primary selected table uses the collector's robust scan pressure correction.
Measured-pressure alternatives and all44 results are available in the linked
full-precision CSVs. Formation uncertainties are statistical propagation/bounds
under the documented collector assumptions, not a systematic model-error band.

# Selected PWidom results

Robust scan pressure correction; μ and transfer dG in meV/H. Errors are propagated quoted sigma; robust errors also include the existing partial-volume assumption.

## 300 GPa / 3000 K

| extra H / 108 O | μH ± σ | transfer dG ± σ | quality flags |
|---:|---:|---:|---|
| 0 | -145.8 ± 53.3 | -524.7 ± 56.1 | low Kish |
| 5 | 181.2 ± 44.9 | -197.7 ± 48.2 | ok |
| 9 | 270.7 ± 45.6 | -108.2 ± 48.9 | F/R disagreement |
| 11 | 514.4 ± 78.0 | 135.5 ± 79.9 | ok |
| 16 | 779.9 ± 34.9 | 401.0 ± 39.0 | low Kish;F/R disagreement |
| 22 | 1049.2 ± 44.9 | 670.3 ± 48.2 | low Kish |
| 27 | 965.5 ± 59.9 | 586.6 ± 62.4 | low Kish;F/R disagreement |

Pure-H reservoir G_H = 378.9 meV/H.

## 300 GPa / 4000 K

| extra H / 108 O | μH ± σ | transfer dG ± σ | quality flags |
|---:|---:|---:|---|
| 0 | -634.6 ± 66.8 | -501.1 ± 67.4 | F/R disagreement |
| 5 | -280.1 ± 62.6 | -146.5 ± 63.3 | ok |
| 9 | -74.7 ± 41.1 | 58.9 ± 42.1 | low Kish;F/R disagreement |
| 11 | -63.5 ± 70.9 | 70.0 ± 71.5 | low Kish;F/R disagreement |
| 16 | 312.0 ± 23.6 | 445.5 ± 25.3 | F/R disagreement |
| 22 | 408.2 ± 13.9 | 541.7 ± 16.7 | low Kish;F/R disagreement |
| 27 | 308.7 ± 81.5 | 442.2 ± 82.1 | F/R disagreement |

Pure-H reservoir G_H = -133.6 meV/H.

## 300 GPa / 5000 K

| extra H / 108 O | μH ± σ | transfer dG ± σ | quality flags |
|---:|---:|---:|---|
| 0 | -1103.5 ± 37.9 | -444.2 ± 47.2 | ok |
| 9 | -698.8 ± 12.1 | -39.5 ± 30.5 | ok |

Pure-H reservoir G_H = -659.3 meV/H.

## 500 GPa / 2000 K

| extra H / 108 O | μH ± σ | transfer dG ± σ | quality flags |
|---:|---:|---:|---|
| 0 | 1328.7 ± 62.8 | -1111.5 ± 66.9 | low Kish;F/R disagreement |
| 9 | 1911.4 ± 70.3 | -528.8 ± 74.0 | low Kish;F/R disagreement |
| 27 | 2909.1 ± 82.0 | 468.9 ± 85.2 | F/R disagreement |

Pure-H reservoir G_H = 2440.1 meV/H.

## 500 GPa / 3000 K

| extra H / 108 O | μH ± σ | transfer dG ± σ | quality flags |
|---:|---:|---:|---|
| 0 | 1273.2 ± 37.2 | -759.3 ± 40.4 | low Kish;F/R disagreement |
| 5 | 1403.7 ± 34.1 | -628.7 ± 37.5 | ok |
| 9 | 1899.8 ± 67.2 | -132.7 ± 69.1 | low Kish;F/R disagreement |
| 11 | 1677.2 ± 71.6 | -355.2 ± 73.3 | F/R disagreement |
| 16 | 2161.7 ± 57.1 | 129.3 ± 59.2 | F/R disagreement |
| 22 | 2447.2 ± 57.0 | 414.7 ± 59.1 | low Kish;F/R disagreement |
| 27 | 2886.8 ± 81.0 | 854.4 ± 82.5 | low Kish;F/R disagreement |

Pure-H reservoir G_H = 2032.5 meV/H.

## 600 GPa / 2000 K

| extra H / 108 O | μH ± σ | transfer dG ± σ | quality flags |
|---:|---:|---:|---|
| 0 | 1973.3 ± 77.3 | -1177.6 ± 79.2 | low Kish;F/R disagreement |
| 5 | 2271.8 ± 77.4 | -879.1 ± 79.3 | low Kish;F/R disagreement |
| 9 | 2475.7 ± 72.5 | -675.1 ± 74.6 | low Kish;F/R disagreement |
| 11 | 2615.5 ± 58.6 | -535.4 ± 61.1 | low Kish;F/R disagreement |
| 16 | 3015.3 ± 62.6 | -135.5 ± 64.9 | low Kish;F/R disagreement |
| 22 | 3294.1 ± 108.3 | 143.3 ± 109.7 | ok |
| 27 | 3431.2 ± 82.8 | 280.4 ± 84.6 | low Kish;F/R disagreement |

Pure-H reservoir G_H = 3150.9 meV/H.

State-level sampled-node minima:

| P_GPa | T_K | points | max_extra_H | dg_at_H2O_meV | dg_at_H2O_err_meV | min_sampled_extra_H | min_sampled_formation_meV_atom | min_formation_stat_sigma_bound |
|---|---|---|---|---|---|---|---|---|
| 300.0 | 3000.0 | 7 | 27 | -524.6798704596958 | 56.08256177687076 | 9 | -7.259854984890503 | 1.3655902583224555 |
| 300.0 | 4000.0 | 7 | 27 | -501.0739549754154 | 67.43395066683749 | 9 | -5.388222934305462 | 1.6147274831908291 |
| 300.0 | 5000.0 | 2 | 9 | -444.1924475427518 | 47.16005744429939 | 9 | -6.536922855114889 | 1.0498518034899254 |
| 500.0 | 2000.0 | 3 | 27 | -1111.4676946557452 | 66.90049349843973 | 27 | -22.562133040757203 | 5.886056890166336 |
| 500.0 | 3000.0 | 7 | 27 | -759.2832375862454 | 40.37514430001641 | 16 | -17.781376013380637 | 2.5931071721738257 |
| 600.0 | 2000.0 | 7 | 27 | -1177.5731075611773 | 79.18427567062204 | 16 | -32.75880898450391 | 3.396132065584677 |

These minima are sampled-node values, not validated equilibrium compositions or
converged DFT hulls. The 300 GPa/5000 K H2.25O melted result is not an SI-ice node.
The time plots used0.1-ps cumulative prefixes up to1ps and showed drift. Their
underlying prefix arrays and numerical uncertainty bands were not uploaded;
no exact digitized values are invented here. Equilibration0.2/0.4ps was questioned,
and energy stationarity alone is insufficient for structural equilibration.

Earlier repository numerical snapshots remain relevant and are preserved:

- [Young full TDI numerical summary](young_tdi_summary_received_2026-10-31.txt).
- [500 GPa/3000 K original scan](../data/pwidom_pm_500GPa_3000K_reported.csv),
  [extension1](../data/pwidom_pm_500GPa_3000K_extension1_reported.csv),
  [nscan1](../data/pwidom_pm_500GPa_3000K_nscan1_reported.csv), and
  [combined provisional integral](../data/pwidom_pm_500GPa_3000K_combined_reported.csv).
- [Perlmutter resource benchmark numbers](../data/perlmutter_pwidom_benchmarks_20261031.csv)
  and [nscan resource excerpt](../data/pwidom_nscan1_resources_excerpt.csv).
- Existing numerical comparison files: dg_fcc_bcc.csv, melt_dG_grace.csv,
  eos_crosscheck.csv, insertion_regsol_fit.csv, liquid_mix_grace.csv in docs/data.
  These are historical calculations, not newly DFT-validated quantities.

## EOS inputs and submitted DFT jobs

The pasted EOS query used416 ice,366H2.25O,301H sources in exact5000-K bins.
Reported global fit domains: iceP139–2506GPa,V/atom1.152–3.275A^3;
H2.25OP287–2760GPa,V/atom1.063–2.521; HP284–1290GPa,V/atom.813–1.543.
These are fit coverage counts, not independent equilibrium samples or proof of
phase purity. The phase filters are path-based. No NkBT/V was added twice.

| P GPa | host | V/atom A^3 | B GPa |
|---|---|---|---|
|400|H2O|2.2896352|1182.0494|
|400|H2.25O|2.2451198|1057.3255|
|400|H|1.3895199|889.984|
|600|H2O|1.9935375|1731.4057|
|600|H2.25O|1.9320333|1636.6127|
|600|H|1.1599448|1343.162|

Exact500-GPa EOS values were not printed in this submission paste; they reside
in the remote state JSON. No values interpolated from400/600 are substituted.
EOS export printed “pure H250atoms”, but dft-plan --h-natoms729 overrides the
simulation atom count at unchanged V/atom. New configurations have729/730 H;
old250/251 configurations are explicitly separate diagnostics.

Confirmed submitted jobs:60649379=500GPa,60649380=400GPa,60649381=600GPa;
all5000K,twoCPU nodes,256tasks,four64-rank systems,48h allowance. Later worker-log evidence confirms production MD activity for all three states;
no completed DFT mu/dG has been supplied. These are historical pasted snapshots,
not a live queue view.
Three jobs reserve at most288node-hours. Accountm3349,qosregular; actual Slurm
account/QoS limits remain authoritative.

## VASP force-engine smoke tests, restart defect and measured timings

Early ASE/VASP calls were all cold. Complete pasted timing/energy/checkpoint
histories are archived in the two cold logs, including60–110s typical ice calls,
19–26s typical250-H calls, and occasional198–295s outliers. These are distinct
from historical nativeVASP LOOP+ real times7.32–18.65s. The original native-MD
core count was not established. Eight workers originally meant four hosts and
two replicas at32ranks, not eight lambda windows. The user requested four64-rank
systems instead. One128-task allocation does not support eight32-rank workers.

Actual failing setup: checkout290c3ca; conflicting LWAVE.FALSE/TRUE entries,
ISTART0,zero-byte WAVECAR. Code6f532a7 switched to unique restart keys and729-H
default;50f6413 fixed a CLI parser typo introduced in55fe2de. Missing plan JSON
and that TypeError prevented launch; neither is physical evidence about H uptake.
No pre-fix numerical result was silently accepted as a warm-start result.

The latest user paste confirms actual warm=True calls and729-H MD:

| host | call | natoms | wall_seconds | warm | energy_eV |
|---|---|---|---|---|---|
| H2O | 2 | 324 | 33.8 | True | -585.79596926 |
| H2O | 3 | 324 | 41.0 | True | -584.16515973 |
| H2O | 4 | 324 | 34.5 | True | -585.1756711 |
| H2O | 5 | 324 | 33.8 | True | -591.60045113 |
| H2O | 6 | 324 | 37.1 | True | -598.04864293 |
| H2p083O | 1 | 333 | 60.5 | False | -599.46277225 |
| H2p083O | 2 | 333 | 29.7 | True | -598.33754693 |
| H2p083O | 3 | 333 | 34.8 | True | -593.24063116 |
| H2p083O | 4 | 333 | 31.7 | True | -588.58285971 |
| H2p083O | 5 | 333 | 40.0 | True | -590.08268907 |
| H2p083O | 6 | 333 | 34.1 | True | -592.50107656 |
| H2p083O | 7 | 333 | 42.0 | True | -591.73482096 |
| H2p25O | 1 | 351 | 70.7 | False | -603.81695604 |
| H2p25O | 2 | 351 | 41.4 | True | -586.8033317 |
| H2p25O | 3 | 351 | 39.7 | True | -575.72219214 |
| H2p25O | 4 | 351 | 45.0 | True | -569.60205636 |
| H2p25O | 5 | 351 | 45.8 | True | -562.80297886 |
| H2p25O | 6 | 351 | 42.1 | True | -560.86814564 |
| pureH729 | 1 | 729 | 77.5 | False | -827.23592738 |
| pureH729 | 2 | 729 | 64.5 | True | -821.16960037 |
| pureH729 | 3 | 729 | 62.2 | True | -802.48001151 |
| pureH729 | 4 | 729 | 68.6 | True | -770.83166914 |

Latest checkpoint evidence: H2O host24/24,5/5frames,full0/24; H2.083O/H2.25O
host resumed11/24,1/5frames; H729started0/24. This proves force-engine execution
and checkpoint continuation, not a completed 20-pair physical DFT free energy.
The smoke budget is2fs equilibration+10fs sampling in each ensemble,dt.5fs:
24 steps/ensemble, 5 frames every 2 fs, one insertion/deletion per frame. This is only
a plumbing/timing test and cannot establish equilibrium, superionicity or melting.
Large initial energy transients are visible; fresh lattice starts are not
pre-equilibrated DFT trajectories.

At current early warm speeds,6000 production MD steps imply roughly57–68h H2O,
50–70h H2.083O,67–77h H2.25O,103–115h H729 BEFORE trial evaluations. These ranges
are rough extrapolations from small transients, not confidence intervals or
completion guarantees. Four workers are concurrent, so their times are not added.
Production sampling1ps equilibration+.5ps sampling perensemble,dt.5fs,requires
3000steps each/6000total;25frames/ensemble every20fs,4trials/frame means200trial
differences plus any uncached base energies. Consequently48h completion is not
assured, despite checkpoint/resubmission support. Previous9s/step-based promises
were too optimistic for the measured ASE execution path.

User's crystal-TDI alternative: one128-core node for24h at assumed9–18s/step,
dt.5fs gives2.4–4.8ps forone lambda,or.24–.48ps/window for ten sequential windows.
Ten nodes with one window each would give2.4–4.8ps/window. All includeeqtime;
packing windows changes cores/window and must be benchmarked. NativeVASP MD
may retain electronic prediction beyond separate ASE-launched singlepoints;
that is an optimisation hypothesis, not demonstrated throughput here.


## Later production update: all three jobs, approximately four hours in

The final supplied tails contain 107 accepted warm=True force-call lines across
12 workers. H cells are729atoms at all three states. The water cells are324,
333 and351atoms. All checkpoints are still in host-ensemble equilibration:
2000steps at0.5fs must elapse before recording production frames. Thus0/25frames
is expected, not an output or resume failure. Host+H dynamics and trial sampling
still follow the host trajectory. The user estimated roughly4h elapsed; no exact
Slurm timestamp was supplied, so no exact global speed is inferred from that.

The500-GPa plan references the old `_1` first-replica data directories;400/600
write under `_4x64`. This is deliberate layout reuse, not evidence of duplicate
workers. Tail median/min/max below uses only eight or nine recent calls per worker.
Remaining time assumes the latest median persists and uses the last SAVED
checkpoint (slightly behind the live force count); includes remaining host and
full MD but excludes trials, SCF/startup changes and any interruption. It is an
estimate, not a promise or an uncertainty interval.

| P_GPa | host | natoms | checkpoint_step | host_total_steps | simulated_host_ps | production_frames | target_frames | tail_median_seconds | tail_min_seconds | tail_max_seconds | remaining_both_MD_hours | remaining_equil_hours |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 400 | H2O_rep0 | 324 | 415 | 3000 | 0.2075 | 0 | 25 | 45.8 | 35.6 | 54.6 | 71.0536 | 20.1647 |
| 400 | H2p083O_rep0 | 333 | 422 | 3000 | 0.211 | 0 | 25 | 39.6 | 35.9 | 48.4 | 61.358 | 17.358 |
| 400 | H2p25O_rep0 | 351 | 386 | 3000 | 0.193 | 0 | 25 | 46.1 | 39.1 | 59.7 | 71.8904 | 20.6682 |
| 400 | pureH_rep0 | 729 | 242 | 3000 | 0.121 | 0 | 25 | 69.6 | 64.8 | 75.2 | 111.3213 | 33.988 |
| 500 | H2O_rep0 | 324 | 508 | 3000 | 0.254 | 0 | 25 | 33.7 | 30.2 | 41.4 | 51.4112 | 13.9668 |
| 500 | H2p083O_rep0 | 333 | 491 | 3000 | 0.2455 | 0 | 25 | 35.1 | 31.4 | 41.7 | 53.7127 | 14.7127 |
| 500 | H2p25O_rep0 | 351 | 377 | 3000 | 0.1885 | 0 | 25 | 41.6 | 39.0 | 54.9 | 64.9769 | 18.7547 |
| 500 | pureH729_rep0 | 729 | 276 | 3000 | 0.138 | 0 | 25 | 61.5 | 54.2 | 66.7 | 97.785 | 29.4517 |
| 600 | H2O_rep0 | 324 | 528 | 3000 | 0.264 | 0 | 25 | 32.0 | 30.5 | 34.1 | 48.64 | 13.0844 |
| 600 | H2p083O_rep0 | 333 | 480 | 3000 | 0.24 | 0 | 25 | 35.6 | 29.8 | 41.7 | 54.5867 | 15.0311 |
| 600 | H2p25O_rep0 | 351 | 434 | 3000 | 0.217 | 0 | 25 | 39.9 | 35.3 | 50.0 | 61.6898 | 17.3565 |
| 600 | pureH_rep0 | 729 | 266 | 3000 | 0.133 | 0 | 25 | 61.2 | 53.2 | 65.9 | 97.478 | 29.478 |


All 107 force-call energies/timings are in
[data/production_latest_calls.csv](grace_dft_audit_2026-11-01/data/production_latest_calls.csv);
the complete pasted log and the earlier inline tail's additional values are
archived in raw/. No temperature, pressure, MSD, structural-order, diffusion,
SCF-iteration, or completed-work-array output was supplied in these tails.
Consequently warm starts and numerical stepping are verified here, whereas
thermal equilibration, SI-phase retention, physically correct density and
converged uptake remain unverified. The instantaneous total energies alone do
not answer those questions. At the measured rate the larger H reservoir remains
the slowest worker; the earlier48h-completion caveat still applies.

## What remains to check before interpreting or retraining

1. Exact training provenance: model export/checkpoint, dataset tier and split,
   frozen-vs-unfrozen variables, loss weights, convergence history. Recalled
   approximately700frames/1–2h fitting is not a recovered log. Benchmark raw and
   fixed-reference errors alongside shifted MAE; do not fit validation offsets
   and then describe them as an independent calibration.
2. Matched H2.25O DFT configurations across temperature, phase, pressure and cell
   size. Evaluate GRACE on original DFT ensembles and VASP on GRACE ensembles.
   Current comparisons confound those factors and show configuration dependence.
3. Audit MD inputs: stoichiometry, true cell/volume, atom order, force path,
   timestep, thermostat units, pressure kinetic term, initial geometry and
   equilibration. Check O-sublattice order/MSD and H diffusion; “fcc” in a path
   is not proof of the dynamical phase. Energy stationarity alone is insufficient.
4. Energy/force consistency, direct vs MD calculator and finite differences.
   Paired batch-vs-single ENERGY replay passed; a full real-GRACE finite-difference
   force check and validation of every TDI force path remain unreported.
5. Larger-H and k-mesh convergence at relevant P/T;729 is a requested size policy,
   not a new convergence proof. Bulk modulus/pressure corrections inherited from
   fits also need validation at the actual enlarged cell and sampled pressures.
6. Completed DFT forward/reverse work distributions, overlap/Kish, block/time
   convergence, pressure and phase diagnostics. Do not interpret selected pair
   residual averages as mu corrections, nor completion as equilibrium convergence.
7. Retraining: retain ordinary and perturbed geometries with correct TOTEN/forces/
   stress labels; both pair members stay in a split. Whole source-run holdouts
   and correlation assessment are preferable to exact-frame-only splitting.
   Compare current, longer-trained, unfrozen and augmented-data challengers under
   a fixed validation protocol. Existing dataset/fitprep provides exact-geometry
   safeguards, not a guarantee against nearby-frame or historical-data leakage.
8. Acceptance includes held-out paired H-addition/transfer errors and weighted
   tails, ordinary forces/stress, and preservation of EOS,bcc–fcc,melting behaviour.
   Proposed20–50meV/H bias/50–100meV/H scatter are working goals, not universal
   sufficient criteria. Updated model sampling/free energies must be rerun after
   validating the challenger; static improvement alone does not validate a hull.

## Data locations and reproducibility

Perlmutter repo /pscratch/sd/v/victornr/H3O; GRACE Python
/pscratch/sd/v/victornr/grace-env/bin/python. Original DFT roots ~/work/tdi/H3O
and ~/work/quartz-work/tdi/h3o. Derived data under runs/, inventories
mined_h3o.csv/inventory_binned.csv, snapshots_s/m/l.extxyz,train/valid_m.extxyz.
Source metadata is relative to ~/work, e.g. ../tdi/H3O/... from the repo. Bundles
runs/H3O_bundle_20261025.tar.gz and20261026.tar.gz. Young repo
/lustre/scratch/mmm0037/H3O with tdi,tdi_long,tdi_prep,tdi_fb,etc. and cached
<tag>_node<N>_traj.extxyz plus <tag>_stage0.json. Use actual lambda values: a
largest quadrature node is not necessarily exactlylambda1. No transfer needed
for original DFT labels already on Perlmutter.

Remote diagnostic roots: runs/vasp_one_snapshot_1,vasp_saved_40_1,
vasp_pureH_k222_1,vasp_training_40_1,vasp_normal_40_1;
runs/plots/grace_training_baseline_1,grace_baseline_5000K_600GPa_1,
vasp_edge_replay_1. DFT500-GPa layout references original first-replica directories
under runs/dft_pwidom_500GPa_5000K_1; the four-worker plan is
runs/dft_pwidom_500GPa_5000K_4x64. H migration uses pureH729_rep0 rather than
resizing old checkpoints. New400/600-GPa plans store their own worker data.

See [validation workflow](../PWIDOM_DFT_VALIDATION.md),
[pure-VASP workflow](../PWIDOM_PURE_VASP.md), and [VASP checks](../PWIDOM_VASP.md).
This report adds evidence; it does not modify simulation inputs, submit/cancel
jobs, publish an environment, or replace existing result provenance.
