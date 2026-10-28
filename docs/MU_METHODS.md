# Routes to mu_H and dg = mu_H(H in SI ice) - G_H(pure H fluid) that do not depend on the insertion TDI

Started 2026-10-28. What we want: dg in the dilute limit and its P and T dependence, and the solubility (common tangent of the solid hull with the H fluid), at
300-600 GPa and 3000-7000 K, with GRACE (classical nuclei), preferably at constant volume. The current route is ghost-proton insertion TDI along
U = U_host + lambda (U_full - U_host), checked by Widom/BAR (noisy at 500 GPa) and by the pure-H method check. Each route below says what it computes,
which assumptions it shares with the insertion TDI, its cost on GRACE, its expected error and how it fails.

GRACE timings used for costs (docs/PERFORMANCE.md): one call on ~325 atoms is 0.16 s alone. Several processes sharing one A100 give ~20 calls/s in total
(8 processes x 2 calls per 0.75 s step). Independent single points are the cheapest work there is: no MD, no time ordering, and any number can share a call.

What the insertion TDI assumes, so we can say which routes are independent of it:
- (a) the soft-core path: lambda -> 0 samples H overlapping O/H, where GRACE has no training data (P2 in CURRENT_ISSUES)
- (b) quadrature in lambda (P3; Kronrod - Gauss was not negligible on the toy)
- (c) sampling at each lambda: heavy tails at small lambda, 2-ps runs
- (d) the reference bookkeeping (ln N!, -kT ln V, Lambda): exact, tested by T0
- (e) the pressure correction from P_host, P_full and B
- (f) the model itself (GRACE vs DFT) and classical nuclei

Every route shares (f). Only direct coexistence avoids (d) and (e).

## Ranking

| rank | route | gives | avoids | GRACE cost per (P, T) | expected error | status |
|---|---|---|---|---|---|---|
| 1 | **R1 proxy-guided Widom / deletion + BAR** (`tdi/pwidom.py`) | mu_ex in any host at fixed V, so dg, mu_H(x) and dg(T) | (a) (b) (c) | 2 x 4 ps MD (2 x 21 min, can run in parallel) + ~3000 batched single points (3-10 min) | +-15-30 meV per host if the GRACE-vs-proxy gap is <~ 1.5 kT | **toy: validated (below)**; GRACE test commands below |
| 2 | R2 direct NVT coexistence (ice slab + H-fluid slab) | the H content of ice in equilibrium with the fluid, so the solubility directly | (a)-(e) | ~650 atoms, 20-50 ps: 1-2 GPU days | x to +-0.01-0.02 H/O; interface and finite-size effects | planned; see the notes below |
| 3 | R3 Hamiltonian replica exchange over the existing lambda grid | same dF as the TDI, better sampled | (c) partly | free: every replica already has U_full - U_host, so a swap needs no new model calls | heavy-tail error at small lambda drops | not done: a small change to tdi/batched.py |
| 4 | R4 nonequilibrium switching (Jarzynski / Crooks, batched) | same dF as the TDI | (b), and (c) as a diagnostic | about the cost of the TDI | similar to the TDI; forward/reverse hysteresis is a free check | no |
| 5 | R7 composition fluctuations in blocks of O Wigner-Seitz cells (small-system Kirkwood-Buff) | d mu_H / d x at fixed V (the hull curvature), not mu | (a)-(d) | none (existing trajectories) | 20-50% on the slope (finite size) | no |
| 6 | R5 dual-box transfer (one lambda run: H decoupled from pure H and coupled into ice) | dg in one run | only the bookkeeping, which T0 already shows is exact | = insertion into ice + insertion into H | the same | not worth doing |
| 7 | R6 expanded ensemble / continuous fractional component | = R3 in one trajectory | (b) | as the TDI | as R3; needs Wang-Landau weights | R3 is simpler with our batched engine |
| 8 | R9 temperature reweighting of the lambda series | d dg / dT from one T | -- | none | too noisy (below) | rejected: run R1 at two T instead |
| -- | R8 PMF / lattice gas on the O lattice | interpretation (site energies, site occupancies) | -- | none | the model is approximate | interpretation only |
| -- | R10 normalizing flows / targeted FEP to pure H | dg | (a)-(c) | weeks of development | no track record for 300+ atoms of a dense mixture | no |
| -- | R11 2PT / entropy from the velocity spectrum | S of each phase | -- | none | ~0.1 kB/atom: amplified x10 per added H | no |
| -- | R12 grand-canonical MC with R1's moves (CBMC insertion and deletion with the proxy) | x_eq at mu_H = G_H | (a)-(c) | see R1 | the acceptance becomes ~ the Kish fraction of R1 instead of ~0 | follow-up if R1 works on GRACE |

## R1. Proxy-guided (Rosenbluth) Widom insertion and deletion, combined with BAR

**What it computes.** The excess chemical potential mu_ex of one H in a host frame ensemble at fixed V, so mu_H(V) = mu_ex + kT ln((N_H+1)/V) + 3 kT ln Lambda.
It uses the same reference counting and the same stage-0 pressure correction as the insertion TDI. dg = mu_H(ice, P) - [mu_H(pure H, P) - fs], where fs is the
ideal-gas finite-size term of the toy scan.

**How.** Plain Widom fails at 450-500 GPa because almost every uniform point overlaps an atom. The average is then carried by rare points that are never drawn.
Here each trial does four things:
1. draws K uniform candidate points (K ~ 10^3-10^4)
2. scores all of them with a cheap pair-potential proxy phi (the toy_fit.py spline pairs fitted to DFT, docs/data/toyfit_t4.json; numpy, ms per frame)
3. picks one candidate with probability exp(-beta phi_k) / sum
4. calls GRACE only at that point

The estimators are exactly unbiased for any proxy; the proxy only sets the variance (derivation in the docstring):
- insertion: exp(-beta mu_ex) = < W exp(-beta (dU_s - phi_s)) >, with W = mean_k exp(-beta phi_k)
- deletion: exp(+beta mu_ex) = < exp(beta (dU_t - phi_t)) / W_t >, with W_t built from the real H t plus K - 1 fresh candidates

Both are the same extended-space work w = dU - phi - kT ln W, so BAR combines them. `--proxy none` is plain Widom and deletion.

**Why it is independent.**
- No lambda path: GRACE is only evaluated at fully coupled, physical configurations, which are the H positions the proxy finds favourable. So there is no soft core, no overlap region and no quadrature.
- No MD at intermediate lambda: the only MD is plain free NVT of the host and of host + 1 H, which stage 0 already runs.
- Insertion and deletion fail in opposite directions. Their agreement is the built-in convergence check, as Widom vs deletion was on the toy.

It shares (d), (e) and (f) with the insertion TDI.

**What it exploits.** The H sublattice is fluid: in each host frame a proton finds favourable places, and the pair proxy knows roughly where they are. Single points
are independent, so they batch perfectly on the GPU (`GraceBatch`, `--batch`). The pure-H reservoir is treated the same way, with the same code and the same proxy.

**Cost on GRACE (one host, 500 GPa).**
- MD: 4 ps of free NVT for the host and 4 ps for host + 1 H, 8000 steps x 0.16 s = 21 min each (two processes on one GPU).
- Single points: 200 frames x (1 + 4 insertions) + 200 x (1 + 4 deletions) = 2000 calls, 2-6 min batched.

So about 45 min of one GPU per host, and dg needs ice and pure H. The insertion TDI needs 10-21 lambda x 2-4 ps of 2-call MD, roughly 10x more.

**Expected error.** The error is set by two things:
- frame-to-frame scatter of W (cheap, so use many frames)
- std of (dU_GRACE - phi) at the selected points: below ~1 kT the Kish effective fraction is > 0.5; at 2-3 kT it is ~0.05-0.2 (still fine with 10^3 points); at > 4 kT the method degrades to plain Widom

The script prints both diagnostics. The pair fit has force RMSE 1.5 eV/A against DFT, so on GRACE the gap is likely 1-3 kT at 6000 K. If it is larger, a better proxy
(refit the pairs to the first ~200 GRACE dU values; a fixed proxy chosen before the production sampling keeps the estimator unbiased) or a larger K helps.

**Failure modes.**
- The proxy misses favourable regions that GRACE has (e.g. H2 or H3+-like local states): then insertion is low-biased at finite sample and deletion is high-biased; they disagree, and BAR is still consistent if the overlap is nonzero.
- Rare host cavities: the frame-to-frame scatter of W needs a long host MD. On the toy, 4 -> 12 ps moved plain BAR by 80 meV, but here W is computed exactly per frame with K candidates, so only the host sampling is left.
- The pressure correction is the same as the TDI's: keep the host near P_target.

**Toy results (fitted toy potential toyfit_t4 = the "truth"; 450 GPa):** see "R1 on the toy" below.

## R2. Direct coexistence at constant V (planned elsewhere)

An ice slab (O lattice strained laterally to its own EOS at P) and a pure-H fluid slab in one periodic cell; the fluid sets the normal pressure. Count the H in
the ice region (O Voronoi cells or a smooth counting function) after equilibration. H diffuses 40-90 A^2/ps, so a 10-A slab equilibrates in ~10 ps.

It avoids every assumption except the model. Costs:
- ~2x atoms (0.3 s per call)
- 20-50 ps
- the interface: excess H at the surface, so measure in the slab interior
- finite-size effects (two interfaces in a ~20-A cell)

The lateral strain of the ice must match P, otherwise the slab sits at the wrong chemical potential.

Biased variant: an umbrella potential on N_H(ice) gives F(N_H), whose slope is mu_H(x) - mu_H(fluid). That is the whole curve, not just x_eq. R1 is the cheap
cross-check for each point.

## R3. Hamiltonian replica exchange over the lambda grid (improves the TDI; shares (a), (d), (e))

The path is linear, so U_lambda(x) = U_host(x) + lambda u(x), with u = U_full - U_host already computed at every step of every replica. A swap of
configurations between lambda_i and lambda_j is accepted with min(1, exp[-beta (lambda_i - lambda_j)(u_j - u_i)]). This costs **zero extra model calls**. In the batched engine all
replicas live in one process, so swaps every ~50 steps are a few lines in tdi/batched.py (lock-step already).

The small-lambda replicas then receive configurations from larger lambda, which cuts the heavy tail at small lambda (P2). Analyse with MBAR (tdi/bar_lambda.py) and
use all replicas.

Caveats:
- the per-lambda series is no longer one trajectory, so restarts and the per-lambda drift columns need care
- it must stay opt-in (results and file formats unchanged)
- neighbouring overlap on the 10-value grid is ~0.05 for insertion, so the Gauss-Kronrod 21-value grid is better for exchange

## R4. Nonequilibrium switching (Jarzynski / Crooks), batched

Many short (0.2-1 ps) switches lambda 0 -> 1 from independent host frames and 1 -> 0 from full frames; BAR on the work distributions. Pros:
- the errors are honest: the trajectories are independent, so no autocorrelation analysis
- forward/reverse hysteresis is a direct measure of dissipation
- no quadrature
- B switches in one GRACE call

Costs and limits:
- same path as the TDI, so (a) is shared
- the cost per (P, T) is about that of the TDI (switch length x number of switches ~ 10 x 1 ps x 2 directions x 2 calls/step)
- the H-sublattice relaxation is fast (~0.1 ps), so the dissipation should be small

Worth it only as an alternative estimator of the same path.

## R5. Dual-box transfer, and R6 expanded ensemble / continuous fractional component

Dual box: U = U_ice(host) + U_H(rest) + lambda u_ice(t_A) + (1 - lambda) u_H(t_B), with two copies of the transferred H. The ideal and placement terms cancel, and the
run gives mu_ex(ice) - mu_ex(H) directly.
- The two boxes are independent, so the variances add: no noise cancellation.
- The cost is the two insertion runs we already do.
- The bookkeeping it cancels is exact already (T0).

Not worth a new code path.

Expanded ensemble / CFC samples lambda as a variable with Wang-Landau weights. The free energy is the same as the TDI's and the path is the same. With the
batched engine, R3 gives the same mixing between lambda values without weights.

## R7. Composition fluctuations in subvolumes (small-system Kirkwood-Buff)

In a solid O matrix with fluid H, the variance of N_H in subvolumes of size L gives (d N_H / d beta mu_H) at fixed V and fixed O lattice, extrapolated as
Gamma(L) = Gamma_inf + c / L. A periodic box with fixed total N_H needs the finite-box correction (Kruger-Vlugt).
- It is a derivative, so it gives the hull curvature and d mu_H / d x, not dg.
- Cost: zero, on the existing traj files.
- Built-in check: the same analysis on the pure-H fluid must reproduce kT rho kappa_T from the EOS (B).
- Error: 108 O / 216 H boxes are small (L ~ 13 A), so expect 20-50% on the curvature.

Convert to constant P with (d mu / d N)_P = (d mu / d N)_V - v_H^2 B / V, where v_H comes from P_full - P_host.

There is a catch in a crystal: a randomly placed cube also counts O, and its O number fluctuates for geometric reasons (lattice discreteness), not
thermodynamic ones. Use subvolumes made of whole O Wigner-Seitz cells: blocks of m neighbouring O, counting the H closest to them. N_O is then fixed, and the H
variance against m gives d N_H / d beta mu_H per O at fixed O lattice. This uses the O lattice as a natural grid.

Compare with R1 run on an H2.25O host. That gives mu_H(x = 0.25) directly at the same cost as x = 0, and is the more direct route to the curvature.

## R9. Temperature reweighting of the lambda series: why not

d(beta dF)/d beta = int dlambda [<u>_lambda - beta cov_lambda(u, U_lambda)]. This is the fixed-V Gibbs-Helmholtz route in disguise. The covariance with the total
energy U (sigma_U ~ 15 eV per frame on GRACE) has a statistical error of about sigma_u sigma_U / sqrt(n_eff) per lambda. With sigma_u ~ 1-5 eV and n_eff ~ 10^3, that is
0.5-2 eV: far more than the ~0.05 eV per 1000 K we want.

Truncating U to the per-atom energies within ~6 A of the ghost would make it usable, but it is an approximation of unknown bias. R1 at two or three T (45 min each) is
cheaper and exact.

## R8, R10-R12 (one line each)

- **R8.** PMF on the O lattice: mu = kT ln(rho(r) Lambda^3) + mu_ex(r) holds at any point r (inhomogeneous potential-distribution theorem). Averaging over r is R1,
  so the PMF adds nothing quantitative, but site energies and occupancies are useful for interpretation (lattice-gas entropy, ideal-solution reference, CURRENT_ISSUES P1).
- **R10.** Flows and targeted FEP need a learned map between dense 300-atom mixtures. That is research in itself, and its errors are hard to certify.
- **R11.** 2PT entropy is approximate (gas-solid partition of the velocity spectrum) to ~0.1 kB/atom; differences per added H amplify that.
- **R12.** Gibbs-ensemble / grand-canonical moves had ~0 acceptance. With R1's Rosenbluth moves the acceptance becomes ~exp(-beta |w - mu|), comparable to R1's Kish
  fraction. At fixed V and mu_H = G_H(P) this samples x_eq directly, but each composition sits at a slightly different P, so iterate V 2-3 times.

## R1 on the toy (CPU, sandbox)
(in progress: smoke test at vpa 2.23, 20 frames: proxy-guided BAR 6548 +- 21 meV with an exact proxy, 6574 +- 53 with a deliberately poor one; plain Widom vs deletion disagree by 870 meV on the same frames. Full 4-ps comparison with the insertion TDI to follow.)
