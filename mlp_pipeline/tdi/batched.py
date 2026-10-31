"""Batched lambda engine: all lambda values of a TDI run in ONE process, one model call per MD step for all of them (instead of one process per lambda value
time-slicing the GPU). Selected with run_real.py --batch-nodes "0 1 2 ..." (young_tdi_par.sh ENGINE=batch).

How it stays identical to the per-lambda engine: every lambda value runs the unchanged run_real.run_node() (LambdaRun, ASE Langevin with its own seeded RNG, the same
chunks, node files, state files, restart checks, logs) in its own thread. Only the model evaluations are routed through a Batcher: a thread that needs energies / forces
submits its structures and waits; when every running lambda thread is waiting, ONE batched call evaluates all of them and each thread continues with its own result.
The integrator, thermostat, RNG streams and bookkeeping are untouched, so the node files have exactly the format (and, with a deterministic model, exactly the values)
of the per-lambda engine. Insertion: the full system and the host alone are submitted together (one call per step instead of two). The MLP pressure of the thermo log
reuses the stress of the same evaluation (same positions) instead of a second model call.

Evaluators (what one batched call does):
  GenericEvaluator  any ASE calculator, one structure after the other with one calculator instance per lambda value (exactly what the per-lambda processes do):
                    used for the toy models; reproduces the per-lambda engine bit for bit (toy_validate / tests).
  GraceBatch        tensorpotential TPCalculator: the structures are joined into one disjoint graph (the model's own data builders, bond indices offset per
                    structure), one model.compute() call; energies = per-structure sums of the atomic energies, forces per atom, stress from the per-bond pair forces.
                    A self-check against the single-structure calculator runs first (run_batched refuses to start if they differ).
"""
import os, sys, threading, time, traceback
import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class Batcher:
    """lock-step gatherer. register(slot) every lambda thread before it starts; evaluate(slot, structures, stress) blocks until every registered slot has
    submitted (or left with unregister), then the last one to arrive runs one evaluator call for all of them."""

    def __init__(self, evaluator):
        self.ev = evaluator; self.cv = threading.Condition(); self.active = set(); self.pending = {}; self.results = {}
        self.ncalls = 0; self.nstruct = 0; self.t_model = 0.0; self.t0 = time.time()

    def register(self, slot):
        with self.cv:
            self.active.add(slot)

    def unregister(self, slot):
        with self.cv:
            self.active.discard(slot); self.pending.pop(slot, None)
            if self.pending and len(self.pending) == len(self.active):
                self._run()

    def evaluate(self, slot, structs, stress=False):
        with self.cv:
            self.pending[slot] = (structs, stress)
            if len(self.pending) == len(self.active):
                self._run()
            while slot not in self.results:
                self.cv.wait()
            r = self.results.pop(slot)
        if isinstance(r, BaseException):
            raise RuntimeError(f"batched model call failed: {r!r}") from r
        return r

    def _run(self):
        """called with the lock held and every active slot waiting: one evaluator call, results handed back"""
        slots = sorted(self.pending); reqs = [self.pending[s] for s in slots]; self.pending = {}
        t = time.time()
        try:
            out = self.ev(slots, reqs)
            for s, o in zip(slots, out):
                self.results[s] = o
        except BaseException as ex:   # every waiting thread gets the error (e.g. GPU out of memory) and fails its lambda value
            for s in slots:
                self.results[s] = ex
        self.t_model += time.time() - t; self.ncalls += 1; self.nstruct += sum(len(r[0]) for r in reqs)
        self.cv.notify_all()

    def stats(self):
        el = time.time() - self.t0
        return f"{self.ncalls} batched calls, {self.nstruct / max(self.ncalls, 1):.1f} structures per call, {1000 * self.t_model / max(self.ncalls, 1):.0f} ms per call, model {100 * self.t_model / max(el, 1e-9):.0f}% of the wall time"


class GenericEvaluator:
    """one calculator instance per slot (as one per lambda process), structures evaluated one after the other; factory() -> a new ASE calculator"""

    def __init__(self, factory):
        self.factory, self.calcs = factory, {}

    def __call__(self, slots, reqs):
        out = []
        for s, (structs, stress) in zip(slots, reqs):
            calc = self.calcs.get(s)
            if calc is None:
                calc = self.calcs[s] = self.factory()
            res = []
            for at in structs:
                at.calc = calc
                e = float(at.get_potential_energy()); f = np.array(at.get_forces())
                res.append((e, f, np.array(at.get_stress(voigt=True)) if stress else None))
                at.calc = None
            out.append(res)
        return out


def _nl_job(job):
    """neighbour list of one structure in a worker process (matscipy holds the GIL: ~45 ms for 250 H atoms at 6 A, so the lists of a batch are built in parallel)"""
    from matscipy.neighbours import neighbour_list as nl
    at, cutoff = job
    return nl("ijD", at, cutoff=cutoff)


class GraceBatch:
    """many structures in one tensorpotential model call (see the module docstring). calc: the TPCalculator of the run (its models, data builders and keys are used,
    so neighbour lists, cutoffs and element maps are exactly the calculator's). max_atoms > 0 splits a call into pieces of at most that many atoms (GPU memory)."""

    PAD_FRAC = 0.05   # neighbour padding margin, as TPCalculator's default pad_neighbors_fraction: shapes are reused, XLA recompiles rarely

    def __init__(self, calc, max_atoms=0, nl_workers=None):
        from tensorpotential import constants as C
        self.C, self.calc, self.max_atoms = C, calc, int(max_atoms or 0)
        try:
            ncpu = len(os.sched_getaffinity(0))
        except AttributeError:
            ncpu = os.cpu_count() or 1
        self.nl_workers = int(os.environ.get("H3O_NL_WORKERS", min(16, ncpu - 1))) if nl_workers is None else int(nl_workers); self.pool = None
        if getattr(calc, "dense_reshape", False):
            raise NotImplementedError("GraceBatch: the calculator runs the dense ('uniform') engine; create it in the default segment_sum mode")
        self.models = list(getattr(calc, "models", None) or [calc.model])
        self.builders = list(calc.data_builders); self.keys = set(calc.data_keys)
        self.bounds = []; self.fast_ok = {}; self.t_extract = 0.0

    # ---- data
    @staticmethod
    def _cutoff(b, at):
        """the cutoff argument GeometricalDataBuilder passes to the neighbour list"""
        if getattr(b, "cutoff_dict", None) is not None:
            from itertools import combinations_with_replacement
            return {(e1, e2): b.cutoff_dict.get((e1, e2), b.cutoff) for e1, e2 in combinations_with_replacement(set(at.get_chemical_symbols()), 2)}
        return b.cutoff

    def _extract_fast(self, b, at, nlres=None):
        """GeometricalDataBuilder.extract_from_ase_atoms without its per-bond python loops (same neighbour list call, same arrays); checked against it once per system size.
        nlres: (i, j, D) already computed (worker pool)"""
        C = self.C
        ind_i, ind_j, bv = nlres if nlres is not None else _nl_job((at, self._cutoff(b, at)))
        mu = np.array([b.elements_map[s] for s in at.get_chemical_symbols()])
        n = len(at); has = np.zeros(n, bool); has[ind_i] = True
        if not has.all():   # atoms without neighbours: a fictitious neighbour beyond the cutoff (as the builder does)
            miss = np.where(~has)[0]
            add = (np.dot(at.cell, np.array([[1, 1, 1] for _ in miss]).reshape(3, -1)).reshape(-1, 3) + b.max_cutoff)
            ind_i = np.append(ind_i, miss); ind_j = np.append(ind_j, np.zeros(len(miss), int)); bv = np.append(bv, add, axis=0)
            o = np.argsort(ind_i); ind_i, ind_j, bv = ind_i[o], ind_j[o], bv[o]
        return {C.ATOMIC_MU_I: mu.astype(np.int32), C.BOND_VECTOR: bv.astype(b.float_dtype), C.BOND_MU_I: mu[ind_i].astype(np.int32), C.BOND_MU_J: mu[ind_j].astype(np.int32),
                C.BOND_IND_I: ind_i.astype(np.int32), C.BOND_IND_J: ind_j.astype(np.int32), C.N_ATOMS_BATCH_REAL: np.array(n).astype(np.int32),
                C.N_STRUCTURES_BATCH_REAL: np.array(1).astype(np.int32), C.N_NEIGHBORS_REAL: np.array(len(bv)).astype(np.int32)}

    def _geo(self):
        return next((b for b in self.builders if type(b).__name__ == "GeometricalDataBuilder"), None)

    def _neighbours(self, structs):
        """neighbour lists of all structures, in parallel worker processes when there are several (None entries: computed later in _extract)"""
        b = self._geo()
        if b is None or self.nl_workers < 1 or len(structs) < 2 or not all(all(at.pbc) and self.fast_ok.get(len(at), True) for at in structs):
            return [None] * len(structs)
        if self.pool is None:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            self.pool = ProcessPoolExecutor(self.nl_workers, mp_context=mp.get_context("spawn"))   # spawn: never fork a process that runs TensorFlow
            print(f"GraceBatch: neighbour lists in {self.nl_workers} worker processes", flush=True)
        jobs = []
        for at in structs:
            x = at.copy(); x.calc = None; x.set_constraint(); jobs.append((x, self._cutoff(b, x)))
        return list(self.pool.map(_nl_job, jobs))

    def _extract(self, at, nlres=None):
        d = {}
        for b in self.builders:
            if type(b).__name__ == "GeometricalDataBuilder" and self.fast_ok.get(len(at), True) and all(at.pbc):
                fast = self._extract_fast(b, at, nlres)
                if len(at) not in self.fast_ok:   # first structure of this size: must equal the builder's own output exactly, else the builder is used from now on
                    ref = b.extract_from_ase_atoms(at.copy())
                    self.fast_ok[len(at)] = set(ref) == set(fast) and all(np.array_equal(np.asarray(ref[k]), np.asarray(fast[k])) and np.asarray(ref[k]).dtype == np.asarray(fast[k]).dtype for k in ref)
                    if not self.fast_ok[len(at)]:
                        print(f"GraceBatch: fast data extraction differs from the builder's for {len(at)} atoms: using the builder (slower)", flush=True); fast = ref
                d.update(fast)
            else:
                d.update(b.extract_from_ase_atoms(at.copy()))
        return d

    def _bound(self, nat, nnb):
        """(atoms, neighbours) padded shape: reuse a known one with the same atom count when it fits without more than 3x the margin of padding, else a new one"""
        ok = [bd for bd in self.bounds if bd[0] == nat + 1 and nnb <= bd[1] <= nnb * (1 + 3 * self.PAD_FRAC) + 16]
        if ok:
            return min(ok)
        bd = (nat + 1, int(np.ceil(nnb * (1 + self.PAD_FRAC))) + 1); self.bounds.append(bd)
        return bd

    def _eval(self, structs, want_stress):
        import tensorflow as tf
        C = self.C; t = time.time()
        pre = [self._extract(at, r) for at, r in zip(structs, self._neighbours(structs))]; self.t_extract += time.time() - t
        batch = {}
        for b in self.builders:
            batch.update(b.join_to_batch(pre))
        nat = int(sum(len(at) for at in structs)); nb = [len(p[C.BOND_VECTOR]) for p in pre]; nnb = int(sum(nb))
        pa, pn = self._bound(nat, nnb)
        for b in self.builders:
            b.pad_batch(batch, {C.PAD_MAX_N_STRUCTURES: 1, C.PAD_MAX_N_ATOMS: pa, C.PAD_MAX_N_NEIGHBORS: pn})
        data = tf.data.Dataset.from_tensors({k: v for k, v in batch.items() if k in self.keys}).get_single_element()
        ea, F, PF = [], [], []
        for m in self.models:
            o = m.compute(data)
            ea.append(np.asarray(o[C.PREDICT_ATOMIC_ENERGY]).reshape(-1)[:nat].astype(np.float64)); F.append(np.asarray(o[C.PREDICT_FORCES])[:nat])
            pf = o.get("z_" + getattr(C, "PREDICT_PAIR_FORCES", "pair_forces")) if hasattr(o, "get") else None
            PF.append(None if pf is None else np.asarray(pf)[:nnb])
        ea, F = np.mean(ea, axis=0), np.mean(F, axis=0)
        pf = None if any(p is None for p in PF) else np.mean(PF, axis=0)
        out, ia, ib = [], 0, 0
        for at, p, n in zip(structs, pre, nb):
            e = float(ea[ia:ia + len(at)].sum()); f = F[ia:ia + len(at)].astype(np.float64)
            for cn in at.constraints:   # as Atoms.get_forces() of the single-structure path (FixGroupCom of the crystals)
                cn.adjust_forces(at, f)
            s = None
            if pf is not None:
                bv = np.asarray(p[C.BOND_VECTOR], np.float64); q = pf[ib:ib + n]
                vir = np.array([*(q * bv).sum(0), (q[:, 1] * bv[:, 0]).sum(), (q[:, 2] * bv[:, 0]).sum(), (q[:, 2] * bv[:, 1]).sum()])   # xx yy zz xy xz yz, as the model
                s = -vir[[0, 1, 2, 5, 4, 3]] / at.get_volume()   # ASE Voigt order, the TPCalculator's convention
            elif want_stress:
                c = at.copy(); c.calc = self.calc; s = np.array(c.get_stress(voigt=True))   # this model version gives no pair forces: single call for the stress
            out.append((e, f, s)); ia += len(at); ib += n
        return out

    def __call__(self, slots, reqs):
        while True:
            try:
                return self._call(reqs)
            except Exception as ex:   # GPU out of memory: split the call into smaller pieces (each new piece size costs one XLA compile)
                if "ResourceExhausted" not in type(ex).__name__ and "out of memory" not in str(ex).lower():
                    raise
                nat = sum(len(at) for structs, _ in reqs for at in structs); big = max(len(at) for structs, _ in reqs for at in structs)
                new = max((self.max_atoms or nat) // 2, big)
                if new >= (self.max_atoms or nat):
                    raise
                print(f"GraceBatch: GPU out of memory with {self.max_atoms or nat} atoms per call: now at most {new} (set --batch-max-atoms / BATCHMAX to start there)", flush=True)
                self.max_atoms = new; self.bounds = []

    def _call(self, reqs):
        flat = [(i, at, st) for i, (structs, st) in enumerate(reqs) for at in structs]
        res = []
        if self.max_atoms:
            piece, na = [], 0
            for x in flat:
                if piece and na + len(x[1]) > self.max_atoms:
                    res += self._eval([y[1] for y in piece], any(y[2] for y in piece)); piece, na = [], 0
                piece.append(x); na += len(x[1])
            if piece:
                res += self._eval([y[1] for y in piece], any(y[2] for y in piece))
        else:
            res = self._eval([x[1] for x in flat], any(x[2] for x in flat))
        out = [[] for _ in reqs]
        for (i, _, _), r in zip(flat, res):
            out[i].append(r)
        return out

    def selfcheck(self, structs, tol_e=1e-4, tol_f=1e-4, tol_s=1e-6):
        """batched vs single-structure TPCalculator on the same structures: max |dE| per atom (eV), |dF| (eV/A), |dstress| (eV/A^3). Raises if above tolerance."""
        b = self(None, [(structs, True)])[0]; worst = [0.0, 0.0, 0.0]   # through __call__: GPU out of memory splits the call instead of crashing
        for at, (e, f, s) in zip(structs, b):
            c = at.copy(); c.calc = self.calc
            e1, f1, s1 = c.get_potential_energy(), c.get_forces(), c.get_stress(voigt=True)
            worst[0] = max(worst[0], abs(e - e1) / len(at)); worst[1] = max(worst[1], float(np.abs(f - f1).max())); worst[2] = max(worst[2], float(np.abs(s - s1).max()) if s is not None else 0.0)
        msg = f"GraceBatch self-check on {len(structs)} structures: max |dE| {worst[0]:.2e} eV/atom, |dF| {worst[1]:.2e} eV/A, |dstress| {worst[2]:.2e} eV/A^3 (batched vs single calls)"
        if worst[0] > tol_e or worst[1] > tol_f or worst[2] > tol_s:
            raise SystemExit(msg + ": ABOVE TOLERANCE, the batched engine is not usable with this model / tensorpotential version (use the per-lambda engine)")
        return msg


class SlotModel:
    """the model as seen by one lambda thread: full system (+ host alone for insertion) evaluated together through the Batcher, cached per positions so that the
    host component and the MLP pressure of the same step reuse that evaluation"""

    def __init__(self, batcher, slot, host_idx=None):
        self.b, self.slot, self.host = batcher, slot, (None if host_idx is None else np.asarray(host_idx))
        self.key = None; self.res = None

    def get(self, atoms, stress=False):
        key = (atoms.positions.tobytes(), np.asarray(atoms.cell).tobytes(), atoms.numbers.tobytes())
        if self.key != key or (stress and self.res[0][2] is None):
            structs = [atoms.copy()] + ([atoms[self.host]] if self.host is not None else [])   # as full_component / subset_component (constraints kept)
            for x in structs:
                x.calc = None
            self.res = self.b.evaluate(self.slot, structs, stress); self.key = key
        return self.res

    def full(self, atoms):   # = engine.full_component(calc)
        e, f, _ = self.get(atoms)[0]
        return float(e), np.array(f)

    def host_part(self, atoms):   # = engine.subset_component(calc, host_idx)
        e, f, _ = self.get(atoms)[1]
        F = np.zeros((len(atoms), 3)); F[self.host] = f
        return float(e), F


class SlotCalc(Calculator):
    """ASE calculator for the MLP pressure of the thermo log (run_real.mlp_pressure_fn): energy, forces and stress of the full system from SlotModel"""
    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__(self, sm):
        super().__init__(); self.sm = sm

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f, s = self.sm.get(self.atoms, stress="stress" in properties)[0]
        self.results = dict(energy=e, free_energy=e, forces=np.array(f))
        if s is not None:
            self.results["stress"] = np.array(s)


class _ThreadOut:
    """sys.stdout replacement: each lambda thread writes to its own log file (as the separate processes did), everything else to the original stream"""

    def __init__(self, default):
        self.default, self.files = default, {}

    def _f(self):
        return self.files.get(threading.get_ident(), self.default)

    def write(self, s):
        return self._f().write(s)

    def flush(self):
        self._f().flush()

    def __getattr__(self, k):
        return getattr(self.default, k)


def make_evaluator(c, max_atoms=0):
    """evaluator for the run's model: GraceBatch for a tensorpotential calculator, GenericEvaluator otherwise"""
    calc = c["calc"]
    if type(calc).__name__ == "TPCalculator" and not os.environ.get("H3O_BATCH_GENERIC"):
        return GraceBatch(calc, max_atoms)
    from importlib import import_module
    spec, dev = c["a"].model, c["a"].device
    return GenericEvaluator(lambda: import_module("04_benchmark").get_calc_spec(spec, dev))


def run_batched(c, st, ks, run_node, status_s=60):
    """every lambda value in ks through run_node(c_k, st, k) in its own thread, model calls batched. Logs: <tag>_node<k>.log per lambda (appended, as before);
    this process's stdout gets the batch status. Returns the exit code (0, or the largest code of a failed lambda value)."""
    a = c["a"]; tag = c["tag"]
    ev = make_evaluator(c, getattr(a, "batch_max_atoms", 0))
    if isinstance(ev, GraceBatch):   # the batched call must reproduce the single-structure calculator before any MD
        s0 = c["at0"].copy(); rng = np.random.default_rng(1); s1 = s0.copy(); s1.positions += rng.normal(0, 0.05, s1.positions.shape)
        test = [s0, s1] + ([s0[c["host_idx"]]] if c.get("host_idx") is not None else [])
        print(ev.selfcheck(test), flush=True)
    bt = Batcher(ev)
    out = _ThreadOut(sys.stdout); sys.stdout = out
    codes = {}

    def work(k):
        fh = open(os.path.join(a.outdir, f"{tag}_node{k}.log"), "a", buffering=1); out.files[threading.get_ident()] = fh
        try:
            sm = SlotModel(bt, k, c.get("host_idx"))
            ck = dict(c, full=sm.full, calc=SlotCalc(sm))
            if c.get("host_idx") is not None:
                ck["E_fn"] = lambda kE: sm.host_part
            print(f"== batched engine: lambda {k} in process {os.getpid()} with {len(ks)} lambda values, one model call per step for all", flush=True)
            run_node(ck, st, k); codes[k] = 0
        except SystemExit as ex:
            codes[k] = ex.code if isinstance(ex.code, int) else 1
        except BaseException:
            traceback.print_exc(file=fh); codes[k] = 1
        finally:
            bt.unregister(k); fh.flush(); out.files.pop(threading.get_ident(), None); fh.close()

    for k in ks:
        bt.register(k)
    th = [threading.Thread(target=work, args=(k,), daemon=True, name=f"lambda{k}") for k in ks]
    for t in th:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in th):
        for t in th:
            t.join(timeout=status_s / len(th))
        if time.time() - t0 >= status_s:
            t0 = time.time(); print(f"-- {time.strftime('%H:%M:%S')} batched engine: {sum(t.is_alive() for t in th)} of {len(th)} lambda running; {bt.stats()}", flush=True)
    sys.stdout = out.default
    print(f"batched engine done: {bt.stats()}; exit codes {codes}", flush=True)
    return max(codes.values()) if codes else 0
