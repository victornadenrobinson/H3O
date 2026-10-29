import numpy as np


def gauss_legendre(n, power=1.0):
    """nodes lam_i in (0,1) and weights w_i with  int_0^1 g(lam) dlam ~ sum w_i g(lam_i).
    power > 1 maps lam = x^power (x Gauss-Legendre on [0,1]) which concentrates nodes at small lambda, where the integrand is steep."""
    x, w = np.polynomial.legendre.leggauss(n)
    x, w = (x + 1) / 2, w / 2
    return x ** power, w * power * x ** (power - 1)


def block_stats(series, nblocks=10, discard=0.0):
    """mean and standard error from block averages (blocks must be longer than the correlation time)."""
    s = np.asarray(series, float)
    s = s[int(len(s) * discard):]
    b = np.array([c.mean() for c in np.array_split(s, nblocks)])
    return float(b.mean()), float(b.std(ddof=1) / np.sqrt(nblocks))


def integrate(means, sems, weights):
    means, sems, weights = map(np.asarray, (means, sems, weights))
    return float(np.sum(weights * means)), float(np.sqrt(np.sum((weights * sems) ** 2)))

def gauss_kronrod(n):
    """Gauss-Kronrod rule G_n subset K_{2n+1} on [-1, 1] (Legendre weight): returns (x, w_kronrod, w_gauss) for the 2n+1 Kronrod nodes sorted ascending; w_gauss is zero at the n+1 added nodes.
    The n+1 added nodes are the zeros of the Stieltjes polynomial E_{n+1} (orthogonal to every polynomial of degree <= n with weight P_n); weights from exactness for polynomials of degree <= 3n+1 (least squares in the Legendre basis)."""
    from numpy.polynomial import legendre as L
    xg, wg = L.leggauss(n); xq, wq = L.leggauss(2 * n + 4)
    P = lambda k, x: L.legval(x, [0] * k + [1])
    Pn = P(n, xq)
    M = np.array([[np.sum(wq * Pn * P(i, xq) * P(k, xq)) for i in range(n + 1)] for k in range(n + 1)])
    rhs = -np.array([np.sum(wq * Pn * P(n + 1, xq) * P(k, xq)) for k in range(n + 1)])
    c = np.concatenate([np.linalg.solve(M, rhs), [1.0]])
    xe = np.sort(L.legroots(c).real)
    x = np.sort(np.concatenate([xg, xe]))
    A = np.array([P(j, x) for j in range(3 * n + 2)]); b = np.array([2.0 if j == 0 else 0.0 for j in range(3 * n + 2)])
    w = np.linalg.lstsq(A, b, rcond=None)[0]
    wG = np.zeros_like(x)
    for xi, wi in zip(xg, wg):
        wG[np.argmin(abs(x - xi))] = wi
    return x, w, wG


def rule_nodes(A):
    """lambda values and weights of a run from its args A (dict or stage0 'args'): rule 'gl' (default) = Gauss-Legendre with A['nodes'] values; rule 'gk' = Gauss-Kronrod: A['ngauss'] Gauss values
    (identical to the 'gl' rule with that many values, so their samples can be kept) followed by the ngauss + 1 added Kronrod values; A['nodes'] must be 2 ngauss + 1. All in lambda = x^power.
    Returns lam, w and, for 'gk', also the Gauss-only subset (lam_g, w_g) whose difference from the full rule estimates the quadrature error (None for 'gl')."""
    p = float(A["power"]); rule = A.get("rule") or "gl"
    if rule == "gl":
        lam, w = gauss_legendre(A["nodes"], p); return lam, w, None
    n = int(A["ngauss"]); assert A["nodes"] == 2 * n + 1, "rule gk needs nodes = 2 ngauss + 1"
    x, wk, wG = gauss_kronrod(n); xg = np.polynomial.legendre.leggauss(n)[0]
    isg = np.array([np.min(abs(xg - xi)) < 1e-12 for xi in x])
    order = np.concatenate([np.where(isg)[0], np.where(~isg)[0]])          # Gauss values first (same indices as the 'gl' run), then the added ones
    xs = (x[order] + 1) / 2; lam = xs ** p; jac = p * xs ** (p - 1)
    w = wk[order] / 2 * jac; wg = wG[order] / 2 * jac
    return lam, w, (lam[:n], wg[:n])
