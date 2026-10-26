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
