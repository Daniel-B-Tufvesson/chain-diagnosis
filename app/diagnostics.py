import numpy as np
from dataclasses import dataclass

def _spectral_density_at_zero(x: np.ndarray, max_lag: int | None = None) -> float:
    """
    Estimate spectral density at frequency 0 using a Bartlett window.
    """
    n = x.size
    if n < 2:
        return np.var(x, ddof=1) # type: ignore

    x_centered = x - np.mean(x)

    if max_lag is None:
        max_lag = max(1, int(np.sqrt(n)))

    gamma0 = np.dot(x_centered, x_centered) / n
    s0 = gamma0

    for k in range(1, min(max_lag, n - 1) + 1):
        gamma_k = np.dot(x_centered[:-k], x_centered[k:]) / n
        w_k = 1.0 - (k / (max_lag + 1.0))  # Bartlett weight
        s0 += 2.0 * w_k * gamma_k

    return float(max(s0, 1e-12))


def autocorrelation_function(chain: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """
    Compute the autocorrelation function for a 1D numpy array (MCMC chain).

    Parameters
    ----------
    chain : np.ndarray
        1D array of MCMC samples.
    max_lag : int | None
        Maximum lag to compute. If None, uses min(100, n-1).

    Returns
    -------
    np.ndarray
        1D array of autocorrelations, where index is lag (0 = 1.0).
    """
    x = np.asarray(chain, dtype=float).ravel()
    n = x.size
    if n < 2:
        return np.array([1.0])
    x_centered = x - np.mean(x)
    var = np.var(x_centered)
    if var == 0:
        return np.ones(1 if max_lag is None else max_lag + 1)
    if max_lag is None:
        max_lag = min(100, n - 1)
    acf = np.empty(max_lag + 1)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            acf[lag] = np.nan
        else:
            acf[lag] = np.dot(x_centered[:-lag], x_centered[lag:]) / ((n - lag) * var)
    return acf


def geweke_diagnostic(
    chain: np.ndarray,
    first: float = 0.1,
    last: float = 0.5,
    max_lag: int | None = None,
) -> float:
    """
    Compute Geweke's z-score for a 1D MCMC chain.

    Parameters
    ----------
    chain : np.ndarray
        1D array of MCMC samples.
    first : float
        Fraction from the start of chain to use (default 0.1).
    last : float
        Fraction from the end of chain to use (default 0.5).
    max_lag : int | None
        Max lag for spectral density estimate; if None, uses sqrt(window_size).

    Returns
    -------
    float
        Geweke z-score. Values near 0 indicate better convergence.
    """
    x = np.asarray(chain, dtype=float).ravel()

    if x.size < 20:
        raise ValueError("Chain is too short for Geweke diagnostic (need at least 20 samples).")
    if not (0 < first < 1) or not (0 < last < 1):
        raise ValueError("'first' and 'last' must be in (0, 1).")
    if first + last >= 1:
        raise ValueError("'first + last' must be < 1.")

    n = x.size
    n_first = int(np.floor(first * n))
    n_last = int(np.floor(last * n))

    if n_first < 2 or n_last < 2:
        raise ValueError("Selected windows are too small. Increase chain length or adjust fractions.")

    x_first = x[:n_first]
    x_last = x[-n_last:]

    mean_first = np.mean(x_first)
    mean_last = np.mean(x_last)

    s0_first = _spectral_density_at_zero(x_first, max_lag=max_lag)
    s0_last = _spectral_density_at_zero(x_last, max_lag=max_lag)

    denom = np.sqrt((s0_first / n_first) + (s0_last / n_last))
    if denom <= 0:
        raise ValueError("Non-positive denominator in Geweke statistic computation.")

    z = (mean_first - mean_last) / denom
    return float(z)

@dataclass
class SingleChainDiagnostic:
    """
    Data class to hold diagnostics for a single MCMC chain.
    """
    acf: np.ndarray
    geweke_z: float

def diagnose_single_chain(chain: np.ndarray) -> SingleChainDiagnostic:
    """
    Compute diagnostics for a single MCMC chain.

    Parameters
    ----------
    chain : np.ndarray
        1D array of MCMC samples.
    Returns
    -------
    SingleChainDiagnostic
        Object containing diagnostic results.
    """
    acf = autocorrelation_function(chain)
    geweke_z = geweke_diagnostic(chain)
    return SingleChainDiagnostic(acf=acf, geweke_z=geweke_z)