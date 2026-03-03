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


def integrated_autocorrelation_time(chain: np.ndarray, max_lag: int | None = None) -> float:
    """
    Compute the integrated autocorrelation time (IAT) for a 1D MCMC chain.

    Parameters
    ----------
    chain : np.ndarray
        1D array of MCMC samples.
    max_lag : int | None
        Maximum lag to sum over. If None, uses min(100, n-1).

    Returns
    -------
    float
        Integrated autocorrelation time.
    """
    acf = autocorrelation_function(chain, max_lag=max_lag)
    # IAT is 1 + 2 * sum_{k=1}^{max_lag} acf_k
    iat = 1.0 + 2.0 * np.nansum(acf[1:])
    return float(iat)

def gelman_rubin_rhat(chains: np.ndarray) -> float:
    """
    Compute the Gelman-Rubin R-hat statistic for multiple MCMC chains.

    Parameters
    ----------
    chains : np.ndarray
        2D array of shape (n_chains, n_samples) containing MCMC samples.

    Returns
    -------
    float
        R-hat statistic.
    """
    chains = np.asarray(chains, dtype=float)
    n_chains, n_samples = chains.shape

    # Mean and variance for each chain
    chain_means = np.mean(chains, axis=1)
    chain_vars = np.var(chains, axis=1, ddof=1)

    # Overall mean
    overall_mean = np.mean(chain_means)

    # Between-chain variance
    B = n_samples * np.var(chain_means, ddof=1)

    # Within-chain variance
    W = np.mean(chain_vars)

    # Estimate of marginal posterior variance
    var_hat = ((n_samples - 1) / n_samples) * W + (1 / n_samples) * B

    # R-hat statistic
    rhat = np.sqrt(var_hat / W)
    return float(rhat)

def cumulative_gelman_rubin_rhat(chains: np.ndarray, min_samples: int = 20) -> np.ndarray:
    """
    Compute the cumulative Gelman-Rubin R-hat statistic for multiple chains.

    Parameters
    ----------
    chains : np.ndarray
        2D array of shape (n_chains, n_samples) containing MCMC samples.
    min_samples : int
        Minimum number of samples to start computing R-hat.

    Returns
    -------
    np.ndarray
        1D array of cumulative R-hat values for each sample index >= min_samples.
    """
    chains = np.asarray(chains, dtype=float)
    n_chains, n_samples = chains.shape
    rhat_values = np.full(n_samples, np.nan)
    for t in range(min_samples, n_samples + 1):
        rhat_values[t - 1] = gelman_rubin_rhat(chains[:, :t])
    return rhat_values


@dataclass
class SingleChainDiagnostic:
    """
    Data class to hold diagnostics for a single MCMC chain.
    """
    acf: np.ndarray
    geweke_z: float
    iat: float
    ess: float
    mcse: float

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
    iat = integrated_autocorrelation_time(chain)
    ess = len(chain) / iat
    mcse = np.std(chain, ddof=1) / np.sqrt(ess) if ess > 0 else np.nan
    return SingleChainDiagnostic(acf=acf, geweke_z=geweke_z, iat=iat, ess=ess, mcse=mcse)

@dataclass
class MultiChainDiagnostics:
    """
    Diagnostics for multiple MCMC chains.
    """
    gelman_rubin: float
    cum_gelman_rubin: np.ndarray

def diagnose_multiple_chains(chains: list[np.ndarray]) -> MultiChainDiagnostics:
    """
    Compute diagnostics for multiple MCMC chains.

    Parameters
    ----------
    chains : list[np.ndarray]
        List of 1D arrays, each representing an MCMC chain.

    Returns
    -------
    MultiChainDiagnostics
        Object containing diagnostics for multiple chains.
    """

    # Stack chains into a 2D array (n_chains x n_samples)
    stacked_chains = np.stack(chains, axis=0)
    gelman_rubin = gelman_rubin_rhat(stacked_chains)
    cum_gelman_rubin = cumulative_gelman_rubin_rhat(stacked_chains)
    
    return MultiChainDiagnostics(gelman_rubin=gelman_rubin, cum_gelman_rubin=cum_gelman_rubin)