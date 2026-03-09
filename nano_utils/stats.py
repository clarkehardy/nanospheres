"""
Statistical models for gas collision spectral fitting.

Provides two iminuit-compatible NLL cost functions:

  GasCollisionNLL       — single-dataset Poisson / negative-binomial fit
  JointGasCollisionNLL  — simultaneous fit across multiple datasets

Both support constrained nuisance parameters (impulse-axis scale ``a`` and
offset ``b``, sphere diameter ``d``) and an optional fractional systematic
``f_sys`` that broadens the likelihood from Poisson to negative binomial.
"""

import numpy as np
from scipy.special import gammaln
from scipy import stats as scipy_stats

from signal_models import gas_collision_spectrum


# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def half_gaus(x, A, sigma):
    """Half-Gaussian background model.

    :param x: Impulse values [keV/c].
    :type x: array_like
    :param A: Amplitude.
    :type A: float
    :param sigma: Width [keV/c].
    :type sigma: float
    :returns: Half-Gaussian evaluated at *x*.
    :rtype: numpy.ndarray
    """
    return A * np.exp(-x**2 / (2 * sigma**2)) / sigma


def signal_plus_bkg(bins, pressure_mbar, sphere_temperature_K, sigma, alpha,
                    A_bkg, d_nm, gas='xenon'):
    """Gas collision spectrum plus half-Gaussian background.

    :param bins: Impulse bin centres [keV/c].
    :type bins: array_like
    :param pressure_mbar: Gas pressure [mbar].
    :type pressure_mbar: float
    :param sphere_temperature_K: Sphere surface temperature [K].
    :type sphere_temperature_K: float
    :param sigma: Impulse resolution (Gaussian sigma) [keV/c].
    :type sigma: float
    :param alpha: Accommodation coefficient (0 = specular, 1 = diffuse).
    :type alpha: float
    :param A_bkg: Amplitude of the half-Gaussian background.
    :type A_bkg: float
    :param d_nm: Sphere diameter [nm].
    :type d_nm: float
    :param gas: Gas species identifier (default ``'xenon'``).
    :type gas: str
    :returns: Differential rate [counts/(50 keV/c)/s].
    :rtype: numpy.ndarray
    """
    signal = gas_collision_spectrum(
        bins, pressure_mbar, sigma,
        temperature_K=sphere_temperature_K,
        diameter_nm=d_nm,
        alpha=alpha,
        gas=gas,
    )
    background = half_gaus(bins, A_bkg, sigma)
    return signal + background


# ---------------------------------------------------------------------------
# Private likelihood helpers
# ---------------------------------------------------------------------------

def _nll_terms(n, mu, f_sys):
    """Sum of per-bin NLL terms (Poisson or negative binomial).

    :param n: Observed counts per bin.
    :type n: numpy.ndarray
    :param mu: Expected counts per bin (must be > 0).
    :type mu: numpy.ndarray
    :param f_sys: Fractional systematic uncertainty.  0 gives pure Poisson;
        positive values give negative binomial with ``r = 1 / f_sys**2``.
    :type f_sys: float
    :returns: Scalar NLL (dropping terms constant in the parameters).
    :rtype: float
    """
    if f_sys > 0:
        r = 1. / f_sys**2
        return float(np.sum(
            -gammaln(n + r) + gammaln(r)
            + (n + r) * np.log(r + mu) - r * np.log(r) - n * np.log(mu)
        ))
    return float(np.sum(mu - n * np.log(mu)))


def _deviance_terms(n, mu, f_sys):
    """Per-bin likelihood-ratio deviance terms.

    Returns an array of non-negative values whose sum is the G-statistic
    (generalised to negative binomial when *f_sys* > 0).  Under the null
    hypothesis (model correct), ``sum(_deviance_terms(...))`` is
    approximately chi-squared distributed.

    Uses ``n_safe = max(n, 1)`` inside logarithms to suppress
    ``RuntimeWarning`` from ``0 * log(0)`` in the ``n = 0`` branch;
    ``np.where`` selects the correct branch so results are unaffected.

    :param n: Observed counts per bin.
    :type n: numpy.ndarray
    :param mu: Expected counts per bin.
    :type mu: numpy.ndarray
    :param f_sys: Fractional systematic uncertainty.
    :type f_sys: float
    :returns: Per-bin deviance contributions.
    :rtype: numpy.ndarray
    """
    n_safe = np.where(n > 0, n, 1.)
    if f_sys > 0:
        r = 1. / f_sys**2
        return np.where(
            n > 0,
            2 * (r * np.log((r + mu) / (r + n_safe))
                 + n_safe * np.log(n_safe * (r + mu) / (mu * (r + n_safe)))),
            2 * r * np.log(1. + mu / r),
        )
    return np.where(n > 0, 2 * (mu - n + n_safe * np.log(n_safe / mu)), 2 * mu)


# ---------------------------------------------------------------------------
# Single-dataset NLL
# ---------------------------------------------------------------------------

class GasCollisionNLL:
    """Poisson / negative-binomial NLL for a single gas-collision dataset.

    Parameters visible to iminuit (via ``__call__`` signature):

    * ``pressure_mbar``        — gas pressure [mbar]
    * ``sphere_temperature_K`` — sphere surface temperature [K]
    * ``sigma``                — impulse resolution [keV/c]
    * ``alpha``                — accommodation coefficient [0, 1]
    * ``A_bkg``                — background amplitude
    * ``d``                    — sphere diameter [nm]  (Gaussian-constrained)
    * ``a``                    — impulse-axis scale factor (Gaussian-constrained, mean 1)
    * ``b``                    — impulse-axis offset [keV/c] (Gaussian-constrained, mean 0)

    :param bins: Bin centres of the fitted region [keV/c].
    :type bins: array_like
    :param counts: Observed counts in each bin.
    :type counts: array_like
    :param scale_factor: Converts model rate to expected counts
        (``mu = model_rate / scale_factor``).
    :type scale_factor: float
    :param gas: Gas species identifier passed to :func:`signal_plus_bkg`.
    :type gas: str
    :param d_nominal: Nominal sphere diameter [nm] (centre of Gaussian prior on ``d``).
    :type d_nominal: float
    :param sigma_a: 1-sigma width of Gaussian prior on ``a``.
    :type sigma_a: float
    :param sigma_b: 1-sigma width of Gaussian prior on ``b`` [keV/c].
    :type sigma_b: float
    :param sigma_d: 1-sigma width of Gaussian prior on ``d`` [nm].
    :type sigma_d: float
    :param f_sys: Fractional systematic per bin (0 = pure Poisson).
    :type f_sys: float
    """

    def __init__(self, bins, counts, scale_factor, gas, d_nominal,
                 sigma_a, sigma_b, sigma_d, f_sys=0.):
        self.bins         = np.asarray(bins)
        self.counts       = np.asarray(counts)
        self.scale_factor = scale_factor
        self.gas          = gas
        self.d_nominal    = d_nominal
        self.sigma_a      = sigma_a
        self.sigma_b      = sigma_b
        self.sigma_d      = sigma_d
        self.f_sys        = f_sys

    def __call__(self, pressure_mbar, sphere_temperature_K, sigma, alpha,
                 A_bkg, d, a, b):
        distorted_bins = a * self.bins + b
        if np.isnan(distorted_bins).any():
            return np.inf
        model_rate = signal_plus_bkg(
            distorted_bins, pressure_mbar, sphere_temperature_K,
            sigma, alpha, A_bkg, d, self.gas,
        )
        mu = np.maximum(model_rate / self.scale_factor, 1e-10)
        nll  = _nll_terms(self.counts, mu, self.f_sys)
        nll += (a - 1.)**2 / (2. * self.sigma_a**2)
        nll += b**2        / (2. * self.sigma_b**2)
        nll += (d - self.d_nominal)**2 / (2. * self.sigma_d**2)
        return nll

    def goodness_of_fit(self, minuit):
        """Likelihood-ratio goodness-of-fit statistic (deviance / G-test).

        Under the null hypothesis (model correct), the statistic is
        approximately chi-squared with ``N_bins - N_free`` degrees of freedom.

        :param minuit: Fitted Minuit object.
        :type minuit: iminuit.Minuit
        :returns: ``(g_stat, ndof, p_value)``
        :rtype: tuple[float, int, float]
        """
        vals = dict(zip(minuit.parameters, minuit.values))
        distorted_bins = vals['a'] * self.bins + vals['b']
        model_rate = signal_plus_bkg(
            distorted_bins,
            vals['pressure_mbar'], vals['sphere_temperature_K'],
            vals['sigma'], vals['alpha'], vals['A_bkg'],
            vals['d'], self.gas,
        )
        mu      = np.maximum(model_rate / self.scale_factor, 1e-10)
        g_stat  = float(np.sum(_deviance_terms(self.counts, mu, self.f_sys)))
        ndof    = len(self.bins) - minuit.nfit
        p_value = float(1 - scipy_stats.chi2.cdf(g_stat, ndof))
        return g_stat, ndof, p_value


# ---------------------------------------------------------------------------
# Joint NLL
# ---------------------------------------------------------------------------

class JointGasCollisionNLL:
    """Joint Poisson / negative-binomial NLL across multiple datasets.

    Shared parameters (one per fit):
        ``sphere_temperature_K``, ``alpha``, ``d``, ``A_bkg``

    Per-dataset parameters (one per dataset, indexed 0 … N-1):
        ``pressure_mbar_{i}``, ``sigma_{i}``, ``a_{i}``, ``b_{i}``

    The sphere-diameter Gaussian constraint is applied once (shared).
    Nuisance constraints on ``a`` and ``b`` are applied independently
    per dataset.

    Parameter layout in ``__call__(*args)``::

        args[0]             sphere_temperature_K
        args[1]             alpha
        args[2]             d
        args[3]             A_bkg
        args[4 : 4+N]       pressure_mbar_0 … pressure_mbar_{N-1}
        args[4+N : 4+2N]    sigma_0 … sigma_{N-1}
        args[4+2N : 4+3N]   a_0 … a_{N-1}
        args[4+3N : 4+4N]   b_0 … b_{N-1}

    Use the :attr:`param_names` property to obtain the canonical parameter
    name list for Minuit instantiation.

    :param datasets: List of dataset dicts, each with keys
        ``'bins'`` (array), ``'counts'`` (array), ``'scale_factor'`` (float).
    :type datasets: list[dict]
    :param gas: Gas species identifier.
    :type gas: str
    :param d_nominal: Nominal sphere diameter [nm].
    :type d_nominal: float
    :param sigma_a: 1-sigma width of Gaussian prior on each ``a_i``.
    :type sigma_a: float
    :param sigma_b: 1-sigma width of Gaussian prior on each ``b_i`` [keV/c].
    :type sigma_b: float
    :param sigma_d: 1-sigma width of Gaussian prior on ``d`` [nm].
    :type sigma_d: float
    :param f_sys: Fractional systematic per bin (0 = pure Poisson).
    :type f_sys: float
    """

    def __init__(self, datasets, gas, d_nominal,
                 sigma_a, sigma_b, sigma_d, f_sys=0.):
        self.data      = datasets
        self.n         = len(datasets)
        self.gas       = gas
        self.d_nominal = d_nominal
        self.sigma_a   = sigma_a
        self.sigma_b   = sigma_b
        self.sigma_d   = sigma_d
        self.f_sys     = f_sys

    @property
    def param_names(self):
        """Canonical iminuit parameter name list for this joint fit.

        :rtype: list[str]
        """
        n = self.n
        return (
            ['sphere_temperature_K', 'alpha', 'd', 'A_bkg']
            + [f'pressure_mbar_{i}' for i in range(n)]
            + [f'sigma_{i}'         for i in range(n)]
            + [f'a_{i}'             for i in range(n)]
            + [f'b_{i}'             for i in range(n)]
        )

    def __call__(self, *args):
        T_sph     = args[0]
        alpha     = args[1]
        d         = args[2]
        A_bkg     = args[3]
        n         = self.n
        pressures = args[4      : 4 +   n]
        sigmas    = args[4 +   n : 4 + 2*n]
        a_vals    = args[4 + 2*n : 4 + 3*n]
        b_vals    = args[4 + 3*n : 4 + 4*n]

        # Shared sphere-diameter constraint — applied once across all datasets
        nll = (d - self.d_nominal)**2 / (2. * self.sigma_d**2)

        for i, ds in enumerate(self.data):
            a = a_vals[i]
            b = b_vals[i]
            distorted_bins = a * ds['bins'] + b
            if np.isnan(distorted_bins).any():
                return np.inf
            model_rate = signal_plus_bkg(
                distorted_bins, pressures[i], T_sph,
                sigmas[i], alpha, A_bkg, d, self.gas,
            )
            mu  = np.maximum(model_rate / ds['scale_factor'], 1e-10)
            nll += _nll_terms(ds['counts'], mu, self.f_sys)
            nll += (a - 1.)**2 / (2. * self.sigma_a**2)
            nll += b**2        / (2. * self.sigma_b**2)

        return nll

    def goodness_of_fit(self, minuit):
        """Summed likelihood-ratio deviance across all datasets.

        :param minuit: Fitted Minuit object.
        :type minuit: iminuit.Minuit
        :returns: ``(g_stat, ndof, p_value)``
        :rtype: tuple[float, int, float]
        """
        vals  = dict(zip(minuit.parameters, minuit.values))
        T_sph = vals['sphere_temperature_K']
        alpha = vals['alpha']
        d     = vals['d']
        A_bkg = vals['A_bkg']

        g_total = 0.
        for i, ds in enumerate(self.data):
            distorted_bins = vals[f'a_{i}'] * ds['bins'] + vals[f'b_{i}']
            model_rate = signal_plus_bkg(
                distorted_bins, vals[f'pressure_mbar_{i}'],
                T_sph, vals[f'sigma_{i}'], alpha, A_bkg, d, self.gas,
            )
            mu       = np.maximum(model_rate / ds['scale_factor'], 1e-10)
            g_total += float(np.sum(_deviance_terms(ds['counts'], mu, self.f_sys)))

        total_bins = sum(len(ds['counts']) for ds in self.data)
        ndof       = total_bins - minuit.nfit
        p_value    = float(1 - scipy_stats.chi2.cdf(g_total, ndof))
        return g_total, ndof, p_value
