"""Statistical models for gas collision spectral fitting.

Provides one iminuit-compatible NLL cost function:

  GasCollisionNLL — simultaneous Poisson / negative-binomial fit across
                    multiple datasets with a single shared ``impulse_scale``
                    nuisance parameter.

The model uses:

* One shared ``impulse_scale`` nuisance parameter with Gaussian prior
  N(1, σ_a²) representing a common calibration uncertainty.
* Count-conserving derived background: ``A_bkg`` is not a free parameter;
  it is derived per-dataset at each likelihood evaluation by enforcing that
  signal + background counts equal the total observed counts above the fit
  cutoff.
* A joint fit over all datasets with shared ``sphere_temperature_K``,
  ``alpha``, and ``impulse_scale`` and per-dataset ``pressure_mbar_i``,
  ``sigma_i``.

The background derivation enforces:

    integral_{q_cut}^inf  A/sigma * exp(-p^2/2/sigma^2) dp
        = N_above - integral_{q_cut}^inf  S dp

where N_above is the total observed counts above the cutoff and S is the
signal spectrum at the current parameters.  In the discrete implementation
this becomes:

    A = (N_above - N_signal_above) * scale_factor * sigma
        / sum_i exp(-scaled_bins_i^2 / (2*sigma^2))

with A clamped to >= 0.
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


def _compute_derived_A_bkg(scaled_bins_above, N_above_obs, N_signal_above,
                            sigma, scale_factor):
    """Derive A_bkg by enforcing count conservation in the fit region.

    Solves the discrete form of the count-conservation constraint:

    .. math::

        \\sum_i \\frac{A}{\\sigma}
            \\exp\\!\\left(-\\frac{(s\\,q_i)^2}{2\\sigma^2}\\right)
            \\frac{1}{\\text{scale\\_factor}}
        = N_{\\text{above}} - N_{\\text{signal,above}}

    for *A*, where *s* is ``impulse_scale`` and the sum runs over all
    above-cut bins.

    :param scaled_bins_above: Impulse-scaled bin centres in the fit region
        (``impulse_scale * q_i``) [keV/c].
    :type scaled_bins_above: numpy.ndarray
    :param N_above_obs: Total observed counts above the fit cutoff.
    :type N_above_obs: float
    :param N_signal_above: Expected signal counts above the fit cutoff from
        the current model parameters.
    :type N_signal_above: float
    :param sigma: Impulse resolution [keV/c].
    :type sigma: float
    :param scale_factor: Converts model rate to expected counts
        (``mu = rate / scale_factor``).
    :type scale_factor: float
    :returns: Derived A_bkg, clamped to >= 0.
    :rtype: float
    """
    N_bkg_above = max(float(N_above_obs) - float(N_signal_above), 0.)
    gauss_sum   = float(np.sum(np.exp(-scaled_bins_above**2 / (2. * sigma**2))))
    if gauss_sum <= 0.:
        return 0.
    return N_bkg_above * scale_factor * sigma / gauss_sum


# ---------------------------------------------------------------------------
# Joint NLL
# ---------------------------------------------------------------------------

class GasCollisionNLL:
    """Joint Poisson / negative-binomial NLL across multiple datasets.

    Shared parameters (one per fit):
        ``sphere_temperature_K``, ``alpha``, ``impulse_scale``

    Per-dataset parameters (one per dataset, indexed 0 … N-1):
        ``pressure_mbar_{i}``, ``sigma_{i}``

    ``A_bkg`` is not a free parameter; it is derived per-dataset at each
    evaluation by enforcing count conservation in each dataset's fit region.

    The sphere diameter is held fixed at *d_nominal* and is not a free
    parameter of the fit.

    Parameter layout in ``__call__(*args)``::

        args[0]             sphere_temperature_K
        args[1]             alpha
        args[2 : 2+N]       pressure_mbar_0 … pressure_mbar_{N-1}
        args[2+N : 2+2N]    sigma_0 … sigma_{N-1}
        args[2+2N]          impulse_scale

    Use the :attr:`param_names` property to obtain the canonical parameter
    name list for Minuit instantiation.

    :param datasets: List of dataset dicts, each with keys
        ``'bins'`` (array), ``'counts'`` (array), ``'scale_factor'`` (float).
    :type datasets: list[dict]
    :param gas: Gas species identifier.
    :type gas: str
    :param d_nominal: Sphere diameter [nm] used in the model (fixed).
    :type d_nominal: float
    :param sigma_a: 1-sigma width of Gaussian prior on ``impulse_scale``.
    :type sigma_a: float
    :param f_sys: Fractional systematic per bin (0 = pure Poisson).
    :type f_sys: float
    """

    def __init__(self, datasets, gas, d_nominal, sigma_a, f_sys=0.):
        self.data      = datasets
        self.n         = len(datasets)
        self.gas       = gas
        self.d_nominal = d_nominal
        self.sigma_a   = sigma_a
        self.f_sys     = f_sys

    @property
    def param_names(self):
        """Canonical iminuit parameter name list for this joint fit.

        :rtype: list[str]
        """
        n = self.n
        names  = ['sphere_temperature_K', 'alpha']
        names += [f'pressure_mbar_{i}' for i in range(n)]
        names += [f'sigma_{i}'         for i in range(n)]
        names += ['impulse_scale']
        return names

    def _get_A_bkg_i(self, ds, signal_rate_i, sigma_i, scaled_bins_i):
        """Derive A_bkg for dataset *ds* at the current parameters.

        :param ds: Dataset dict with ``'bins'``, ``'counts'``, ``'scale_factor'``.
        :param signal_rate_i: Signal model rate at *scaled_bins_i*.
        :param sigma_i: Current impulse resolution [keV/c].
        :param scaled_bins_i: ``impulse_scale * ds['bins']`` [keV/c].
        :returns: Derived A_bkg.
        :rtype: float
        """
        N_signal_above = float(np.sum(signal_rate_i / ds['scale_factor']))
        return _compute_derived_A_bkg(
            scaled_bins_i, float(np.sum(ds['counts'])),
            N_signal_above, sigma_i, ds['scale_factor'],
        )

    def get_A_bkg(self, minuit, dataset_index):
        """Return A_bkg for dataset *dataset_index* given a fitted Minuit object.

        Recomputes the derived value at the best-fit parameters.

        :param minuit: Fitted Minuit object.
        :type minuit: iminuit.Minuit
        :param dataset_index: Index into the dataset list.
        :type dataset_index: int
        :returns: Background amplitude A_bkg for that dataset.
        :rtype: float
        """
        vals    = dict(zip(minuit.parameters, minuit.values))
        i       = dataset_index
        ds      = self.data[i]
        scale_i = vals['impulse_scale']
        sigma_i = vals[f'sigma_{i}']
        scaled_bins_i = scale_i * ds['bins']
        signal_rate_i = gas_collision_spectrum(
            scaled_bins_i, vals[f'pressure_mbar_{i}'], sigma_i,
            temperature_K=vals['sphere_temperature_K'],
            diameter_nm=self.d_nominal, alpha=vals['alpha'], gas=self.gas,
        )
        return self._get_A_bkg_i(ds, signal_rate_i, sigma_i, scaled_bins_i)

    def _nll_at_scale(self, ds, pressure, sigma, T_sph, alpha, A_bkg_shared, s):
        """Per-dataset Poisson/NegBin NLL at a fixed impulse scale *s*.

        Returns ``np.inf`` on NaN inputs.

        :param ds: Dataset dict with ``'bins'``, ``'counts'``, ``'scale_factor'``.
        :param pressure: Gas pressure [mbar].
        :param sigma: Impulse resolution [keV/c].
        :param T_sph: Sphere temperature [K].
        :param alpha: Accommodation coefficient.
        :param A_bkg_shared: Ignored (A_bkg is always derived).
        :param s: Impulse scale factor.
        :returns: Scalar NLL contribution (no prior term).
        :rtype: float
        """
        scaled_bins = s * ds['bins']
        if np.isnan(scaled_bins).any():
            return np.inf
        signal_rate = gas_collision_spectrum(
            scaled_bins, pressure, sigma,
            temperature_K=T_sph, diameter_nm=self.d_nominal,
            alpha=alpha, gas=self.gas,
        )
        A_bkg      = self._get_A_bkg_i(ds, signal_rate, sigma, scaled_bins)
        model_rate = signal_rate + half_gaus(scaled_bins, A_bkg, sigma)
        mu = np.maximum(model_rate / ds['scale_factor'], 1e-10)
        return _nll_terms(ds['counts'], mu, self.f_sys)

    def __call__(self, *args):
        vals      = dict(zip(self.param_names, args))
        T_sph     = vals['sphere_temperature_K']
        alpha     = vals['alpha']
        n         = self.n
        pressures = [vals[f'pressure_mbar_{i}'] for i in range(n)]
        sigmas    = [vals[f'sigma_{i}']          for i in range(n)]

        s   = vals['impulse_scale']
        nll = (s - 1.)**2 / (2. * self.sigma_a**2)
        for i, ds in enumerate(self.data):
            nll += self._nll_at_scale(ds, pressures[i], sigmas[i], T_sph, alpha, None, s)
        return nll

    def goodness_of_fit(self, minuit):
        """Summed likelihood-ratio deviance across all datasets.

        Under the null hypothesis (model correct), the statistic is
        approximately chi-squared with ``N_bins - N_free`` degrees of freedom.

        :param minuit: Fitted Minuit object.
        :type minuit: iminuit.Minuit
        :returns: ``(g_stat, ndof, p_value)``
        :rtype: tuple[float, int, float]
        """
        vals  = dict(zip(minuit.parameters, minuit.values))
        T_sph = vals['sphere_temperature_K']
        alpha = vals['alpha']
        s     = vals['impulse_scale']

        g_total = 0.
        for i, ds in enumerate(self.data):
            sigma_i = vals[f'sigma_{i}']
            A_bkg   = self.get_A_bkg(minuit, i)

            scaled_bins = s * ds['bins']
            model_rate  = signal_plus_bkg(
                scaled_bins, vals[f'pressure_mbar_{i}'],
                T_sph, sigma_i, alpha, A_bkg,
                self.d_nominal, self.gas,
            )
            mu       = np.maximum(model_rate / ds['scale_factor'], 1e-10)
            g_total += float(np.sum(_deviance_terms(ds['counts'], mu, self.f_sys)))

        total_bins = sum(len(ds['counts']) for ds in self.data)
        ndof       = total_bins - minuit.nfit
        p_value    = float(1 - scipy_stats.chi2.cdf(g_total, ndof))
        return g_total, ndof, p_value
