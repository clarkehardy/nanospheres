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
                    A_bkg, d_nm, gas='xenon', shape_param=None, impulse_scale=1.0):
    """Gas collision spectrum plus half-Gaussian background.

    *bins* should be the physical (scaled) impulse positions, i.e.
    ``impulse_scale * measured_bins``.  The background is evaluated in
    measured-impulse space (``bins / impulse_scale``) so that it depends
    only on ``sigma`` and not on the calibration scale, breaking the
    ``sigma / impulse_scale`` degeneracy.

    :param bins: Physical impulse bin centres (``impulse_scale * q``) [keV/c].
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
    :param shape_param: Optional spectral-tilt exponent applied at measured
        impulse coordinates (default ``None`` = no tilt).
    :type shape_param: float or None
    :param impulse_scale: Calibration scale factor *s* (default 1).  Used
        to recover the measured-impulse bins as ``bins / impulse_scale``.
    :type impulse_scale: float
    :returns: Differential rate [counts/(50 keV/c)/s].
    :rtype: numpy.ndarray
    """
    s             = float(impulse_scale)
    measured_bins = np.asarray(bins) / s
    # Multiply signal by s (Jacobian): a measured bin of width Δq spans s·Δq
    # in physical space, so expected signal counts = s · rate · Δq/50 · T.
    # Background is already defined in measured space so carries no Jacobian.
    signal     = s * gas_collision_spectrum(
        bins, pressure_mbar, sigma,
        temperature_K=sphere_temperature_K,
        diameter_nm=d_nm,
        alpha=alpha,
        gas=gas,
    )
    background = half_gaus(measured_bins, A_bkg, sigma)
    if shape_param is not None:
        return np.exp(measured_bins * shape_param) * (signal + background)
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


def _compute_derived_A_bkg(bins_above, N_above_obs, N_signal_above,
                            sigma, scale_factor, shape_param=0.0):
    """Derive A_bkg by enforcing count conservation in the fit region.

    With a spectral-tilt exponent *shape_param* the expected counts are:

    .. math::

        \\mu_i = \\frac{e^{\\lambda q_i}\\bigl[s\\,\\text{signal}(s q_i)
                        + A/\\sigma\\,e^{-q_i^2/2\\sigma^2}\\bigr]}
                       {\\text{scale\\_factor}}

    Enforcing :math:`\\sum_i \\mu_i = N_{\\text{above}}` and solving for *A*:

    .. math::

        A = \\frac{(N_{\\text{above}} - N_{\\text{signal,above}})\\,
                   \\sigma\\,\\text{scale\\_factor}}
                  {\\sum_i e^{\\lambda q_i - q_i^2 / 2\\sigma^2}}

    where :math:`N_{\\text{signal,above}} = s\\sum_i
    e^{\\lambda q_i}\\,\\text{signal}(s q_i)/\\text{scale\\_factor}`.
    At :math:`\\lambda = 0` this reduces to the unweighted formula.

    :param bins_above: Measured-impulse bin centres in the fit region
        (``q_i``, **not** scaled) [keV/c].
    :type bins_above: numpy.ndarray
    :param N_above_obs: Total observed counts above the fit cutoff.
    :type N_above_obs: float
    :param N_signal_above: Shape-weighted, Jacobian-corrected expected signal
        counts (``s * sum(exp(λ·q) * signal_rate / scale_factor)``).
    :type N_signal_above: float
    :param sigma: Impulse resolution [keV/c].
    :type sigma: float
    :param scale_factor: Converts model rate to expected counts per bin.
    :type scale_factor: float
    :param shape_param: Spectral-tilt exponent λ (default 0).
    :type shape_param: float
    :returns: Derived A_bkg, clamped to >= 0.
    :rtype: float
    """
    N_bkg_above = max(float(N_above_obs) - float(N_signal_above), 0.)
    gauss_sum   = float(np.sum(
        np.exp(float(shape_param) * bins_above - bins_above**2 / (2. * sigma**2))
    ))
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
    :param use_pressure_eff: If ``True``, replace each ``pressure_mbar_i``
        parameter with ``pressure_eff_i = impulse_scale * pressure_mbar_i``.
        This removes the amplitude degeneracy between ``impulse_scale`` and
        pressure, leaving ``impulse_scale`` to be constrained by spectral
        shape alone.  Physical pressures are recovered via
        :meth:`get_pressures`.  Default ``False``.
    :type use_pressure_eff: bool
    :param use_N_signal: If ``True``, replace each ``pressure_mbar_i`` parameter
        with ``N_signal_i`` — the total expected (shape-weighted) signal counts in
        dataset *i*'s fit window.  This removes the amplitude degeneracy between
        ``impulse_scale`` and pressure: ``impulse_scale`` then only enters the NLL
        through the normalised spectral shape, not through the total count.
        Physical pressures are recovered via :meth:`get_pressures`.
        Mutually exclusive with ``use_pressure_eff``.  Default ``False``.
    :type use_N_signal: bool
    """

    def __init__(self, datasets, gas, d_nominal, sigma_a, f_sys=0.,
                 init_sigmas=None, use_pressure_eff=False, use_N_signal=False):
        if use_pressure_eff and use_N_signal:
            raise ValueError('use_pressure_eff and use_N_signal are mutually exclusive')
        self.data             = datasets
        self.n                = len(datasets)
        self.gas              = gas
        self.d_nominal        = d_nominal
        self.sigma_a          = sigma_a
        self.f_sys            = f_sys
        self.use_pressure_eff = use_pressure_eff
        self.use_N_signal     = use_N_signal
        if init_sigmas is None:
            self.init_sigmas = 50 * np.ones(self.n)
        else:
            self.init_sigmas = init_sigmas

    @property
    def param_names(self):
        """Canonical iminuit parameter name list for this joint fit.

        :rtype: list[str]
        """
        n = self.n
        names  = ['sphere_temperature_K', 'alpha']
        if self.use_N_signal:
            names += [f'N_signal_{i}' for i in range(n)]
        elif self.use_pressure_eff:
            names += [f'pressure_eff_{i}' for i in range(n)]
        else:
            names += [f'pressure_mbar_{i}' for i in range(n)]
        names += [f'sigma_{i}' for i in range(n)]
        names += ['impulse_scale']
        names += ['shape_param']
        return names
    
    def _get_pressure(self, vals, i):
        """Return physical pressure for dataset *i* from a parameter dict.

        Handles both the standard ``pressure_mbar_i`` parameterisation and
        the ``pressure_eff_i`` reparameterisation transparently.

        :param vals: Dict mapping parameter names to values.
        :param i: Dataset index.
        :returns: Physical pressure [mbar].
        :rtype: float
        """
        if self.use_pressure_eff:
            return float(vals[f'pressure_eff_{i}']) / float(vals['impulse_scale'])
        return float(vals[f'pressure_mbar_{i}'])

    def _compute_mu(self, ds, amplitude, sigma, T_sph, alpha, s, shape_param):
        """Expected counts per bin for one dataset.

        Centralises the two signal parameterisations so that
        :meth:`_nll_at_scale`, :meth:`_gof_at_scale`, and
        :meth:`goodness_of_fit` all use identical logic.

        When ``use_N_signal=True``, *amplitude* is ``N_signal_i`` — the total
        shape-weighted expected signal counts in the fit window.  The spectrum
        is normalised internally, so ``impulse_scale`` enters only through the
        spectral shape (peak position / width), not through the total count.

        When ``use_N_signal=False``, *amplitude* is the physical pressure
        (already extracted from the parameter dict by the caller).

        :param ds: Dataset dict (keys ``'bins'``, ``'counts'``, ``'scale_factor'``).
        :param amplitude: ``N_signal_i`` or ``pressure_mbar_i`` depending on mode.
        :type amplitude: float
        :param sigma: Impulse resolution [keV/c].
        :type sigma: float
        :param T_sph: Sphere temperature [K].
        :type T_sph: float
        :param alpha: Accommodation coefficient.
        :type alpha: float
        :param s: Impulse scale factor.
        :type s: float
        :param shape_param: Spectral-tilt exponent λ.
        :type shape_param: float
        :returns: Per-bin expected counts (clipped to 1e-10).
        :rtype: numpy.ndarray
        """
        scaled_bins   = s * ds['bins']
        shape_weights = np.exp(float(shape_param) * ds['bins'])

        if self.use_N_signal:
            N_signal_i = float(amplitude)
            # Spectrum shape at unit pressure (linear in P)
            signal_shape = gas_collision_spectrum(
                scaled_bins, 1.0, sigma,
                temperature_K=T_sph, diameter_nm=self.d_nominal,
                alpha=alpha, gas=self.gas,
            )
            shaped_signal = shape_weights * signal_shape
            shaped_sum    = float(np.sum(shaped_signal))
            # Per-bin signal: normalised so total = N_signal_i
            mu_signal = (N_signal_i * shaped_signal / shaped_sum
                         if shaped_sum > 0 else np.zeros_like(ds['bins']))
            # Background from count conservation
            A_bkg  = _compute_derived_A_bkg(
                ds['bins'], float(np.sum(ds['counts'])),
                N_signal_i, sigma, ds['scale_factor'],
                shape_param=float(shape_param),
            )
            mu_bkg = shape_weights * half_gaus(ds['bins'], A_bkg, sigma) / ds['scale_factor']
            return np.maximum(mu_signal + mu_bkg, 1e-10)
        else:
            pressure    = float(amplitude)
            signal_rate = gas_collision_spectrum(
                scaled_bins, pressure, sigma,
                temperature_K=T_sph, diameter_nm=self.d_nominal,
                alpha=alpha, gas=self.gas,
            )
            A_bkg    = self._get_A_bkg_i(ds, signal_rate, sigma,
                                          impulse_scale=s,
                                          shape_param=float(shape_param))
            bkg_rate = half_gaus(ds['bins'], A_bkg, sigma)
            total_rate = self._apply_shape_param(
                ds['bins'], s * signal_rate + bkg_rate, float(shape_param),
            )
            return np.maximum(total_rate / ds['scale_factor'], 1e-10)

    def _apply_shape_param(self, impulses, rate, shape_param):
        return np.exp(shape_param * impulses) * rate

    def _get_A_bkg_i(self, ds, signal_rate_i, sigma_i, impulse_scale=1.0,
                     shape_param=0.0):
        """Derive A_bkg for dataset *ds* at the current parameters.

        The background Gaussian is integrated over the **measured**-impulse
        bins (``ds['bins']``), which removes the ``sigma / impulse_scale``
        degeneracy: the background shape depends only on ``sigma``.

        The shape-tilt weighting ``exp(λ·q)`` is applied to both the signal
        count estimate and the Gaussian denominator so that count conservation
        holds for any *shape_param* value.

        :param ds: Dataset dict with ``'bins'``, ``'counts'``, ``'scale_factor'``.
        :param signal_rate_i: Signal model rate evaluated at the scaled bins
            ``impulse_scale * ds['bins']`` (no shape applied).
        :param sigma_i: Current impulse resolution [keV/c].
        :param impulse_scale: Calibration scale factor *s*.
        :type impulse_scale: float
        :param shape_param: Spectral-tilt exponent λ (default 0).
        :type shape_param: float
        :returns: Derived A_bkg.
        :rtype: float
        """
        shape_factor   = np.exp(float(shape_param) * ds['bins'])
        N_signal_above = float(impulse_scale) * float(
            np.sum(shape_factor * np.asarray(signal_rate_i) / ds['scale_factor'])
        )
        return _compute_derived_A_bkg(
            ds['bins'], float(np.sum(ds['counts'])),
            N_signal_above, sigma_i, ds['scale_factor'],
            shape_param=shape_param,
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
        vals        = dict(zip(minuit.parameters, minuit.values))
        i           = dataset_index
        ds          = self.data[i]
        scale_i     = float(vals['impulse_scale'])
        sigma_i     = float(vals[f'sigma_{i}'])
        shape_param = float(vals['shape_param'])
        if self.use_N_signal:
            return _compute_derived_A_bkg(
                ds['bins'], float(np.sum(ds['counts'])),
                float(vals[f'N_signal_{i}']), sigma_i, ds['scale_factor'],
                shape_param=shape_param,
            )
        pressure_i  = self._get_pressure(vals, i)
        signal_rate_i = gas_collision_spectrum(
            scale_i * ds['bins'], pressure_i, sigma_i,
            temperature_K=float(vals['sphere_temperature_K']),
            diameter_nm=self.d_nominal, alpha=float(vals['alpha']),
            gas=self.gas,
        )
        return self._get_A_bkg_i(
            ds, signal_rate_i, sigma_i,
            impulse_scale=scale_i, shape_param=shape_param,
        )

    def _nll_at_scale(self, ds, amplitude, sigma, T_sph, alpha, _unused, s, shape_param):
        """Per-dataset Poisson/NegBin NLL at a fixed impulse scale *s*.

        *amplitude* is ``N_signal_i`` when ``use_N_signal=True``, otherwise
        pressure [mbar].  Returns ``np.inf`` on NaN inputs.

        :param ds: Dataset dict with ``'bins'``, ``'counts'``, ``'scale_factor'``.
        :param amplitude: N_signal_i or pressure_mbar_i.
        :param sigma: Impulse resolution [keV/c].
        :param T_sph: Sphere temperature [K].
        :param alpha: Accommodation coefficient.
        :param _unused: Ignored (legacy A_bkg_shared argument).
        :param s: Impulse scale factor.
        :param shape_param: Spectral-tilt exponent.
        :returns: Scalar NLL contribution (no prior term).
        :rtype: float
        """
        if np.isnan(s * ds['bins']).any():
            return np.inf
        mu = self._compute_mu(ds, amplitude, sigma, T_sph, alpha, s, shape_param)
        return _nll_terms(ds['counts'], mu, self.f_sys)

    def _gof_at_scale(self, ds, amplitude, sigma, T_sph, alpha, s, shape_param):
        """Per-dataset G-statistic (likelihood-ratio deviance) at a fixed scale.

        *amplitude* is ``N_signal_i`` when ``use_N_signal=True``, otherwise
        pressure [mbar].

        :param ds: Dataset dict with ``'bins'``, ``'counts'``, ``'scale_factor'``.
        :param amplitude: N_signal_i or pressure_mbar_i.
        :param sigma: Impulse resolution [keV/c].
        :param T_sph: Sphere temperature [K].
        :param alpha: Accommodation coefficient.
        :param s: Impulse scale factor.
        :param shape_param: Spectral-tilt exponent.
        :returns: Scalar G-statistic contribution.
        :rtype: float
        """
        if np.isnan(s * ds['bins']).any():
            return np.inf
        mu = self._compute_mu(ds, amplitude, sigma, T_sph, alpha, s, shape_param)
        return float(np.sum(_deviance_terms(ds['counts'], mu, self.f_sys)))

    def __call__(self, *args):
        vals        = dict(zip(self.param_names, args))
        T_sph       = vals['sphere_temperature_K']
        alpha       = vals['alpha']
        shape_param = vals['shape_param']
        s           = vals['impulse_scale']

        nll  = (s - 1.)**2 / (2. * self.sigma_a**2)
        nll += shape_param**2 / (2. * (1. / 500.)**2)
        for i, ds in enumerate(self.data):
            sigma_i = vals[f'sigma_{i}']
            if self.use_N_signal:
                amplitude_i = vals[f'N_signal_{i}']
            else:
                amplitude_i = self._get_pressure(vals, i)
            nll += (sigma_i - self.init_sigmas[i])**2 / (2 * (0.1 * self.init_sigmas[i])**2)
            nll += self._nll_at_scale(ds, amplitude_i, sigma_i, T_sph, alpha, None, s, shape_param)
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
        vals        = dict(zip(minuit.parameters, minuit.values))
        T_sph       = vals['sphere_temperature_K']
        alpha       = vals['alpha']
        s           = vals['impulse_scale']
        shape_param = vals['shape_param']

        g_total = 0.
        for i, ds in enumerate(self.data):
            sigma_i = vals[f'sigma_{i}']
            if self.use_N_signal:
                amplitude_i = vals[f'N_signal_{i}']
            else:
                amplitude_i = self._get_pressure(vals, i)
            mu = self._compute_mu(ds, amplitude_i, sigma_i, T_sph, alpha, s, shape_param)
            g_total += float(np.sum(_deviance_terms(ds['counts'], mu, self.f_sys)))

        total_bins = sum(len(ds['counts']) for ds in self.data)
        ndof       = total_bins - minuit.nfit
        p_value    = float(1 - scipy_stats.chi2.cdf(g_total, ndof))
        return g_total, ndof, p_value

    def get_pressures(self, minuit):
        """Return physical pressures and propagated uncertainties for all datasets.

        Only meaningful when ``use_pressure_eff=True``.  Computes

        .. math::

            P_i = P_{\\mathrm{eff},i} / s

        and propagates the covariance between ``pressure_eff_i`` and
        ``impulse_scale`` via first-order error propagation:

        .. math::

            \\sigma_{P_i}^2 =
                \\frac{\\sigma_{P_{\\mathrm{eff},i}}^2}{s^2}
                + \\frac{P_{\\mathrm{eff},i}^2\\,\\sigma_s^2}{s^4}
                - \\frac{2\\,P_{\\mathrm{eff},i}}{s^3}
                  \\,\\mathrm{Cov}(P_{\\mathrm{eff},i},\\,s)

        :param minuit: Fitted Minuit object with valid covariance matrix.
        :type minuit: iminuit.Minuit
        :returns: List of ``(pressure_mbar, sigma_pressure_mbar)`` tuples,
            one per dataset.
        :rtype: list[tuple[float, float]]
        :raises ValueError: If ``use_pressure_eff`` is ``False``.
        """
        if not (self.use_pressure_eff or self.use_N_signal):
            raise ValueError(
                'get_pressures() requires use_pressure_eff=True or use_N_signal=True; '
                'read pressure_mbar_i directly from minuit.values otherwise.'
            )
        vals = dict(zip(minuit.parameters, minuit.values))
        cov  = minuit.covariance
        s    = float(vals['impulse_scale'])

        if self.use_N_signal:
            # P_i = N_signal_i / F_i, where
            # F_i = s * sum(exp(λ·q) · spectrum(s·q, P=1)) / scale_factor
            # is the total shaped signal counts per unit pressure.
            #
            # Error propagation (first-order, F ∝ s approximation):
            #   dP/dN = 1/F_i
            #   dP/ds ≈ -P_i/s   (dominant: Jacobian term in F_i)
            #
            # var_P = (1/F)² var_N + (P/s)² var_s - 2*(P/s/F) cov(N,s)
            T_sph       = float(vals['sphere_temperature_K'])
            alpha       = float(vals['alpha'])
            shape_param = float(vals['shape_param'])
            var_s       = float(cov['impulse_scale', 'impulse_scale'])
            results     = []
            for i, ds in enumerate(self.data):
                sigma_i = float(vals[f'sigma_{i}'])
                N_i     = float(vals[f'N_signal_{i}'])
                shape_weights = np.exp(shape_param * ds['bins'])
                signal_shape  = gas_collision_spectrum(
                    s * ds['bins'], 1.0, sigma_i,
                    temperature_K=T_sph, diameter_nm=self.d_nominal,
                    alpha=alpha, gas=self.gas,
                )
                F_i = s * float(np.sum(shape_weights * signal_shape)) / ds['scale_factor']
                P_i = N_i / F_i if F_i > 0. else 0.

                var_N  = float(cov[f'N_signal_{i}', f'N_signal_{i}'])
                cov_Ns = float(cov[f'N_signal_{i}', 'impulse_scale'])
                dP_dN  = 1. / F_i if F_i > 0. else 0.
                dP_ds  = -P_i / s if s > 0. else 0.
                var_P  = (dP_dN**2 * var_N
                          + dP_ds**2 * var_s
                          + 2. * dP_dN * dP_ds * cov_Ns)
                results.append((P_i, float(np.sqrt(max(var_P, 0.)))))
            return results

        # pressure_eff mode
        results = []
        for i in range(self.n):
            p_eff   = float(vals[f'pressure_eff_{i}'])
            p_phys  = p_eff / s

            var_peff = float(cov[f'pressure_eff_{i}', f'pressure_eff_{i}'])
            var_s    = float(cov['impulse_scale',      'impulse_scale'])
            cov_ps   = float(cov[f'pressure_eff_{i}', 'impulse_scale'])

            var_p = (var_peff / s**2
                     + p_eff**2 * var_s / s**4
                     - 2. * p_eff * cov_ps / s**3)
            results.append((p_phys, float(np.sqrt(max(var_p, 0.)))))
        return results

    def scan(self, T_values, alpha_values, pressures=None, sigmas=None,
             impulse_scale=1.0, shape_param=0.0, amplitudes=None):
        """Scan the NLL over a grid of (sphere_temperature_K, alpha).

        All other parameters are held fixed at the supplied values.  The
        grid minimum is a reliable starting point for the full Minuit fit
        when ``alpha`` and ``sphere_temperature_K`` are sensitive to
        initialisation.

        :param T_values: 1-D array of ``sphere_temperature_K`` values to
            scan [K].
        :type T_values: array_like
        :param alpha_values: 1-D array of accommodation-coefficient values
            to scan (should lie in [0, 1]).
        :type alpha_values: array_like
        :param pressures: Pressure [mbar] held fixed for each dataset
            (used when ``use_N_signal=False``).  Defaults to
            ``ds['ccg_pressure']`` for each dataset.
        :type pressures: list[float] or None
        :param amplitudes: ``N_signal_i`` values held fixed for each dataset
            (used when ``use_N_signal=True``).  Required in that mode; there
            is no sensible default.  A rough estimate is 50–80 % of the
            total observed counts in each dataset.
        :type amplitudes: list[float] or None
        :param sigmas: Impulse resolution [keV/c] held fixed for each
            dataset.  Defaults to ``self.init_sigmas``.
        :type sigmas: list[float] or None
        :param impulse_scale: Fixed calibration scale (default 1.0).
        :type impulse_scale: float
        :param shape_param: Fixed shape parameter (default 0.0).
        :type shape_param: float
        :returns: Dictionary with keys:

            * ``'gof_grid'`` – 2-D ``ndarray`` of shape
              ``(len(T_values), len(alpha_values))`` containing the summed
              likelihood-ratio G-statistic at each grid point.  Row *i*
              corresponds to ``T_values[i]``, column *j* to
              ``alpha_values[j]``.
            * ``'T_values'``, ``'alpha_values'`` – the scan arrays as
              passed in.
            * ``'best_T'`` – ``sphere_temperature_K`` at the grid minimum.
            * ``'best_alpha'`` – ``alpha`` at the grid minimum.
            * ``'best_gof'`` – G-statistic at the grid minimum.

        :rtype: dict
        """
        T_arr     = np.asarray(T_values, dtype=float)
        alpha_arr = np.asarray(alpha_values, dtype=float)

        s = float(impulse_scale)

        if self.use_N_signal:
            if amplitudes is None:
                raise ValueError(
                    'scan() requires amplitudes=[N_signal_i, ...] when use_N_signal=True. '
                    'A rough initial estimate is ~0.5 * sum(ds["counts"]) per dataset.'
                )
            scan_amplitudes = [float(a) for a in amplitudes]
        else:
            if pressures is None:
                pressures = [ds['ccg_pressure'] for ds in self.data]
            scan_amplitudes = [float(p) for p in pressures]

        if sigmas is None:
            sigmas = list(self.init_sigmas)

        gof_grid = np.empty((len(T_arr), len(alpha_arr)))

        for i_T, T_sph in enumerate(T_arr):
            for i_a, alpha in enumerate(alpha_arr):
                gof = 0.
                for i_ds, ds in enumerate(self.data):
                    gof += self._gof_at_scale(
                        ds,
                        scan_amplitudes[i_ds],
                        float(sigmas[i_ds]),
                        T_sph, alpha, s, float(shape_param),
                    )
                gof_grid[i_T, i_a] = gof

        best_idx = np.unravel_index(np.argmin(gof_grid), gof_grid.shape)
        return {
            'gof_grid':     gof_grid,
            'T_values':     T_arr,
            'alpha_values': alpha_arr,
            'best_T':       float(T_arr[best_idx[0]]),
            'best_alpha':   float(alpha_arr[best_idx[1]]),
            'best_gof':     float(gof_grid[best_idx]),
        }

    def _clone_minuit(self, minuit):
        """Reconstruct a fresh Minuit instance from *minuit*'s current state.

        Copies parameter values, step sizes, limits, and fixed flags so the
        clone starts from the same point and respects the same constraints.

        :param minuit: Source Minuit object.
        :type minuit: iminuit.Minuit
        :returns: New, un-minimised Minuit object.
        :rtype: iminuit.Minuit
        """
        from iminuit import Minuit as _Minuit
        init_vals = [float(minuit.values[p]) for p in minuit.parameters]
        m = _Minuit(self, *init_vals, name=list(minuit.parameters))
        for p in minuit.parameters:
            m.errors[p] = float(minuit.errors[p])
            m.limits[p] = minuit.limits[p]
            m.fixed[p]  = bool(minuit.fixed[p])
        return m

    def profile(self, minuit, param, param2=None, n_points=20, plot=False):
        """Profile likelihood ratio scan for one or two parameters.

        Fixes the profiled parameter(s) at each scan point and re-minimises
        over all remaining free parameters, returning the profile
        log-likelihood ratio

        .. math::

            -2\\,\\Delta\\ln\\mathcal{L} =
                2\\bigl(\\mathrm{NLL}_{\\mathrm{fixed}}
                        - \\mathrm{NLL}_{\\mathrm{best}}\\bigr).

        The scan range is estimated as ±3σ around the best-fit value using
        MINOS errors when available and HESSE errors otherwise.

        Confidence-interval thresholds for :math:`-2\\Delta\\ln\\mathcal{L}`:

        * **1-D** (1 d.o.f.): 1σ → 1.00, 2σ → 4.00, 3σ → 9.00
        * **2-D** (2 d.o.f.): 1σ → 2.30, 2σ → 6.18, 3σ → 11.83

        :param minuit: Fitted Minuit object with valid ``fval``.
        :type minuit: iminuit.Minuit
        :param param: Parameter name to profile.
        :type param: str
        :param param2: If given, perform a 2-D profile scan over (*param*,
            *param2*).
        :type param2: str or None
        :param n_points: Number of scan points along each axis (default 20).
        :type n_points: int
        :param plot: Display a profile plot (1-D) or contour plot (2-D) with
            1σ, 2σ, and 3σ contours when ``True`` (default ``False``).
        :type plot: bool
        :returns: Dictionary.  For a 1-D scan:

            * ``'param'`` – parameter name.
            * ``'values'`` – scanned values (length *n_points*).
            * ``'delta_nll'`` – :math:`-2\\Delta\\ln\\mathcal{L}` at each point.
            * ``'fig'``, ``'ax'`` – Matplotlib objects (present only when
              *plot* is ``True``).

            For a 2-D scan:

            * ``'params'`` – ``(param, param2)``.
            * ``'values_1'``, ``'values_2'`` – scanned values for each axis.
            * ``'delta_nll'`` – 2-D array of shape ``(n_points, n_points)``;
              ``delta_nll[i, j]`` is at ``values_1[i]``, ``values_2[j]``.
            * ``'fig'``, ``'ax'`` – Matplotlib objects (present only when
              *plot* is ``True``).

        :rtype: dict
        """
        nll_best = minuit.fval

        def _get_range(p):
            best = float(minuit.values[p])
            try:
                me = minuit.merrors[p]
                lo = best + 3. * me.lower   # me.lower < 0
                hi = best + 3. * me.upper   # me.upper > 0
            except KeyError:
                err = float(minuit.errors[p])
                lo  = best - 3. * err
                hi  = best + 3. * err
            return np.linspace(lo, hi, n_points)

        if param2 is None:
            # ------------------------------------------------------------------
            # 1-D profile
            # ------------------------------------------------------------------
            scan_vals = _get_range(param)
            delta_nll = np.empty(n_points)

            for i, val in enumerate(scan_vals):
                m = self._clone_minuit(minuit)
                m.values[param] = val
                m.fixed[param]  = True
                m.migrad()
                delta_nll[i] = 2. * (m.fval - nll_best)

            result = {'param': param, 'values': scan_vals, 'delta_nll': delta_nll}

            if plot:
                import matplotlib.pyplot as plt
                plt.style.use('clarke-default')
                fig, ax = plt.subplots()
                ax.plot(scan_vals, delta_nll)
                for level, ls, lbl in zip(
                    [1., 4., 9.], ['--', '-.', ':'], ['1σ', '2σ', '3σ']
                ):
                    ax.axhline(level, ls=ls, color='C1', lw=1., label=lbl)
                ax.set_xlabel(param)
                ax.set_ylabel(r'$-2\,\Delta\ln\mathcal{L}$')
                ax.legend()
                plt.tight_layout()
                plt.show()
                result['fig'] = fig
                result['ax']  = ax

            return result

        else:
            # ------------------------------------------------------------------
            # 2-D profile
            # ------------------------------------------------------------------
            scan_vals_1 = _get_range(param)
            scan_vals_2 = _get_range(param2)
            delta_nll   = np.empty((n_points, n_points))

            for i, v1 in enumerate(scan_vals_1):
                for j, v2 in enumerate(scan_vals_2):
                    m = self._clone_minuit(minuit)
                    m.values[param]  = v1
                    m.values[param2] = v2
                    m.fixed[param]   = True
                    m.fixed[param2]  = True
                    m.migrad()
                    delta_nll[i, j] = 2. * (m.fval - nll_best)

            result = {
                'params':    (param, param2),
                'values_1':  scan_vals_1,
                'values_2':  scan_vals_2,
                'delta_nll': delta_nll,
            }

            if plot:
                import matplotlib.pyplot as plt
                from scipy.stats import chi2 as _chi2
                plt.style.use('clarke-default')
                # 2-d.o.f. chi-squared thresholds
                sigma_levels = _chi2.ppf([0.6827, 0.9545, 0.9973], df=2)
                fig, ax = plt.subplots()
                # delta_nll[i,j]: axis-0 = param (y), axis-1 = param2 (x)
                CS = ax.contour(
                    scan_vals_2, scan_vals_1, delta_nll,
                    levels=sigma_levels,
                    linestyles=['--', '-.', ':'],
                )
                ax.clabel(CS, fmt={l: s for l, s in
                                   zip(sigma_levels, ['1σ', '2σ', '3σ'])})
                ax.plot(
                    minuit.values[param2], minuit.values[param],
                    '+', ms=10, label='best fit',
                )
                ax.set_xlabel(param2)
                ax.set_ylabel(param)
                ax.legend()
                plt.tight_layout()
                plt.show()
                result['fig'] = fig
                result['ax']  = ax

            return result
