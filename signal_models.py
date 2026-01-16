"""
Signal models for gas collision spectra in trapped nanosphere experiments.

This module provides functions to compute the expected impulse spectrum
from gas molecule collisions with a trapped nanosphere.
"""

import numpy as np
from scipy.special import erf


# Physical constants
kb = 1.380649e-23  # Boltzmann constant (J/K)
c = 299792458      # Speed of light (m/s)
SI2ev = (1 / 1.6e-19) * c  # Conversion from SI momentum to eV/c
ev2SI = 1 / SI2ev          # Conversion from eV/c to SI momentum

# Nanosphere properties (83 nm radius silica sphere)
SPHERE_RADIUS = 83e-9  # meters
SPHERE_SURFACE_AREA = 4 * np.pi * SPHERE_RADIUS**2  # m^2

# Gas properties
XE_MASS_AMU = 131.3  # Xenon atomic mass in amu


def _gauss(x, A, mu, sigma):
    """Gaussian function."""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def _fmb(dp, mg, vbar):
    """Maxwell-Boltzmann momentum distribution factor."""
    return np.exp(-dp**2 / (8 * mg**2 * vbar**2)) / np.sqrt(2 * np.pi * vbar**2)


def _xi(x):
    """Correction factor for specular reflection."""
    return (np.sqrt(np.pi) * x * (1 - 2 / (x**2)) * erf(x / 2) * np.exp(-x**2 / 8)
            + 2 * np.exp(-3 * x**2 / 8))


def _dgamma_dp(pp_kev, A, mg_amu, p_mbar, alpha, T=293):
    """
    Compute differential collision rate dGamma/dp for isothermal gas.

    Parameters
    ----------
    pp_kev : array_like
        Momentum transfer values in keV/c
    A : float
        Sphere surface area in m^2
    mg_amu : float
        Gas molecule mass in atomic mass units
    p_mbar : float
        Gas pressure in mbar
    alpha : float
        Accommodation coefficient (0 = specular, 1 = diffuse)
    T : float, optional
        Temperature in Kelvin (default: 293)

    Returns
    -------
    array_like
        Differential rate in Hz/keV
    """
    pp = pp_kev * 1000 * ev2SI  # Convert to SI (N*s)
    mg = mg_amu * 1.660538921e-27  # Convert to kg
    p_pascal = p_mbar * 100  # Convert to Pascal
    ng = p_pascal / (kb * T)  # Number density

    vbar = np.sqrt(kb * T / mg)  # Thermal velocity

    rate = ((ng * A * pp / (4 * mg**2)) * _fmb(pp, mg, vbar)
            * (1 - alpha + alpha * _xi(pp / (mg * vbar))))
    rate_hz_kev = rate * 1000 * ev2SI  # Convert to Hz/keV
    return rate_hz_kev


def _get_drdqz(qq, drdq):
    """
    Project isotropic differential rate onto the z-axis.

    Parameters
    ----------
    qq : array_like
        Momentum values
    drdq : array_like
        Isotropic differential rate dR/dq

    Returns
    -------
    tuple
        (qq, drdqz) - momentum values and z-projected differential rate
    """
    drdq_iso = drdq / (4 * np.pi * qq**2)

    ret = np.empty_like(qq)
    for i, q in enumerate(qq):
        xx = qq[qq >= q]
        integrand = drdq_iso[qq >= q]
        # Factor of 2 for both +z and -z directions
        ret[i] = 2 * 2 * np.pi * np.trapz(integrand * xx, xx)

    return qq, ret


def _smear_drdqz_gauss(qq, drdqz, sigma_kev):
    """
    Convolve spectrum with a Gaussian kernel to model detector resolution.

    Parameters
    ----------
    qq : array_like
        Momentum values in keV/c
    drdqz : array_like
        Z-projected differential rate
    sigma_kev : float
        Gaussian smearing width (detector resolution) in keV/c

    Returns
    -------
    tuple
        (qq, smeared_drdqz) - momentum values and smeared differential rate
    """
    dq = qq[1] - qq[0]
    qq_gauss = np.arange(-2000, 2000, dq)
    gauss_kernel = _gauss(qq_gauss, A=1, mu=0, sigma=sigma_kev)

    # Pad the array to minimize edge effects
    pad_len = gauss_kernel.size
    if qq[0] >= dq:
        padded_drdqz = np.pad(drdqz, (pad_len, 0), mode='symmetric')
    else:
        padded_drdqz = np.pad(drdqz, (pad_len, 0), mode='reflect')
    padded_drdqz = np.pad(padded_drdqz, (0, pad_len), mode='constant', constant_values=0)

    convolved = np.convolve(padded_drdqz, gauss_kernel, mode='valid')

    idx_start = (convolved.size - drdqz.size) // 2
    ret = convolved[idx_start:idx_start + drdqz.size] / np.sum(gauss_kernel)

    return qq, ret


def gas_collision_spectrum(impulses_kev, pressure_mbar, resolution_kev,
                           temperature_K=293, diameter_nm=166.):
    """
    Compute the expected gas collision spectrum for a trapped nanosphere.

    Calculates the differential collision rate from gas molecules (Xenon)
    impacting a silica nanosphere, projected onto the measurement axis
    and convolved with the detector resolution.

    Parameters
    ----------
    impulses_kev : array_like
        Array of impulse (momentum transfer) values in keV/c at which to
        evaluate the spectrum.
    pressure_mbar : float
        Chamber pressure in mbar.
    resolution_kev : float
        Detector resolution (Gaussian sigma) in keV/c, used for smearing.
    temperature_K : float, optional
        Gas/nanosphere temperature in Kelvin (default: 293 K).
    diameter_nm : float, optional
        Nanosphere diameter in nanometers (default: 166 nm).

    Returns
    -------
    numpy.ndarray
        Collision rate in counts/(50 keV/c)/second, same shape as impulses_kev.

    Examples
    --------
    >>> import numpy as np
    >>> from signal_models import gas_collision_spectrum
    >>> q = np.linspace(1, 5000, 1000)
    >>> spectrum = gas_collision_spectrum(q, pressure_mbar=1e-9, resolution_kev=200, diameter_nm=100)
    """
    impulses_kev = np.asarray(impulses_kev)

    # Compute sphere surface area from diameter
    radius_m = (diameter_nm * 1e-9) / 2.
    surface_area = 4 * np.pi * radius_m**2

    # Compute differential rate for Xenon gas with diffuse reflection (alpha=1)
    dr_dq = _dgamma_dp(impulses_kev, surface_area, XE_MASS_AMU,
                       pressure_mbar, alpha=1, T=temperature_K)

    # Project onto measurement axis (z)
    _, drdqz = _get_drdqz(impulses_kev, dr_dq)

    # Apply detector resolution smearing
    _, drdqz_smeared = _smear_drdqz_gauss(impulses_kev, drdqz, resolution_kev)

    # Convert to counts per 50 keV/c bin per second
    rate_per_50kev = 50 * drdqz_smeared

    return rate_per_50kev
