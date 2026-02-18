"""Plotting functions for nanosphere data.

Each function accepts a :class:`~data_processing.NanoFile` instance whose
``compute_and_fit_psd`` method has already been called, and a
:class:`matplotlib.axes.Axes` object to draw into.  An optional *title*
string is applied when provided.

These functions are called both by :meth:`NanoFile.compute_and_fit_psd`
(when saving a PDF) and by the live-data GUI.
"""

import numpy as np
from scipy.special import voigt_profile


def _susc(omega, A, omega_0, gamma):
    """Complex mechanical susceptibility (mirrors data_processing.susc)."""
    return A / (omega_0**2 - omega**2 - 1j * gamma * omega)


def plot_raw_asd(nf, ax, title=None):
    """Plot the raw ASD with Voigt fit, scaled susceptibility and noise floor.

    :param nf: NanoFile instance after ``compute_and_fit_psd``.
    :param ax: Matplotlib axes to draw into.
    :param title: Optional axes title.
    """
    std   = nf.spectrum_to_density
    freqs = nf.freqs

    scaled_susc = (
        np.abs(_susc(2*np.pi*freqs, 1/nf.mass_sphere, nf.omega_0, nf.gamma))
        * nf.p_amp * nf.mass_sphere
        * np.sqrt(2 * nf.omega_0**2 * nf.gamma / np.pi) * std
    )

    ax.semilogy(freqs*1e-3, np.sqrt(nf.Pxx_z_raw)*std,
                alpha=1.0, label='$z$ raw')
    ax.semilogy(freqs*1e-3, np.sqrt(nf.Pxx_z_filt)*std,
                alpha=1.0, label='$z$ filtered')
    ax.semilogy(freqs*1e-3,
                np.sqrt(nf.p_amp**2 * voigt_profile(
                    2*np.pi*freqs - nf.omega_0, nf.sigma, nf.gamma/2)
                    + nf.noise_floor) * std,
                alpha=1.0, lw=1.0, label='Voigt fit', zorder=100)
    ax.semilogy(freqs*1e-3, scaled_susc,
                alpha=1.0, lw=1.0, label=r'Scaled $\chi$')
    ax.axhline(np.sqrt(nf.noise_floor)*std, lw=1.0, color='C4',
               label='Noise floor')
    ax.text(nf.omega_0*1e-3/2/np.pi, 0.9*np.sqrt(nf.noise_floor)*std,
            '{:.1f} kHz\n effective bandwidth'.format(nf.effective_bw),
            ha='center', va='top')

    if title:
        ax.set_title(title)
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel(r'Raw ASD [V/$\sqrt{\mathrm{Hz}}$]')
    ax.set_ylim([1e-7, 1e-1])
    ax.set_xlim([20, 100])
    ax.legend(loc='upper right')

    params = np.array([nf.p_amp, nf.omega_0, nf.sigma, nf.gamma])
    exp  = np.floor(np.log10(np.abs(params))).astype(int)
    mant = params / np.power(10., exp)
    ax.text(0.03, 0.97,
            '$A={:.3f}\\times10^{{{:.0f}}}$, \n'.format(mant[0], exp[0])
            + '$\\omega_0={:.3f}\\times10^{{{:.0f}}}'.format(mant[1], exp[1])
            + '~\\mathrm{s^{-1}}$\n'
            + '$\\sigma={:.3f}\\times10^{{{:.0f}}}'.format(mant[2], exp[2])
            + '~\\mathrm{s^{-1}}$\n'
            + '$\\gamma={:.3f}\\times10^{{{:.0f}}}'.format(mant[3], exp[3])
            + '~\\mathrm{s^{-1}}$',
            ha='left', va='top', transform=ax.transAxes)
    ax.grid(which='both')


def plot_force_asd(nf, ax, title=None):
    """Plot the force-calibrated ASD in N/√Hz.

    :param nf: NanoFile instance after ``compute_and_fit_psd``.
    :param ax: Matplotlib axes to draw into.
    :param title: Optional axes title.
    """
    std   = nf.spectrum_to_density
    freqs = nf.freqs

    asd = (np.sqrt(nf.Pxx_z_filt) * std * nf.meters_per_volt
           / np.abs(nf.susceptibility(2*np.pi*freqs)))

    ax.semilogy(freqs*1e-3, asd, alpha=1.0, label='$z$ raw')
    if title:
        ax.set_title(title)
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel(r'Calibrated ASD [$\mathrm{N/\sqrt{Hz}}$]')
    ax.set_xlim([20, 100])
    ax.set_ylim([1e-21, 1e-18])
    ax.text(0.03, 0.97, '$\\Delta p={:.1f}$ keV/c'.format(nf.p_sens),
            ha='left', va='top', transform=ax.transAxes)
    ax.grid(which='both')


def plot_position_asd(nf, ax, title=None):
    """Plot the position-calibrated ASD in m/√Hz.

    :param nf: NanoFile instance after ``compute_and_fit_psd``.
    :param ax: Matplotlib axes to draw into.
    :param title: Optional axes title.
    """
    std   = nf.spectrum_to_density
    freqs = nf.freqs

    asd = np.sqrt(nf.Pxx_z_filt) * std * nf.meters_per_volt

    ax.semilogy(freqs*1e-3, asd)
    if title:
        ax.set_title(title)
    ax.set_xlabel('Frequency [kHz]')
    ax.set_ylabel(r'Calibrated ASD [$\mathrm{m/\sqrt{Hz}}$]')
    ax.set_xlim([20, 100])
    ax.set_ylim([1e-14, 1e-9])
    ax.text(0.03, 0.97,
            '$n={:.0f}$ phonons,\n$T={:.0f}$ mK'.format(
                nf.n_avg, nf.T_eff * 1e3),
            ha='left', va='top', transform=ax.transAxes)
    ax.grid(which='both')
