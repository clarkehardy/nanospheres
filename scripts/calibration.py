import argparse
import numpy as np
from glob import glob
import os
import gc
from pathlib import Path
import yaml

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages

from scipy.optimize import curve_fit
import scipy.signal as sig
from scipy import stats

from joblib import Parallel, delayed, cpu_count

import sys
sys.path.insert(0, '/Users/clarke/Code/nanospheres/')
import data_processing as dp

plt.style.use('clarke-default')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Calibration script for nanosphere impulse sensor')
parser.add_argument('--data-type', type=str, default='gas_collisions',
                    help='Type of data (default: gas_collisions)')
parser.add_argument('--sphere-date', type=str, default='20251212',
                    help='Sphere date identifier (default: 20251212)')
parser.add_argument('--max-files', type=int, default=25,
                    help='Maximum number of files to process per dataset (default: 25)')
parser.add_argument('--drive-path', type=str, default='/Users/clarke/Data/',
                    help='Path to data drive (default: /Users/clarke/Data/)')
parser.add_argument('--sphere-diam', type=float, default=100.,
                    help='Nanosphere diameter in nanometers (default: 100)')
args = parser.parse_args()

data_type = args.data_type
sphere_date = args.sphere_date
max_files = args.max_files
drive_path = args.drive_path
d_sphere = args.sphere_diam

base_path = f'{drive_path}/{data_type}/pulse_calibration/sphere_{sphere_date}/'

if not os.path.exists(drive_path):
    print('Error: check that the external drive is plugged in!')

folders = glob(base_path + '*')

datasets = {}

for folder in folders:
    print(folder.split(base_path)[-1])
    all_items = glob(folder + '/*.hdf5')
    subfolders = np.unique(['_'.join(s.split('_')[:-1]) for s in all_items])
    sub_datasets = []
    for subfolder in subfolders:
        sub_datasets.append(subfolder.split(base_path)[-1].split('/')[-1])
        print('\t' + subfolder.split(base_path)[-1].split('/')[-1])
    datasets[folder.split(base_path)[-1]] = sub_datasets

pulse_amps_1e = np.asarray([12, 34, 57, 80, 103, 127, 150, 174]) # impulses applied to the particle in eV/c
pulse_amps_V = np.arange(2.5, 21, 2.5) # pulse amplitudes applied to lens holder 1 in V

# t_window = 2e-3 # the window length in ms
search_window = 5e-5
fit_window = 1e-1 # time window for resonance fits
f_cutoff_high = 1e5 # upper cutoff frequency for the bandpass filter
f_cutoff_low = 2.5e4 # lower cutoff frequency for the bandpass filter
f_cutoff = [f_cutoff_low, f_cutoff_high]

config = {
    'fit_window': fit_window,
    'search_window': search_window,
    'meters_per_volt': None,
    'f_cutoff': f_cutoff,
    'apply_notch': False,
    'calibrate': True,
    'd_sphere_nm': d_sphere
}

def gaus(x, A, mu, sigma):
    """Gaussian function."""
    return A*np.exp(-(x - mu)**2/2/sigma**2)

def slope(x, y, sigma=1):
    """Analytic form of the best-fit slope to two
    series of data (linear fit through origin).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    sigma = np.asarray(sigma)

    w = 1.0 / sigma**2
    denom = np.sum(w * x**2)
    if denom == 0:
        raise ValueError("Degenerate x values; slope is undefined.")

    return np.sum(w * x * y) / denom

def quad(x, a, b, c):
    """Quadratic function for chi2 vs impulse fit."""
    return a*x**2 + b*x + c

def process_dataset(dataset_ind):
    dataset = list(datasets.keys())[dataset_ind]

    if 'electric_calibration' in dataset:
        return

    # get the list of impulse amplitudes in that dataset
    amp_list = np.argsort([float(d.split('v')[0].split('_')[-1]) for d in datasets[dataset]])
    amp_list = np.concat(([amp_list[0]], amp_list))

    pulse_amps_keV = []
    impulses = []
    resonance_params = []
    fit_success = []
    pulses = []
    recon_inds = []
    timestamps = []
    pulse_times = []
    meters_per_volt = []
    impulse_rms = []

    # loop through all files in that dataset in the correct order
    for i, file_ind in enumerate(amp_list):
        filename = datasets[list(datasets.keys())[dataset_ind]][file_ind]
        plot_path = 'figures/' + base_path.split(drive_path)[-1] + dataset + '/' + filename

        nd = dp.NanoDataset(base_path + dataset + '/' + filename, plot_path, verbose=True,
                            max_files=max_files, ds_factor=10, config=config)
        nd.load_calibration_data(global_params=False, pulse_amps_1e=pulse_amps_1e,
                                 pulse_amps_V=pulse_amps_V, noise=i==0)
        impulses.append(nd.impulses.copy())
        pulse_amps_keV.append([nd.pulse_amp_keV.copy(), 0][i==0])
        resonance_params.append(nd.resonance_params.copy())
        fit_success.append(nd.fit_success.copy())
        pulses.append(nd.pulses.copy())
        recon_inds.append(nd.recon_impulse_inds.copy())
        pulse_times.append(nd.pulse_times.copy())
        timestamps.append(nd.timestamps.copy())
        meters_per_volt.append(nd.meters_per_volt.copy())
        impulse_rms.append(nd.impulse_rms.copy())

        if i == 0:
            freqs = np.copy(nd.freqs)
        del nd
        gc.collect()

    pulse_amps_keV = np.array(pulse_amps_keV)
    min_len = np.amin([len(l) for l in impulses])
    impulses = np.array([l[:min_len] for l in impulses])
    pulses = np.array([l[:min_len] for l in pulses])
    recon_inds = np.array([l[:min_len] for l in recon_inds])
    resonance_params = np.array([l[:min_len] for l in resonance_params])
    fit_success = np.array([l[:min_len] for l in fit_success])
    pulse_times = np.array([l[:min_len] for l in pulse_times])
    timestamps = np.array([l[:min_len] for l in timestamps])
    meters_per_volt = np.mean(meters_per_volt)
    impulse_rms = np.array([l[:min_len] for l in impulse_rms])

    # Create mask based on RMS noise and fit convergence
    rms_cut = 6e-18 if d_sphere < 150 else 1e-17 # cut on RMS noise in Newtons
    rms_mask = impulse_rms < rms_cut
    mask = rms_mask & fit_success

    # Plot RMS over time
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for i in range(impulse_rms.shape[0]):
        ax.plot(pulse_times[i], np.abs(impulse_rms[i]), label='{:.0f} keV'.format(pulse_amps_keV[i]))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Force RMS [N]')
    ax.set_yscale('log')
    ax.axhline(rms_cut, ls='--', color='k')
    ax.legend(ncol=3, fontsize=10)
    ax.grid(which='both')
    ax.set_title('Deconvolved force RMS over time')
    fig.savefig(plot_path + '_rms.pdf')
    plt.close(fig)

    # Resonance parameter drift plots
    n_repeats = (pulse_times.shape[1] + timestamps.shape[1] - 1) // timestamps.shape[1]
    pulse_timestamps = pulse_times + np.tile(timestamps, (1, n_repeats))[:, :pulse_times.shape[1]]
    res_evol = np.array([resonance_params[...,i].flatten()[np.argsort(pulse_timestamps.flatten())] for i in range(3)])
    res_times = np.sort(pulse_timestamps.flatten()) - np.amin(pulse_timestamps)

    fig, ax = plt.subplots(3, 2, figsize=(6, 8), sharex='col', sharey='row', width_ratios=[4, 1], layout='constrained')
    ax[0,0].semilogy(res_times, res_evol[0], marker='.', ms=2, ls='none')
    ax[0,1].hist(res_evol[0], bins=np.logspace(np.log10(np.amin(res_evol[0])), \
                np.log10(np.amax(res_evol[0])), 20), histtype='step', orientation='horizontal')
    ax[1,0].plot(res_times, res_evol[1]/2/np.pi, marker='.', ms=2, ls='none')
    ax[1,1].hist(res_evol[1]/2/np.pi, bins=20, histtype='step', orientation='horizontal')
    ax[2,0].semilogy(res_times, res_evol[2]/2/np.pi, marker='.', ms=2, ls='none')
    ax[2,1].hist(res_evol[2]/2/np.pi, bins=np.logspace(np.log10(np.amin(res_evol[2]/2/np.pi)), \
                np.log10(np.amax(res_evol[2]/2/np.pi)), 20), \
                histtype='step', orientation='horizontal')
    ax[0,0].set_ylabel(r'$A$ [$\mathrm{V^2\,Hz^3}$]')
    ax[1,0].set_ylabel(r'$\omega_0/2\pi$ [$\mathrm{s}^{-1}$]')
    ax[2,0].set_ylabel(r'$\gamma/2\pi$ [$\mathrm{s}^{-1}$]')
    ax[2,0].set_xlabel('Time since start [s]')
    ax[2,1].set_xlabel('Counts')
    [ax[i, j].grid(which='both') for i in range(3) for j in range(2)]
    fig.suptitle('Resonance parameter drift')
    fig.savefig(plot_path + '_res_evol.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    freq_range = np.mean(res_evol[1])/2/np.pi/1e3 + np.array((-5, 5))
    omegas = 2*np.pi*1e3*np.linspace(np.amin(freq_range), np.amax(freq_range), 200)
    colors = [plt.get_cmap('plasma', res_evol.shape[1])(i) for i in range(res_evol.shape[1])]
    for i, params in enumerate(res_evol.T):
        ax.plot(1e-3*omegas/2/np.pi, dp.abs_susc2(omegas, *params), alpha=0.3, lw=0.3, color=colors[i])
    ax.set_yscale('log')
    ax.set_xlabel('Frequency [kHz]')
    ax.set_xlim(freq_range)
    ax.set_ylabel(r'$|\chi(\omega)|^2$')
    ax.set_title('Resonance drift')
    ax.grid()
    fig.colorbar(ScalarMappable(norm=Normalize(0, np.amax(res_times)), cmap='plasma'), ax=ax, label='Time since start [s]', pad=0.02)
    fig.savefig(plot_path + '_res_drift.pdf')
    plt.close(fig)

    # Build a time array and get cropped waveforms
    half_window = 25
    peak_ind = np.argmax(np.abs(np.mean(pulses[-1], axis=0)))
    times = np.arange(0, pulses.shape[-1]/2./freqs[-1], 1./2./freqs[-1])
    times -= times[peak_ind]
    time_slice = times[..., peak_ind - half_window:peak_ind + half_window]

    # Get cropped waveforms centered around the true impulse time, with masking
    pulses_true_imp = np.copy(pulses[..., peak_ind - half_window:peak_ind + half_window])
    pulses_true_imp[~mask] = np.nan
    mean_pulses_true_imp = np.nanmean(pulses_true_imp, axis=1)

    fig, ax = plt.subplots(mean_pulses_true_imp.shape[0], figsize=(6, 3*mean_pulses_true_imp.shape[0]), \
                        sharex=True, sharey=True, layout='constrained')
    for i in range(mean_pulses_true_imp.shape[0]):
        ax[i].plot(time_slice*1e6, pulses_true_imp[i].T*1e-8, lw=0.2, alpha=0.1, color='k')
        ax[i].plot(time_slice*1e6, mean_pulses_true_imp[i]*1e-8)
        ax[i].set_ylabel('Amplitude [$10^8$ au]')
        ax[i].set_title('{:.0f} keV/c impulse'.format(pulse_amps_keV[i]))
        ax[i].grid(which='both')
    ax[-1].set_xlabel(r'Time [$\mu$s]')
    fig.suptitle('Average waveforms')
    fig.savefig(plot_path + '_avg_wfms_uncal.pdf')
    plt.close(fig)

    # Calibration
    means = []
    errs = []

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for i, imp_all in enumerate(impulses):
        imp = imp_all[mask[i]]
        mean = np.mean(np.abs(imp))
        std = np.std(np.abs(imp))
        counts, bins = np.histogram(np.abs(imp), bins=np.linspace(np.amax((0, mean - 4*std)), mean + 4*std, 20))
        bins = (bins[:-1] + bins[1:])/2.
        p, _ = curve_fit(gaus, bins, counts, p0=(10, np.mean(np.abs(imp)), np.std(np.abs(imp))))
        plot_bins = np.linspace(bins[0], bins[-1], 200)
        ax.step(bins, counts, color='C' + str(i), where='mid')
        ax.plot(plot_bins, gaus(plot_bins, *p), color='C' + str(i))
        means.append(p[1])
        errs.append(np.abs(p[2]))
        ax.axvline(p[1], ls='--', color='C' + str(i))
    ax.set_xlabel('Reconstructed impulse [au]')
    ax.set_ylabel('Counts')
    ax.set_title('Reconstructed impulses before calibration')
    ax.grid()
    fig.savefig(plot_path + '_recon_uncal.pdf')
    plt.close(fig)

    means = np.array(means)
    errs = np.array(errs)

    exclude_first = 2  # how many of the first datasets to exclude from the calibration line fit

    # Compute the scaling from Newtons to keV/c using slope through origin
    to_keV = 1./slope(pulse_amps_keV[exclude_first:], means[exclude_first:], errs[exclude_first:])

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    plot_amps = np.linspace(0, np.amax(pulse_amps_keV), 2)
    ax.errorbar(pulse_amps_keV[exclude_first:], to_keV*means[exclude_first:], \
                to_keV*errs[exclude_first:], marker='.', ls='none', label='Calibration data')
    ax.errorbar(pulse_amps_keV[:exclude_first], to_keV*means[:exclude_first], \
                to_keV*errs[:exclude_first], marker='.', ls='none', label='Excluded points')
    ax.plot(plot_amps, plot_amps, label='$y=x$')
    ax.set_xlabel('True impulse [keV/c]')
    ax.set_ylabel('Reconstructed impulse [keV/c]')
    ax.set_title('Accuracy of impulse calibration')
    ax.legend()
    ax.grid()
    fig.savefig(plot_path + '_calibration.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.plot(pulse_amps_keV, to_keV*errs, marker='o', ls='none', fillstyle='none')
    ax.set_xlabel('True impulse [keV/c]')
    ax.set_ylabel('Resolution [keV/c]')
    ax.set_title('Resolution as a function of impulse')
    ax.grid()
    fig.savefig(plot_path + '_resolution.pdf')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for i, imp_all in enumerate(impulses):
        imp = imp_all[mask[i]]
        if i == 0:
            label = 'Search bias$={:.0f}$ keV/c'.format(errs[i]*to_keV)
        else:
            label = '$\\sigma_p = {:.0f}$ keV/c'.format(errs[i]*to_keV)
        imp_cal = np.abs(imp)*to_keV
        counts, bins = np.histogram(imp_cal, bins=np.linspace(0, 3000, 100))
        counts = counts * 50/(bins[1] - bins[0])
        bins = (bins[:-1] + bins[1:])/2.
        p0 = (100, np.mean(imp_cal), np.std(imp_cal))
        try:
            p, _ = curve_fit(gaus, bins, counts, p0=p0)
        except RuntimeError:
            p = p0
        plot_bins = np.linspace(bins[0], bins[-1], 1000)
        ax.errorbar(bins, counts, np.sqrt(counts), color='C' + str(i), ls='none', marker='.', ms=4)
        ax.plot(plot_bins, gaus(plot_bins, *p), color='C' + str(i), label=label)
    ax.set_xlim([0, means[-1]*to_keV + 3*errs[-1]*to_keV])
    ax.set_xlabel('Impulse amplitude (keV/c)')
    ax.set_ylabel('Counts/(50 keV/c)')
    ax.grid()
    ax.legend(ncol=len(impulses)//3, fontsize=10)
    ax.set_title(f'Impulse calibration for a {config["d_sphere_nm"]:.0f} nm sphere')
    fig.savefig(plot_path + '_recon_cal.pdf')
    plt.close(fig)

    resolutions = errs*to_keV
    for pulse, res in zip(pulse_amps_keV, resolutions):
        print('{:.0f} keV/c impulse:\t {:.1f} keV resolution'.format(pulse, res))

    print('----------------------------------------------')
    print('Mean resolution:\t {:.1f} keV'.format(np.mean(resolutions[2:])))

    # Calibrated waveforms plot
    fig, ax = plt.subplots(mean_pulses_true_imp.shape[0], figsize=(6, 3*mean_pulses_true_imp.shape[0]), \
                       sharex=True, sharey=True, layout='constrained')
    for i in range(mean_pulses_true_imp.shape[0]):
        ax[i].plot(time_slice*1e6, to_keV*pulses_true_imp[i].T, lw=0.2, alpha=0.1, color='k')
        ax[i].plot(time_slice*1e6, to_keV*mean_pulses_true_imp[i], marker='.')
        ax[i].axhline(-pulse_amps_keV[i], color='k', ls='--')
        ax[i].set_ylabel('Amplitude [keV/c]')
        ax[i].set_title('{:.0f} keV/c impulse'.format(pulse_amps_keV[i]))
        ax[i].grid(which='both')
    ax[-1].set_xlabel(r'Time [$\mu$s]')
    fig.suptitle('Average waveforms after calibration')
    fig.savefig(plot_path + '_avg_wfms_cal.pdf')
    plt.close(fig)

    # Pulse shape comparison
    colors_viridis = [plt.get_cmap('viridis', len(pulse_amps_keV))(i) for i in range(len(pulse_amps_keV))]
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for i, mp in enumerate(mean_pulses_true_imp):
        if i == 0:
            continue
        ax.plot(time_slice*1e6, to_keV*mp/pulse_amps_keV[i], alpha=0.5, color=colors_viridis[i],
                label='{:.0f} keV/c'.format(pulse_amps_keV[i]))
    ax.plot(time_slice*1e6, np.mean(to_keV*mean_pulses_true_imp[1:]/pulse_amps_keV[1:, None], axis=0),
            color='k', lw=1.5, label='Average')
    ax.set_xlabel(r'Time [$\mu$s]')
    ax.set_ylabel('Relative amplitude')
    ax.set_title('Pulse shapes for all amplitudes')
    ax.legend(ncol=len(pulse_amps_keV)//4, columnspacing=1., fontsize=10)
    ax.grid(which='both')
    fig.savefig(plot_path + '_pulse_shapes.pdf')
    plt.close(fig)

    pulse_1_keV = -to_keV*mean_pulses_true_imp[-1]/pulse_amps_keV[-1]
    resolution = np.sqrt(np.mean(resolutions[2:]**2))

    # Chi2 vs impulse plot
    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ndof = len(pulse_1_keV) - 1

    x_min, x_max = 0, 1800
    y_min, y_max = 1e-1, 1e1
    nx, ny = 50, 50

    xc = np.linspace(x_min, x_max, nx)
    yc_log = np.linspace(np.log10(y_min), np.log10(y_max), ny)
    X, YLOG = np.meshgrid(xc, yc_log)

    xvals, yvals = [], []

    for i, (pu, im) in enumerate(zip(pulses_true_imp, impulses)):
        chi2 = np.nansum(
            (to_keV * pu - to_keV * im[:, None] * pulse_1_keV[None, :])**2
            / resolution**2,
            axis=1,
        )

        x = to_keV * np.abs(im)
        y = chi2 / ndof

        xvals.append(np.mean(x))
        yvals.append(np.mean(y))

        m = np.isfinite(x) & np.isfinite(y) & (y > 0)
        x = x[m]
        y = y[m]
        if x.size < 20:
            continue

        logy = np.log10(y)
        data = np.vstack([x, logy])
        kde = stats.gaussian_kde(data, bw_method="scott")
        grid = np.vstack([X.ravel(), YLOG.ravel()])
        Z = kde(grid).reshape(YLOG.shape)

        dx = xc[1] - xc[0]
        dy = yc_log[1] - yc_log[0]
        pdf = Z * dx * dy

        pdf_flat = pdf.ravel()
        if not np.any(pdf_flat > 0):
            continue

        idx = np.argsort(pdf_flat)[::-1]
        cdf = np.cumsum(pdf_flat[idx])

        levels_pdf = []
        for frac in (0.68, 0.95):
            j = min(np.searchsorted(cdf, frac), len(idx) - 1)
            levels_pdf.append(pdf_flat[idx][j])

        l1, l2 = sorted(levels_pdf)
        levels_fill = [l1, l2, pdf.max()]

        Y = 10**YLOG
        ax.contourf(X, Y, pdf, levels=levels_fill, colors=[f"C{i}", f"C{i}"], alpha=[0.25, 0.75])

    xvals = np.asarray(xvals)
    yvals = np.asarray(yvals)

    try:
        p_quad, _ = curve_fit(quad, xvals, yvals)
        xplot = np.linspace(x_min, x_max, 100)
        ax.plot(xvals, yvals, color='k', ls='none', marker='.')
        ax.plot(xplot, quad(xplot, *p_quad), '--k')
    except:
        ax.plot(xvals, yvals, color='k', ls='none', marker='.')

    ax.set_yscale("log")
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel("Impulse [keV/c]")
    ax.set_ylabel(r"$\chi^2 / \mathrm{ndof}$")
    ax.set_title('Goodness of fit for calibration data')
    fig.savefig(plot_path + '_chi2_vs_imp.pdf')
    plt.close(fig)

    # Save config
    config['keV_per_N'] = float(to_keV)
    config['template'] = [float(i) for i in pulse_1_keV]
    config['resolution'] = float(resolution)
    config['meters_per_volt'] = float(meters_per_volt)

    config_path = 'configs/' + base_path.split('pulse_calibration/')[-1] + dataset
    os.makedirs(config_path, exist_ok=True)

    with open(config_path + '/config.yaml', 'w') as f:
        yaml.safe_dump(config, f)

Parallel(n_jobs=cpu_count())(delayed(process_dataset)(dataset_ind) for dataset_ind in range(len(datasets)))
