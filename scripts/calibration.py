import numpy as np
from glob import glob
import os
import gc
from pathlib import Path
import yaml

from matplotlib import pyplot as plt
from cycler import cycler
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

plt.style.use('thesis')
plt.rcParams.update({'figure.dpi': 150})
plt.rcParams.update({'axes.prop_cycle': cycler(color=['#5994cd', '#d74164', '#4eaa76', '#d88300', '#7365cf', \
                                                      '#c85e3e', '#83a23e', '#c851b1', '#1850a1'])})

data_type = 'gas_collisions'
sphere_date = '20251212'
max_files = 25

base_path = f'/Users/clarke/Data/{data_type}/pulse_calibration/sphere_{sphere_date}/'
drive_path = '/Users/clarke/Data/'

print(base_path)

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
    'calibrate': True
}

def gaus(x, A, mu, sigma):
    return A*np.exp(-(x - mu)**2/2/sigma**2)

def linear(x, m, b):
    return m*x + b

def to_keV(x, m, b):
    return (x - b)/m

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
    pulses = []
    recon_inds = []
    timestamps = []
    pulse_times = []
    meters_per_volt = []

    # loop through all four files in that dataset in the correct order
    for i, file_ind in enumerate(amp_list):
        filename = datasets[list(datasets.keys())[dataset_ind]][file_ind]
        plot_path = 'figures/' + base_path.split(drive_path)[-1] + dataset + '/' + filename

        nd = dp.NanoDataset(base_path + dataset + '/' + filename, plot_path, verbose=True, max_files=max_files, config=config)
        nd.load_calibration_data(global_params=False, pulse_amps_1e=pulse_amps_1e, pulse_amps_V=pulse_amps_V, noise=i==0)
        impulses.append(nd.impulses.copy())
        pulse_amps_keV.append([nd.pulse_amp_keV.copy(), 0][i==0])
        resonance_params.append(nd.resonance_params.copy())
        pulses.append(nd.pulses.copy())
        recon_inds.append(nd.recon_impulse_inds.copy())
        pulse_times.append(nd.pulse_times.copy())
        timestamps.append(nd.timestamps.copy())
        meters_per_volt.append(nd.meters_per_volt.copy())

        if i == 0:
            freqs = np.copy(nd.freqs)
        del nd

    pulse_amps_keV = np.array(pulse_amps_keV)
    min_len = np.amin([len(l) for l in impulses])
    impulses = np.array([l[:min_len] for l in impulses])
    pulses = np.array([l[:min_len] for l in pulses])
    recon_inds = np.array([l[:min_len] for l in recon_inds])
    resonance_params = np.array([l[:min_len] for l in resonance_params])
    pulse_times = np.array([l[:min_len] for l in pulse_times])
    timestamps = np.array([l[:min_len] for l in timestamps])
    meters_per_volt = np.mean(meters_per_volt)

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
                np.log10(np.amax(res_evol[:,2]/2/np.pi)), 20), \
                histtype='step', orientation='horizontal')
    ax[0,0].set_ylabel(r'$A$ [$\mathrm{V^2\,Hz^3}$]')
    ax[1,0].set_ylabel(r'$\omega_0/2\pi$ [$\mathrm{s}^{-1}$]')
    ax[2,0].set_ylabel(r'$\gamma/2\pi$ [$\mathrm{s}^{-1}$]')
    ax[2,0].set_xlabel('Time since start [s]')
    ax[2,1].set_xlabel('Counts')
    fig.suptitle('Resonance parameter drift')
    fig.savefig(plot_path + '_res_evol.pdf')

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

    # build a time array
    half_window = 250
    peak_ind = np.argmax(np.abs(np.mean(pulses[-1], axis=0)))
    times = np.arange(0, pulses.shape[-1]/2./freqs[-1], 1./2./freqs[-1])
    times -= times[peak_ind]
    time_slice = times[..., peak_ind - half_window:peak_ind + half_window]

    # get the cropped waveforms centered around the true impulse time
    pulses_true_imp = pulses[..., peak_ind - half_window:peak_ind + half_window]
    mean_pulses_true_imp = np.mean(pulses_true_imp, axis=1)

    # get the cropped waveforms centered around the reconstructed impulse time
    offsets = np.arange(2*half_window) - half_window
    time_inds = recon_inds[..., None] + offsets

    fig, ax = plt.subplots(mean_pulses_true_imp.shape[0], figsize=(6, 3*mean_pulses_true_imp.shape[0]), \
                        sharex=True, sharey=True, layout='constrained')
    for i in range(mean_pulses_true_imp.shape[0]):
        ax[i].plot(time_slice*1e6, pulses_true_imp[i].T*1e-8, lw=0.2, alpha=0.1, color='k')
        ax[i].plot(time_slice*1e6, mean_pulses_true_imp[i]*1e-8)
        ax[i].set_ylabel('Amplitude [$10^8$ au]')
        ax[i].set_title('{:.0f} keV/c impulse'.format(pulse_amps_keV[i]))
    ax[-1].set_xlabel(r'Time [$\mu$s]')
    fig.suptitle('Average waveforms')
    fig.savefig(plot_path + 'avg_wfms_uncal.pdf')

    means = []
    errs = []

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for i, imp in enumerate(impulses):
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
    # ax.set_xlim([0, 250])
    ax.set_title('Reconstructed impulses before calibration')
    ax.grid()
    fig.savefig(plot_path + '_recon_uncal.pdf')

    means = np.array(means)
    errs = np.array(errs)

    exclude_first = 2 # how many of the first datasets to exclude from the calibration line fit

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    plot_amps = np.linspace(0, np.amax(pulse_amps_keV), 2)
    cal_params, _ = curve_fit(linear, pulse_amps_keV[exclude_first:], means[exclude_first:])
    ax.errorbar(pulse_amps_keV[exclude_first:], to_keV(means[exclude_first:], *cal_params), \
                to_keV(errs[exclude_first:], cal_params[0], 0), marker='.', ls='none', label='Calibration data')
    ax.errorbar(pulse_amps_keV[:exclude_first], to_keV(means[:exclude_first], *cal_params), \
                to_keV(errs[:exclude_first], cal_params[0], 0), marker='.', ls='none', label='Excluded points')
    ax.plot(plot_amps, plot_amps, label='$y=x$')
    # ax.text(0.95, 0.05, '${}x+{}$'.format(*cal_params), ha='right', va='bottom', transform=ax.transAxes)
    ax.set_xlabel('True impulse [keV/c]')
    ax.set_ylabel('Reconstructed impulse [au]')
    ax.set_title('Accuracy of impulse calibration')
    ax.legend()
    ax.grid()
    fig.savefig(plot_path + '_calibration.pdf')

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.plot(pulse_amps_keV, to_keV(errs, cal_params[0], 0), marker='o', ls='none', fillstyle='none')
    ax.set_xlabel('True impulse [keV/c]')
    ax.set_ylabel('Resolution [keV/c]')
    ax.set_title('Resolution as a function of impulse')
    ax.grid()
    fig.savefig(plot_path + '_resolution.pdf')

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for i, imp in enumerate(impulses):
        if i == 0:
            label = 'Search bias$={:.0f}$ keV/c'.format(errs[i]/cal_params[0])
        else:
            label = '$\\sigma_p = {:.0f}$ keV/c'.format(errs[i]/cal_params[0])
        imp_cal = np.abs(imp)/cal_params[0] - cal_params[1]/cal_params[0]
        counts, bins = np.histogram(imp_cal, bins=np.linspace(0, 3000, 100))
        counts = counts * 50/(bins[1] - bins[0])
        bins = (bins[:-1] + bins[1:])/2.
        p0 = (100, np.mean(imp_cal), np.std(imp_cal))
        try:
            p, _ = curve_fit(gaus, bins, counts, p0=p0)
        except RuntimeError:
            p = p0
        plot_bins = np.linspace(bins[0], bins[-1], 1000)
        # ax.step(bins, counts, color='C' + str(i - 1), alpha=0.5, where='mid')
        ax.errorbar(bins, counts, np.sqrt(counts), color='C' + str(i), ls='none', marker='.', ms=4)
        ax.plot(plot_bins, gaus(plot_bins, *p), color='C' + str(i), label=label)
    ax.set_xlim([0, means[-1]/cal_params[0] + 3*errs[-1]/cal_params[0]])
    ax.set_xlabel('Impulse amplitude (keV/c)')
    ax.set_ylabel('Counts/(50 keV/c)')
    ax.grid()
    ax.legend(ncol=len(impulses)//3, fontsize=10)
    ax.set_title('Reconstructed impulses after calibration')
    fig.savefig(plot_path + '_recon_cal.pdf')

    resolutions = errs/cal_params[0]
    for pulse, res in zip(pulse_amps_keV, resolutions):
        print('{:.0f} keV/c impulse:\t {:.1f} keV resolution'.format(pulse, res))

    print('----------------------------------------------')
    print('Mean resolution:\t {:.1f} keV'.format(np.mean(resolutions[2:])))

    fig, ax = plt.subplots(mean_pulses_true_imp.shape[0], figsize=(6, 3*mean_pulses_true_imp.shape[0]), \
                       sharex=True, sharey=True, layout='constrained')
    for i in range(mean_pulses_true_imp.shape[0]):
        ax[i].plot(time_slice*1e6, to_keV(pulses_true_imp[i].T, *cal_params), lw=0.2, alpha=0.1, color='k')
        ax[i].plot(time_slice*1e6, to_keV(mean_pulses_true_imp[i], *cal_params))
        ax[i].axhline(-pulse_amps_keV[i], color='k', ls='--')
        ax[i].set_ylabel('Amplitude [keV/c]')
        ax[i].set_title('{:.0f} keV/c impulse'.format(pulse_amps_keV[i]))
        ax[i].grid(which='both')
    ax[-1].set_xlabel(r'Time [$\mu$s]')
    fig.suptitle('Average waveforms after calibration')
    fig.savefig(plot_path + '_avg_wfms_cal.pdf')

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    for i, mp in enumerate(mean_pulses_true_imp):
        if i == 0:
            continue
        ax.plot(time_slice*1e6, to_keV(mp, *cal_params)/pulse_amps_keV[i], alpha=0.3, label='{:.0f} keV/c'.format(pulse_amps_keV[i]))
    ax.plot(time_slice*1e6, np.mean(to_keV(mean_pulses_true_imp[1:], *cal_params)/pulse_amps_keV[1:, None], axis=0), color='k', label='Average')
    ax.set_xlabel(r'Time [$\mu$s]')
    ax.set_ylabel('Relative amplitude')
    ax.set_title('Pulse shapes for all amplitudes')
    ax.legend(ncol=len(pulse_amps_keV)//4, columnspacing=1., fontsize=10)

    pulse_1_keV = -to_keV(mean_pulses_true_imp[-1], *cal_params)/pulse_amps_keV[-1]

    config['cal_factors'] = [float(1/cal_params[0]), float(-cal_params[1]/cal_params[0])]
    config['template'] = [float(i) for i in pulse_1_keV]
    config['resolution'] = float(np.sqrt(np.mean(resolutions[2:]**2)))
    config['meters_per_volt'] = float(meters_per_volt)

    with open(base_path + dataset + '/config.yaml', 'w') as f:
        yaml.safe_dump(config, f)

Parallel(n_jobs=cpu_count())(delayed(process_dataset)(dataset_ind) for dataset_ind in range(len(datasets)))