import numpy as np
from glob import glob
import os
import gc
from pathlib import Path
import yaml
import gc

from matplotlib import pyplot as plt
from cycler import cycler
from matplotlib.backends.backend_pdf import PdfPages

from scipy.optimize import curve_fit
from scipy import stats

from joblib import Parallel, delayed, cpu_count

import sys
sys.path.insert(0, '/Users/clarke/Code/nanospheres/')
import data_processing as dp

plt.style.use('thesis')
plt.rcParams.update({'figure.dpi': 150})
plt.rcParams.update({'axes.prop_cycle': cycler(color=['#5994cd', '#d74164', '#4eaa76', '#d88300', '#7365cf', \
                                                      '#c85e3e', '#83a23e', '#c851b1', '#1850a1'])})

max_files = 300

base_path = '/Users/clarke/Data/gas_collisions/pulse_calibration/sphere_20251212/'
drive_path = '/Users/clarke/Data/'

if not os.path.exists(drive_path):
    print('Error: check that the external drive is plugged in!')

folders = glob(base_path + '*')

cal_datasets = {}

for folder in folders:
    all_items = glob(folder + '/*')
    subfolders = np.unique(['_'.join(s.split('_')[:-1]) for s in all_items])
    sub_datasets = []
    for subfolder in subfolders:
        sub_datasets.append(subfolder.split(base_path)[-1].split('/')[-1])
    cal_datasets[folder.split(base_path)[-1]] = sub_datasets

dataset_ind = 3
dataset = list(cal_datasets.keys())[dataset_ind]

with open(base_path + dataset + '/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['calibrate'] = False

template = np.array(config['template'])
resolution = config['resolution']

for key in config.keys():
    print(key + ':\t ', config[key])

# get the background datasets
base_path = '/Users/clarke/Data/gas_collisions/background_data/sphere_20251212/'

folders = glob(base_path + '*')

bkg_datasets = {}

for folder in folders:
    print(folder.split(base_path)[-1])
    all_items = glob(folder + '/*')
    subfolders = np.unique(['_'.join(s.split('_')[:-1]) for s in all_items])
    sub_datasets = []
    for subfolder in subfolders:
        sub_datasets.append(subfolder.split(base_path)[-1].split('/')[-1])
        print('\t' + subfolder.split(base_path)[-1].split('/')[-1])
    bkg_datasets[folder] = sub_datasets

# get the xenon datasets
base_path = '/Users/clarke/Data/gas_collisions/xenon_data/sphere_20251212/'

folders = glob(base_path + '*')

xe_datasets = {}

for folder in folders:
    print(folder.split(base_path)[-1])
    all_items = glob(folder + '/*')
    subfolders = np.unique(['_'.join(s.split('_')[:-1]) for s in all_items])
    sub_datasets = []
    for subfolder in subfolders:
        sub_datasets.append(subfolder.split(base_path)[-1].split('/')[-1])
        print('\t' + subfolder.split(base_path)[-1].split('/')[-1])
    xe_datasets[folder] = sub_datasets

# get the krypton datasets
base_path = '/Users/clarke/Data/gas_collisions/krypton_data/sphere_20251212/'

folders = glob(base_path + '*')

kr_datasets = {}

for folder in folders:
    print(folder.split(base_path)[-1])
    all_items = glob(folder + '/*')
    subfolders = np.unique(['_'.join(s.split('_')[:-1]) for s in all_items])
    sub_datasets = []
    for subfolder in subfolders:
        sub_datasets.append(subfolder.split(base_path)[-1].split('/')[-1])
        print('\t' + subfolder.split(base_path)[-1].split('/')[-1])
    kr_datasets[folder] = sub_datasets

dataset_dict = bkg_datasets | xe_datasets | kr_datasets

def half_gaus(x, A, sigma):
    return A*np.exp(-x**2/2/sigma**2)

def process_dataset(dataset_ind):

    dataset = list(dataset_dict.keys())[dataset_ind]

    plot_path = 'figures/' + dataset.split(drive_path)[-1] + '/' + dataset_dict[dataset][0]

    print('Starting dataset ' + dataset + '/' + dataset_dict[dataset][0])

    nd = dp.NanoDataset(dataset + '/' + dataset_dict[dataset][0], plot_path, verbose=True, max_files=max_files, \
                        config=config, ds_factor=10)
    nd.load_search_data()
    nd.save_to_hdf5()

    noise_thresh = 150 # keV

    noise_cut = (nd.impulse_rms < noise_thresh) & (nd.impulse_rms > 10)

    # also cut based on the fit to the resonance
    res_cut = nd.resonance_params[:, 2] < 2*np.pi*100

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.semilogy(nd.pulse_times, nd.impulse_rms)
    ax.axhline(noise_thresh, ls='--', color='C1', label='{:.0f} keV noise cut ({:.1f}% surviving)'\
                                                        .format(noise_thresh, 100*sum(noise_cut)/len(noise_cut)))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('RMS noise [keV]')
    ax.set_ylim([1e1, 1e4])
    ax.legend(loc='upper right')
    fig.savefig(plot_path + '_noise_cut.pdf')
    fig.clf()
    plt.close()
    del fig, ax
    gc.collect()

    z_cut = 2
    ndof = len(template) - 1
    alpha = (1 - stats.norm.cdf(z_cut))*2

    pvals = 1. - stats.chi2.cdf(nd.chi2, df=ndof)
    chi2_cut = np.zeros(nd.chi2.shape, dtype=bool)
    chi2_cut[pvals > alpha] = True

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    bins_y = np.linspace(0, 300, 200)
    bins_x = np.linspace(0, 1000, 200)
    h = ax.hist2d(np.abs(nd.impulses), nd.impulse_rms, bins=(bins_x, bins_y), cmin=1, cmax=1e5)
    ax.axhline(noise_thresh, ls='--', color='red', label='{:.0f} keV noise cut'.format(noise_thresh))
    ax.set_xlabel('Impulse [keV/c]')
    ax.set_ylabel('RMS noise [keV]')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_title('RMS noise cut')
    fig.colorbar(h[3], ax=ax, label='Counts', pad=0.01)
    fig.savefig(plot_path + '_noise_2d_hist.pdf')
    fig.clf()
    plt.close()
    del fig, ax
    gc.collect()

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    bins_y = np.logspace(np.log10(2e-2), 1, 200)
    bins_x = np.linspace(0, 700, 200)
    h = ax.hist2d(np.abs(nd.impulses), nd.chi2/ndof, bins=(bins_x, bins_y), cmin=1, cmax=1e5)
    ax.axhline(stats.chi2.ppf(1 - alpha, ndof)/ndof, ls='--', color='red', label=r'{:.0f}$\sigma$ goodness-of-fit cut'.format(z_cut))
    ax.set_yscale('log')
    ax.set_xlabel('Impulse [keV/c]')
    ax.set_ylabel(r'Reduced $\chi^2$')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_title('Goodness of fit cut')
    fig.colorbar(h[3], ax=ax, label='Counts', pad=0.01)
    fig.savefig(plot_path + '_chi2_2d_hist.pdf')
    fig.clf()
    plt.close()
    del fig, ax
    gc.collect()

    bin_edges = np.linspace(0, 800, 101)

    counts_all, _ = np.histogram(nd.impulses, bins=bin_edges)
    counts_acc, _ = np.histogram(nd.impulses[chi2_cut & noise_cut], bins=bin_edges)

    bins = (bin_edges[1:] + bin_edges[:-1])/2.

    ppb = len(nd.impulses)/len(bins)
    exp = np.floor(np.log10(ppb))
    coeff = 5*np.ceil(ppb/10**exp)

    fit_ind = int(1.5*np.argmax(counts_all))

    p, _ = curve_fit(half_gaus, bins[fit_ind:], counts_acc[fit_ind:], p0=(ppb, np.std(nd.impulses[~np.isnan(nd.impulses)])))

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.step(bins, counts_all, where='mid', label='All events')
    ax.step(bins, np.histogram(nd.impulses[~res_cut], bins=bin_edges)[0], label='Rejected by fit cut')
    ax.step(bins, np.histogram(nd.impulses[~noise_cut & res_cut], bins=bin_edges)[0], label='Rejected by noise cut')
    ax.step(bins, np.histogram(nd.impulses[~chi2_cut & noise_cut & ~res_cut], bins=bin_edges)[0], label=r'Rejected by $\chi^2$ cut')
    ax.step(bins, counts_acc, where='mid', label='Passing all cuts')
    ax.plot(bins, half_gaus(bins, *p), label='$\\Delta p={:.1f}$ keV/c'.format(np.abs(p[1])))
    ax.set_xlim([0, 1.05*np.amax(bins)])
    ax.set_ylim([5e-1, coeff*10**exp])
    ax.set_xlabel('Impulse [keV/c]')
    ax.set_ylabel('Counts')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(plot_path + '_impulse_spectra.pdf')
    fig.clf()
    plt.close()
    del fig, ax
    gc.collect()

print(f'Running on {cpu_count()} CPUs')

Parallel(n_jobs=cpu_count()//2)(delayed(process_dataset)(dataset_ind)for dataset_ind in range(len(dataset_dict)))