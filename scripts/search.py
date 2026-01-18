import argparse
import numpy as np
from glob import glob
import os
import gc
from pathlib import Path
import yaml
import gc

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from scipy.optimize import curve_fit
from scipy import stats

from joblib import Parallel, delayed, cpu_count

import sys
sys.path.insert(0, '/Users/clarke/Code/nanospheres/')
import data_processing as dp

plt.style.use('clarke-default')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Search script for nanosphere impulse sensor')
parser.add_argument('--max-files', type=int, default=300,
                    help='Maximum number of files to process per dataset (default: 300)')
parser.add_argument('--drive-path', type=str, default='/Users/clarke/Data/',
                    help='Path to data drive (default: /Users/clarke/Data/)')
parser.add_argument('--sphere-date', type=str, default='20251212',
                    help='Sphere date identifier (default: 20251212)')
parser.add_argument('--config-path', type=str, default='',
                    help='Path to the config file to be used')
args = parser.parse_args()

max_files = args.max_files
drive_path = args.drive_path
sphere_date = args.sphere_date
config_path = args.config_path

if not os.path.exists(drive_path):
    print('Error: check that the external drive is plugged in!')

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config['calibrate'] = False

template = np.array(config['template'])
resolution = config['resolution']

for key in config.keys():
    print(key + ':\t ', config[key])

# get the background datasets
base_path = f'{drive_path}/gas_collisions/background_data/sphere_{sphere_date}/'

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
base_path = f'{drive_path}/gas_collisions/xenon_data/sphere_{sphere_date}/'

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
base_path = f'{drive_path}/gas_collisions/krypton_data/sphere_{sphere_date}/'

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
    # Apply style in worker process (not inherited from main process)
    plt.style.use('clarke-default')

    dataset = list(dataset_dict.keys())[dataset_ind]

    plot_path = 'figures/' + dataset.split(drive_path)[-1] + '/' + dataset_dict[dataset][0]

    print('Starting dataset ' + dataset + '/' + dataset_dict[dataset][0])

    nd = dp.NanoDataset(dataset + '/' + dataset_dict[dataset][0], plot_path, verbose=True, max_files=max_files, \
                        config=config, ds_factor=10)
    nd.load_search_data()
    nd.save_to_hdf5()

    noise_thresh = 80 if config['d_sphere_nm'] < 150 else 150 # keV

    noise_mask = (nd.impulse_rms < noise_thresh) & (nd.impulse_rms > 10)

    # also cut based on the fit to the resonance
    res_mask = nd.resonance_params[:, 2] < 2*np.pi*100

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.semilogy(nd.pulse_times, nd.impulse_rms)
    ax.axhline(noise_thresh, ls='--', color='C1', label='{:.0f} keV noise cut ({:.1f}% surviving)'\
                                                        .format(noise_thresh, 100*sum(noise_mask)/len(noise_mask)))
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('RMS noise [keV]')
    ax.set_ylim([1e0, 1e4])
    ax.legend(loc='upper right')
    ax.grid(which='both')
    fig.savefig(plot_path + '_noise_cut.pdf')
    fig.clf()
    plt.close()
    del fig, ax
    gc.collect()

    chi2_cut = 1.
    ndof = len(template) - 1

    chi2_mask = np.zeros(nd.chi2.shape, dtype=bool)
    chi2_mask[nd.chi2 > chi2_cut] = True

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    bins_y = np.linspace(0, 300, 200)
    bins_x = np.linspace(0, 1000, 200)
    h = ax.hist2d(np.abs(nd.impulses), nd.impulse_rms, bins=(bins_x, bins_y), norm='log')
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
    h = ax.hist2d(np.abs(nd.impulses), nd.chi2/ndof, bins=(bins_x, bins_y), norm='log')
    ax.axhline(chi2_cut, ls='--', color='red', label='Goodness-of-fit cut')
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
    counts_acc, _ = np.histogram(nd.impulses[chi2_mask & noise_mask], bins=bin_edges)

    bins = (bin_edges[1:] + bin_edges[:-1])/2.
    bin_width = bin_edges[1] - bin_edges[0]  # keV/c per bin

    # Compute total observation time: number of search windows * search window duration
    total_time = len(nd.impulses) * nd.search_window  # seconds

    # Scale to rate per 50 keV/c per second
    scale_factor = (50. / bin_width) / total_time
    rate_all = counts_all * scale_factor
    rate_acc = counts_acc * scale_factor

    fit_ind = int(1.5*np.argmax(counts_all))

    p, _ = curve_fit(half_gaus, bins[fit_ind:], rate_acc[fit_ind:], p0=(rate_acc.max(), np.std(nd.impulses[~np.isnan(nd.impulses)])))

    fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
    ax.step(bins, rate_all, where='mid', label='All events')
    ax.step(bins, np.histogram(nd.impulses[~res_mask], bins=bin_edges)[0] * scale_factor, label='Rejected by fit cut')
    ax.step(bins, np.histogram(nd.impulses[~noise_mask & res_mask], bins=bin_edges)[0] * scale_factor, label='Rejected by noise cut')
    ax.step(bins, np.histogram(nd.impulses[~chi2_mask & noise_mask & ~res_mask], bins=bin_edges)[0] * scale_factor, label=r'Rejected by $\chi^2$ cut')
    ax.step(bins, rate_acc, where='mid', label='Passing all cuts')
    ax.plot(bins, half_gaus(bins, *p), '--', label='$\\Delta p={:.1f}$ keV/c'.format(np.abs(p[1])))
    ax.set_xlim([0, 1.05*np.amax(bins)])
    ax.set_ylim([1e-2, 5e4])
    ax.set_xlabel('Impulse [keV/c]')
    ax.set_ylabel('Rate [counts/(50 keV/c)/s]')
    ax.set_yscale('log')
    ax.legend()
    ax.set_title('{:.1f} s livetime'.format(total_time))
    ax.grid(which='both')
    fig.savefig(plot_path + '_impulse_spectra.pdf')
    fig.clf()
    plt.close()
    del fig, ax
    gc.collect()

print(f'Running on {cpu_count()} CPUs')

Parallel(n_jobs=cpu_count()//2)(delayed(process_dataset)(dataset_ind)for dataset_ind in range(len(dataset_dict)))