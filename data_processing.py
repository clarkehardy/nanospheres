from zlib import Z_BEST_COMPRESSION
import h5py
import numpy as np
from glob import glob
from pathlib import Path
import gc
import warnings

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.integrate import trapezoid

def abs_susc2(omega, A, omega_0, gamma):
    """Magnitude of the mechanical susceptibility of the trapped nanosphere squared.

    :param omega: array of angular frequencies
    :type omega: numpy.ndarray
    :param A: amplitude of the susceptibility peak
    :type A: float
    :param omega_0: resonant angular frequency
    :type omega_0: float
    :param gamma: resonance peak width
    :type gamma: float
    :return: squared magnitude of the susceptibility of the trapped nanosphere at each angular frequency
    :rtype: numpy.ndarray
    """
    if (gamma < 0) or (A < 0):
        return -np.inf
    return A**2*np.abs(1/(omega_0**2 - omega**2 - 1j*gamma*omega))**2

def susc(omega, A, omega_0, gamma):
    """Complex mechanical susceptibility of the trapped nanosphere.

    :param omega: array of angular frequencies
    :type omega: numpy.ndarray
    :param A: amplitude of the susceptibility peak
    :type A: float
    :param omega_0: resonant angular frequency
    :type omega_0: float
    :param gamma: resonance peak width
    :type gamma: float
    :return: susceptibility of the trapped nanosphere at each angular frequency
    :rtype: numpy.ndarray
    """
    if gamma < 0:
        return -np.inf
    return A/(omega_0**2 - omega**2 - 1j*gamma*omega)

class NanoFile:
    """Class to handle an individual file containing nanosphere data.
    """

    def __init__(self, file_path, f_cutoff=[2e4, 1e5], t_window=1e-3, verbose=False):
        """Initializes a NanoFile object containing data from a single HDF5 file.

        :param file_path: path to the HDF5 file to load
        :type file_path: string
        :param f_cutoff: upper and lower cutoff frequencies, defaults to [2e4, 1e5]
        :type f_cutoff: list, optional
        :param t_window: pulse window in seconds, defaults to 1e-3
        :type t_window: float, optional
        :param verbose: whether to print verbose output, defaults to False
        :type verbose: bool, optional
        """
        self.file_path = file_path
        charge_p = file_path.split('_p')[-1].split('e_')[0]
        charge_n = file_path.split('_n')[-1].split('e_')[0]
        self.n_charges = float([charge_p, charge_n][len(charge_p) > len(charge_n)])
        self.pulse_amp_V = float(file_path.split('v_')[0].split('_')[-1])
        self.f_cutoff = f_cutoff
        self.t_window = t_window
        self.search_window = 5e-4 # search in a 50 us window
        self.noise_floor = 0.
        self.window_func = lambda N: sig.windows.hann(N) #sig.windows.tukey(N, 0.05)
        # pulse height is undefined until the calibration is done
        self.pulse_amp_keV = 1.
        self.verbose = verbose
        self.load_file()

    def calibrate_pulse_amp(self, pulse_amps_1e, pulse_amps_V):
        """Loads the calibration factors to convert pulse amplitudes into keV/c.

        :param pulse_amps_1e: amplitude of pulses in keV/c assuming a 1e charge
        :type pulse_amps_1e: list
        :param pulse_amps_V: amplitude of pulses delivered in volts
        :type pulse_amps_V: list
        """
        self.pulse_amp_keV = pulse_amps_1e[np.argmin(np.abs(pulse_amps_V - self.pulse_amp_V))]*self.n_charges

    def get_force_array(self, impulse_inds, global_params=True, file_num=0, pdfs=None, search=True):
        """Computes the force at each impulse in the data and returns the results in an array.

        :param impulse_inds: indices at which impulses were applied
        :type impulse_inds: numpy.ndarray
        :param global_params: whether to use global resonance parameters (vs pulse-by-pulse)
        :type global_params: bool, optional
        :param file_num: file number, defaults to 0
        :type file_num: int, optional
        :param pdfs: PDFs in which to save the figures, defaults to None
        :type pdfs: string, optional
        :param search: whether to search for the force peak in a window, defaults to True
        :type search: bool, optional
        """
        time_domain_pdf, freq_domain_pdf, optimal_filter_pdf, res_fit_pdf, impulse_win_pdf = pdfs
        forces = []
        resonance_params = []
        for i, ind in enumerate(impulse_inds):
            if self.verbose:
                print('    -> Computing force for impulse at t={:.5f} seconds...'.format(self.times[ind]))
            times_win, z_raw_win, z_filt_win = self.get_impulse_window(ind, file_num, i, impulse_win_pdf)
            p, success = self.fit_susceptibility(z_filt_win, file_num, i, res_fit_pdf)
            if not success:
                print(f'Fit for impulse {i + 1} failed! Skipping')
                continue
            resonance_params.append(p)
            forces.append(self.compute_force(times_win, z_filt_win, file_num, i, [p, None][int(global_params)], \
                                             time_domain_pdf, freq_domain_pdf, optimal_filter_pdf, search))
        
        self.resonance_params = np.array(resonance_params)
        self.forces = np.array(forces)

    def load_file(self):
        """Load data from the HDF5 file and do some preliminary processing.
        """

        with h5py.File(self.file_path, 'r') as f:
            self.z_raw = np.array(f['data/channel_d'])*f['data/channel_d'].attrs['adc2mv']*1e-3
            self.imp_raw = np.array(f['data/channel_g'])*f['data/channel_g'].attrs['adc2mv']*1e-3
            self.mon_raw = np.array(f['data/channel_f'])*f['data/channel_f'].attrs['adc2mv']*1e-3

            self.f_samp = 1./f['data'].attrs['delta_t']

        self.n_samp = len(self.z_raw)
        self.t_int = self.n_samp/self.f_samp
        self.times = np.arange(0, self.t_int, 1/self.f_samp)

    def compute_and_fit_psd(self, file_num=None, pdf=None):
        """Filter the data and compute the power spectral density for the filtered data stream.

        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param pdf: PDF in which to save the spectra figure, defaults to None
        :type pdf: PdfPages, optional
        """

        self.bandpass = sig.butter(3, [self.f_cutoff[0], self.f_cutoff[1]], btype='bandpass', \
                                    output='sos', fs=self.f_samp)
        self.z_filtered = sig.sosfiltfilt(self.bandpass, self.z_raw)

        self.freqs, Pxx_z_raw = sig.welch(self.z_raw, fs=self.f_samp, noverlap=0, nperseg=self.f_samp*0.01)
        _, Pxx_z_filt = sig.welch(self.z_filtered, fs=self.f_samp, noverlap=0, nperseg=self.f_samp*0.01)
        _, Pxx_mon_raw = sig.welch(self.mon_raw, fs=self.f_samp, noverlap=0, nperseg=self.f_samp*0.01)

        # integrate the noise spectrum
        # noise_int = np.sqrt(trapezoid(Pxx_z_filt, self.freqs))

        p = self.fit_voigt_profile(Pxx_z_raw)

        impr_win = 1.5*p[1]/2./np.pi + np.array((0, 1e4))
        impr_inds = [np.argmin(np.abs(n - self.freqs)) for n in impr_win]
        self.noise_floor = np.mean(Pxx_z_raw[impr_inds[0]:impr_inds[1]])

        self.gamma = p[-1]
        self.omega_0 = p[1]

        if pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.semilogy(self.freqs*1e-3, Pxx_z_raw, alpha=1.0, label='$z$ raw')
            ax.semilogy(self.freqs*1e-3, Pxx_z_filt, alpha=0.5, label='$z$ filtered')
            ax.semilogy(self.freqs*1e-3, p[0]*voigt_profile(2*np.pi*self.freqs - p[1], p[2], p[3]) \
                        + self.noise_floor, alpha=0.5, label='Voigt fit')
            ax.semilogy(self.freqs*1e-3, Pxx_mon_raw, alpha=1.0, label='Monitoring raw')
            ax.set_title('{:.0f} keV impulse, file {}'.format(self.pulse_amp_keV, file_num + 1))
            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel(r'PSD [$\mathrm{V^2/Hz}$]')
            ax.set_ylim([1e-16, 1e-4])
            ax.set_xlim([10, 100])
            ax.legend(loc='upper right')
            ax.grid(which='both')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

    def fit_voigt_profile(self, Pxx_z_filt):
        """Fits a Voigt profile to the spectrum for a full file.

        :param Pxx_z_filt: the filtered PSD for the file
        :type Pxx_z_filt: numpy.ndarray
        :return: the best fit parameters
        :rtype: numpy.ndarray
        """
        f_0_range = [20e3, 100e3]
        f_0_inds = [np.argmin(np.abs(self.freqs - f)) for f in f_0_range]

        omega_0_guess = 2*np.pi*self.freqs[f_0_inds[0] + np.argmax(Pxx_z_filt[f_0_inds[0]:f_0_inds[1]])]
        gamma_guess = 2*np.pi*1e-1
        sigma_guess = 2*np.pi*1e2
        A_guess = 5e-3
        p0 = [A_guess, omega_0_guess, sigma_guess, gamma_guess]

        fit_bw = 2*np.pi*2e4
        fit_inds = (2*np.pi*self.freqs > omega_0_guess - fit_bw/2.) & (2*np.pi*self.freqs < omega_0_guess + fit_bw/2.)

        voigt_scaled = lambda omega, A, omega_0, sigma, gamma: np.log(np.abs(A)*voigt_profile(omega - omega_0, sigma, gamma))
        try:
            p, _ = curve_fit(voigt_scaled, 2*np.pi*self.freqs[fit_inds], np.log(Pxx_z_filt[fit_inds]), p0=p0)
        except:
            print('Voigt profile fit failed!')
            return p0

        return p

    def get_impulse_inds(self, impulse_thresh=0.5, file_num=None, pdf=None):
        """Gets the indices in the time series data at which impulses were applied.

        :param impulse_thresh: threshold in volts, defaults to 1.
        :type impulse_thresh: float, optional
        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param pdf: PDF in which to save the impulse data figure, defaults to None
        :type pdf: PdfPages, optional
        :return: indices at which impulses were applied
        :rtype: numpy.ndarray
        """
        impulse_inds = np.argwhere(self.imp_raw > impulse_thresh)
        impulse_inds = np.hstack((impulse_inds[0, 0], impulse_inds[1:, 0][np.diff(impulse_inds[:, 0]) > 5]))

        if pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.plot(self.times, self.imp_raw)
            ax.set_ylabel('Impulse [V]')
            ax.set_xlabel('Time [s]')
            ax.set_title('{:.0f} keV impulse, file {}'.format(self.pulse_amp_keV, file_num + 1))
            ax.plot(self.times[impulse_inds], self.imp_raw[impulse_inds], ls='none', marker='.')
            ax.grid(which='both')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return impulse_inds

    def get_noise_inds(self, impulse_inds):
        """Gets a set of indices in the time series data between impulses.

        :param impulse_inds: indices of impulses that were applied.
        :type impulse_inds: numpy.ndarray
        :return: a list of indices
        :rtype: numpy.ndarray
        """
        return impulse_inds - int(0.01*np.mean(np.diff(impulse_inds)))

    def get_impulse_window(self, impulse_ind, file_num=None, impulse_num=None, pdf=None):
        """Gets the time window around an impulse.

        :param impulse_ind: index defining the time of the impulse
        :type impulse_ind: int
        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param impulse_num: impulse number, defaults to None
        :type impulse_num: int, optional
        :param pdf: PDF in which the time series data should be saved, defaults to None
        :type pdf: PdfPages, optional
        :return: times, raw z data, and filtered z data around the impulse
        :rtype: tuple
        """
        t_win_pm = int(self.t_window*self.n_samp/self.t_int)

        # if the impulse is within the time window of the start/end of the file,
        # wrap around and pad with noise data from the other end
        if impulse_ind - t_win_pm < 0:
            times_win = np.concat((self.times[impulse_ind - t_win_pm:], self.times[:impulse_ind + t_win_pm]))
            z_raw_win = np.concat((self.z_raw[impulse_ind - t_win_pm:], self.z_raw[:impulse_ind + t_win_pm]))
            z_filt_win = np.concat((self.z_filtered[impulse_ind - t_win_pm:], self.z_filtered[:impulse_ind + t_win_pm]))
        elif impulse_ind + t_win_pm > len(self.times):
            times_win = np.concat((self.times[impulse_ind - t_win_pm:], self.times[:t_win_pm - impulse_ind]))
            z_raw_win = np.concat((self.z_raw[impulse_ind - t_win_pm:], self.z_raw[:t_win_pm - impulse_ind]))
            z_filt_win = np.concat((self.z_filtered[impulse_ind - t_win_pm:], self.z_filtered[:t_win_pm - impulse_ind]))
        else:
            times_win = self.times[impulse_ind - t_win_pm:impulse_ind + t_win_pm]
            z_raw_win = self.z_raw[impulse_ind - t_win_pm:impulse_ind + t_win_pm]
            z_filt_win = self.z_filtered[impulse_ind - t_win_pm:impulse_ind + t_win_pm]

        if pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.plot((times_win - np.mean(times_win))*1e6, z_raw_win, label='Raw')
            ax.plot((times_win - np.mean(times_win))*1e6, z_filt_win, label='Filtered')
            ax.set_xlabel(r'Time [$\mu$s]')
            ax.set_ylabel('$z$ response [V]')
            ax.set_title('{:.0f} keV impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            ax.legend()
            ax.grid(which='both')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return times_win, z_raw_win, z_filt_win

    def fit_susceptibility(self, z_filt_win, file_num=None, impulse_num=None, pdf=None):
        """Fits the mechanical susceptibility of the trapped nanosphere.

        :param z_filt_win: filtered z data
        :type z_filt_win: numpy.ndarray
        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param impulse_num: impulse number, defaults to None
        :type impulse_num: int, optional
        :param pdf: PDF in which to save the susceptiblity fit, defaults to None
        :type pdf: PdfPages, optional
        :return: best-fit parameters for the resonance and a success flag
        :rtype: tuple
        """
        window = self.window_func(len(z_filt_win)//2)
        freq_filt, Pxx_filt = sig.welch(window*z_filt_win[:len(z_filt_win)//2], \
                                        fs=self.f_samp, noverlap=0, nperseg=self.f_samp*0.001)
        fit_bw_hz = 4e3
        f_win_pm = int(fit_bw_hz/2./np.diff(freq_filt)[0])
        peak_ind = f_win_pm + np.argmax(Pxx_filt[f_win_pm:])

        freq_fit = freq_filt[peak_ind - f_win_pm:peak_ind + f_win_pm + 1]
        Pxx_fit = Pxx_filt[peak_ind - f_win_pm:peak_ind + f_win_pm + 1]

        omega = 2*np.pi*freq_fit
        omega_0_guess = 2*np.pi*freq_fit[np.argmax(Pxx_fit)]
        gamma_guess = 2*np.pi*1e-1
        A_guess = np.amax(Pxx_fit)*gamma_guess**2*omega_0_guess**2
        p0 = (A_guess, omega_0_guess, gamma_guess)
        fit_log = True
        try:
            if fit_log:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fit_func = lambda omega, A, omega_0, gamma: np.log(abs_susc2(omega, A, omega_0, gamma))
                    p, _ = curve_fit(fit_func, omega, np.log(Pxx_fit), p0=p0)
            else:
                p, _ = curve_fit(abs_susc2, omega, Pxx_fit, p0=p0)
            success = True
        except RuntimeError:
            print('Error: fitting failed!')
            print('Parameter guess: ', p0)
            p = p0
            success = False
        plot_freq = np.linspace(freq_filt[peak_ind - 3*f_win_pm], freq_filt[peak_ind + 3*f_win_pm + 1], 1000)

        if pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.plot(freq_filt, Pxx_filt, label='Measured')
            ax.set_xlim([plot_freq[0], plot_freq[-1]])
            ax.set_ylim([0, 1.3*max(Pxx_fit)])
            exp = np.floor(np.log10(p)).astype(int)
            mant = p / np.power(10., exp)
            ax.plot(plot_freq, abs_susc2(2*np.pi*plot_freq, *p), \
                    label='$A={:.3f}\\times10^{{{:.0f}}}$, \n'.format(mant[0], exp[0]) + \
                          '$\\omega_0={:.3f}\\times10^{{{:.0f}}}'.format(mant[1], exp[1])  + '~\\mathrm{s^{-1}}$\n' \
                         + '$\\gamma={:.3f}\\times10^{{{:.0f}}}'.format(mant[2], exp[2]) + '~\\mathrm{s^{-1}}$')#\n' \
                        #  + '$C={:.3f}\\times10^{{{:.0f}}}$'.format(mant[3], exp[3]))
            ax.set_xlabel('Frequency [Hz]')
            ax.set_ylabel(r'PSD [$\mathrm{V^2/Hz}$]')
            ax.set_title('{:.0f} keV impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            ax.legend(loc='upper right')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return p, success

    def compute_force(self, times_win, z_filt_win, file_num=None, impulse_num=None, params=None, \
                      time_domain_pdf=None, freq_domain_pdf=None, optimal_filter_pdf=None, search=True):
        """Compute the force imparted by an impulse.

        :param times_win: array of times around the impulse
        :type times_win: numpy.ndarray
        :param z_filt_win: filtered z data around the impulse
        :type z_filt_win: numpy.ndarray
        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param impulse_num: impulse_number, defaults to None
        :type impulse_num: int, optional
        :param params: resonance params to use. If None, uses global parameters
        :type params: numpy.ndarray
        :param time_domain_pdf: PDF in which to save the time domain figure, defaults to None
        :type time_domain_pdf: PdfPages, optional
        :param freq_domain_pdf: PDF in which to save the frequency domain figure, defaults to None
        :type freq_domain_pdf: PdfPages, optional
        :param optimal_filter_pdf: PDF in which to save the optimal filter figure, defaults to None
        :type optimal_filter_pdf: PdfPages, optional
        :param search: whether to search for the force peak in a window, defaults to True
        :type search: bool, optional
        :return: force imparted by the impulse
        :rtype: float
        """

        # window the data. Detrending will have been taken care of by the bandpass already
        z_filt_win *= self.window_func(len(z_filt_win))

        z_fft = 2*np.fft.rfft(z_filt_win)/len(z_filt_win)
        freq_fft = np.fft.rfftfreq(n=len(z_filt_win), d=1./self.f_samp)
        if params is None:
            chi = susc(2.*np.pi*freq_fft, 1., self.omega_0, self.gamma)
        else:
            chi = susc(2.*np.pi*freq_fft, 1., params[1], params[2])
        self.susc = chi
        self.susc_freq = freq_fft

        C = 5e-23
        # C = 100*self.noise_floor/(5e5)**2#p[0]**2

        # model the noise PSD as proportional to the susceptibility plus constant imprecision
        J_psd = np.abs(chi)**2 + C

        # construct the optimal filter from the susceptibility and the noise
        optimal = np.conj(chi)/J_psd

        # compute the force from the product of Z(omega) and the optimal filter
        F_fft = z_fft*optimal

        # deconvolve to get force (Wiener filter picture) or get amplitude of signals matching template 
        # vs time (optimal filter picture)
        f_td = np.fft.irfft(F_fft, n=len(z_filt_win))

        lpf = sig.butter(3, self.f_cutoff[1], btype='lowpass', output='sos', fs=self.f_samp)
        f_filt = sig.sosfiltfilt(lpf, f_td)
        f_filt = np.copy(f_td)
        f_filt[:1000] = 0
        f_filt[-1000:] = 0
        f_td[:1000] = 0
        f_td[-1000:] = 0

        if search:
            width = int(self.search_window*self.f_samp)
            force_ind = len(times_win)//2 - width//2 + np.argmax(np.abs(f_filt[len(times_win)//2 - width//2:\
                                                                 len(times_win)//2 + width//2]))
            force = f_filt[force_ind]
        else:
            force = np.std(f_filt)
            force_ind = len(times_win)//2

        if optimal_filter_pdf:
            fig, ax = plt.subplots(2, figsize=(6, 6), layout='constrained')
            sort_inds = np.concat((np.arange(len(times_win)//2, len(times_win)), np.arange(0, len(times_win)//2)))
            ax[0].plot((times_win - np.mean(times_win))*1e6, np.fft.irfft(optimal)[sort_inds], \
                       label='$C$ from noise')
            ax[0].plot((times_win - np.mean(times_win))*1e6, np.fft.irfft(np.conj(chi)/(np.abs(chi)**2 + 10*C))[sort_inds], \
                       label=r'$C\times10$')
            ax[0].plot((times_win - np.mean(times_win))*1e6, np.fft.irfft(np.conj(chi)/(np.abs(chi)**2 + 0.1*C))[sort_inds], \
                       label=r'$C/10$')
            ax[0].set_xlim([-20, 20])
            ax[0].set_xlabel(r'Time [$\mu$s]')
            ax[0].set_ylabel('Filter magnitude [au]')
            ax[0].legend()
            ax[0].set_title('{:.0f} keV impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            ax[1].semilogy(freq_fft*1e-3, np.abs(optimal))
            ax[1].semilogy(freq_fft*1e-3, np.abs(np.conj(chi)/(np.abs(chi)**2 + 10*C)))
            ax[1].semilogy(freq_fft*1e-3, np.abs(np.conj(chi)/(np.abs(chi)**2 + 0.1*C)))
            ax[1].set_xlabel('Frequency [kHz]')
            ax[1].set_ylabel('Filter magnitude [au]')
            ax[1].set_xlim([0, 1e2])
            optimal_filter_pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        if time_domain_pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.plot((times_win - np.mean(times_win))*1e6, z_filt_win, label='$z$ position')
            ax.axvline(0, color='C4', lw=1, ls='--', label='Impulse time', zorder=1000)
            ax2 = ax.twinx()
            ax2.plot((times_win - np.mean(times_win))*1e6, f_td, color='C1', label='Force')
            ax2.plot((times_win - np.mean(times_win))*1e6, f_filt, color='C2', label='Force filtered')
            ax2.plot((times_win[force_ind] - np.mean(times_win))*1e6, force, color='C3', marker='.', ms=5, zorder=100)
            ax.set_xlabel(r'Time [$\mu$s]')
            ax.set_ylabel('$z$ response [V]')
            ax2.set_ylabel('Force [au]')
            ax.set_title('{:.0f} keV impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            time_domain_pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax, ax2
            gc.collect()

        if freq_domain_pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.semilogy(freq_fft*1e-3, np.abs(z_fft)**2/np.amax(np.abs(z_fft)**2), label=r'$\tilde{z}(\omega)$')
            ax.semilogy(freq_fft*1e-3, np.abs(chi)**2/np.amax(np.abs(chi)**2), label=r'$\chi(\omega)$')
            ax.semilogy(freq_fft*1e-3, J_psd/np.amax(J_psd), label=r'$J(\omega)$')
            ax.semilogy(freq_fft*1e-3, np.abs(F_fft)**2/np.amax(np.abs(F_fft)**2), label=r'$\tilde{F}(\omega)$')
            ax.legend(ncol=2)
            ax.set_xlabel('Frequency [kHz]')
            ax.set_xlim([0, 1e2])
            ax.set_ylabel('Magnitude [au]')
            ax.set_ylim([1e-12, 2e0])
            ax.set_title('{:.0f} keV impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            freq_domain_pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return force
    
class NanoDataset:
    """Class to handle a dataset containing multiple files of nanosphere data.
    """

    def __init__(self, path, plot_path=None, f_cutoff=[2e4, 1e5], t_window=1e-3, max_files=1000, verbose=False):
        """Initializes a NanoDataset object

        :param path: path to the files to be loaded
        :type path: string
        :param plot_path: path specifying where figures should be saved, defaults to None
        :type plot_path: string, optional
        :param f_cutoff: filter cutoff frequencies in Hz, defaults to [2e4, 1e5]
        :type f_cutoff: list, optional
        :param t_window: pulse window in seconds, defaults to 1e-3
        :type t_window: float, optional
        :param max_files: maximum number of files to load, defaults to 1000
        :type max_files: int, optional
        :param verbose: whether to print verbose output, defaults to False
        :type verbose: bool, optional
        """
        self.file_paths = glob(path + '_*.hdf5')
        self.file_inds = [int(f.split('_')[-1].split('.hdf5')[0]) for f in self.file_paths]
        sort_inds = np.argsort(self.file_inds)
        self.file_paths = np.array(self.file_paths)[sort_inds][:max_files]
        self.file_inds = np.array(self.file_inds)[sort_inds][:max_files]
        self.plot_path = plot_path
        self.f_cutoff = f_cutoff
        self.t_window = t_window
        self.verbose = verbose
        self.create_pdfs()

    def create_pdfs(self):
        """Create the PDF files that the figures will be added to.
        """
        if self.plot_path:
            Path('/'.join(self.plot_path.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
            self.time_domain_pdf = PdfPages(self.plot_path + '_time_domain.pdf')
            self.freq_domain_pdf = PdfPages(self.plot_path + '_freq_domain.pdf')
            self.optimal_filter_pdf = PdfPages(self.plot_path + '_optimal_filter.pdf')
            self.res_fit_pdf = PdfPages(self.plot_path + '_res_fit.pdf')
            self.impulse_win_pdf = PdfPages(self.plot_path + '_impulse_window.pdf')
            self.spectra_pdf = PdfPages(self.plot_path + '_spectra.pdf')
            self.impulse_times_pdf = PdfPages(self.plot_path + '_impulse_times.pdf')
        else:
            self.time_domain_pdf, self.freq_domain_pdf, self.optimal_filter_pdf, self.res_fit_pdf, \
                self.impulse_win_pdf, self.spectra_pdf, self.impulse_times_pdf = None, None, None, None, None, None, None

    def close_pdfs(self):
        """Close the PDF files.
        """
        if self.plot_path:
            self.time_domain_pdf.close()
            self.freq_domain_pdf.close()
            self.optimal_filter_pdf.close()
            self.res_fit_pdf.close()
            self.impulse_win_pdf.close()
            self.spectra_pdf.close()
            self.impulse_times_pdf.close()
    
    def load_files(self, global_params=True, pulse_amps_1e=None, pulse_amps_V=None, \
                   noise=False, search=True):
        """Load the datasets as NanoFile objects and extract relevant data from them.

        :param global_params: whether to use global resonance parameters (vs pulse-by-pulse), defaults to True
        :type global_params: bool, optional
        :param pulse_amps_1e: amplitudes of applied impulses in keV/c assuming 1e charge, defaults to None
        :type pulse_amps_1e: list, optional
        :param pulse_amps_V: amplitudes of applied impulses in V, defaults to None
        :type pulse_amps_V: list, optional
        :param noise: whether to use noise indices instead of impulse indices, defaults to False
        :type noise: bool, optional
        :param search: whether to search for the force peak in a window, defaults to True
        :type search: bool, optional
        """
        forces = []
        res_params = []
        suscs = []

        for i, fp in zip(self.file_inds, self.file_paths):
            if self.verbose:
                print('Loading file {}...'.format(i+1))
            nf = NanoFile(fp, f_cutoff=self.f_cutoff, t_window=self.t_window, verbose=self.verbose)
            nf.calibrate_pulse_amp(pulse_amps_1e, pulse_amps_V)
            nf.compute_and_fit_psd(file_num=i, pdf=self.spectra_pdf)
            impulse_inds = nf.get_impulse_inds(file_num=i, pdf=self.impulse_times_pdf)
            if noise:
                impulse_inds = nf.get_noise_inds(impulse_inds)
            nf.get_force_array(impulse_inds=impulse_inds, global_params=global_params, file_num=i, \
                               search=search, pdfs=(self.time_domain_pdf, self.freq_domain_pdf, \
                               self.optimal_filter_pdf, self.res_fit_pdf, self.impulse_win_pdf))
            forces.append(nf.forces.copy())
            suscs.append(nf.susc.copy())
            if res_params is None:
                res_params = [nf.resonance_params.copy()]
            else:
                res_params.append(nf.resonance_params.copy())
            if i == 0:
                self.pulse_amp_keV = nf.pulse_amp_keV.copy()
                self.freqs = nf.susc_freq.copy()

            del nf
            gc.collect()

        self.forces = np.concatenate(forces)
        self.resonance_params = np.concatenate(res_params)
        self.suscs = np.array(suscs)
        self.close_pdfs()