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

    def __init__(self, file_path, f_cutoff=[2e4, 1e5], t_window=1e-3, search_window=5e-5, \
                 verbose=False):
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
        self.f_cutoff = f_cutoff
        self.t_window = t_window
        self.search_window = search_window
        self.noise_floor = 0.
        self.window_func = lambda N: sig.windows.boxcar(N) #sig.windows.tukey(N, 0.05) #sig.windows.boxcar(N) #sig.windows.hann(N)
        self.verbose = verbose
        self.deconvolved_pulses = []
        self.recon_impulse_inds = []
        self.pulse_times = []
        self.pulse_amp_keV = None
        self.load_file()

    def load_file(self):
        """Load data from the HDF5 file and do some preliminary processing.
        """

        with h5py.File(self.file_path, 'r') as f:
            self.z_raw = np.array(f['data/channel_d'])*f['data/channel_d'].attrs['adc2mv']*1e-3
            try:
                self.imp_raw = np.array(f['data/channel_g'])*f['data/channel_g'].attrs['adc2mv']*1e-3
            except:
                pass
            self.mon_raw = np.array(f['data/channel_f'])*f['data/channel_f'].attrs['adc2mv']*1e-3

            self.f_samp = 1./f['data'].attrs['delta_t']

            self.timestamp = f['data'].attrs['timestamp']

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
            if self.pulse_amp_keV:
                ax.set_title('{:.0f} keV/c impulse, file {}'.format(self.pulse_amp_keV, file_num + 1))
            else:
                ax.set_title('File {}'.format(file_num + 1))
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

    def calibrate_pulse_amp(self, pulse_amps_1e, pulse_amps_V):
        """Loads the calibration factors to convert pulse amplitudes into keV/c.

        :param pulse_amps_1e: amplitude of pulses in keV/c assuming a 1e charge
        :type pulse_amps_1e: list
        :param pulse_amps_V: amplitude of pulses delivered in volts
        :type pulse_amps_V: list
        """
        charge_p = self.file_path.split('_p')[-1].split('e_')[0]
        charge_n = self.file_path.split('_n')[-1].split('e_')[0]
        self.n_charges = float([charge_p, charge_n][len(charge_p) > len(charge_n)])
        self.pulse_amp_V = float(self.file_path.split('v_')[0].split('_')[-1])
        self.pulse_amp_keV = pulse_amps_1e[np.argmin(np.abs(pulse_amps_V - self.pulse_amp_V))]*self.n_charges

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
            ax.set_title('{:.0f} keV/c impulse, file {}'.format(self.pulse_amp_keV, file_num + 1))
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
        noise_inds = impulse_inds - int(0.1*np.mean(np.diff(impulse_inds)))
        if np.any(noise_inds < 0):
            noise_inds = impulse_inds + int(0.1*np.mean(np.diff(impulse_inds)))
        return noise_inds

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
            times_win = np.concat((self.times[impulse_ind - t_win_pm:], self.times[:impulse_ind + t_win_pm - len(self.times)]))
            z_raw_win = np.concat((self.z_raw[impulse_ind - t_win_pm:], self.z_raw[:impulse_ind + t_win_pm - len(self.times)]))
            z_filt_win = np.concat((self.z_filtered[impulse_ind - t_win_pm:], self.z_filtered[:impulse_ind + t_win_pm - len(self.times)]))
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
            ax.set_title('{:.0f} keV/c impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            ax.legend()
            ax.grid(which='both')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return times_win, z_raw_win, z_filt_win

    def search_all_data(self, file_num=None, max_windows=1000000, freq_domain_pdf=None, time_domain_pdf=None):
        search_win = int(self.search_window*self.f_samp)
        time_win = int(self.t_window*self.f_samp)
        forces = []
        
        first_center = time_win // 2
        last_center = len(self.times) - time_win // 2
        centers = np.arange(first_center, last_center, search_win)
        num_windows = len(centers)
        
        for i, center in enumerate(centers):
            if i >= max_windows:
                break
            if self.verbose:
                print('    -> Computing force for window {} of {}...'.format(i + 1, num_windows))
            start_idx = center - time_win // 2
            end_idx = center + time_win // 2
            times_win = np.copy(self.times[start_idx:end_idx])
            z_filt_win = np.copy(self.z_filtered[start_idx:end_idx])
            forces.append(self.compute_force(times_win, z_filt_win, file_num=file_num, impulse_num=i, \
                          freq_domain_pdf=freq_domain_pdf, time_domain_pdf=time_domain_pdf))

        self.forces = np.array(forces)

    def get_force_array(self, impulse_inds, global_params=True, file_num=0, pdfs=None):
        """Computes the force at each impulse in the data and returns the results in an array.

        :param impulse_inds: indices at which impulses were applied
        :type impulse_inds: numpy.ndarray
        :param global_params: whether to use global resonance parameters (vs pulse-by-pulse)
        :type global_params: bool, optional
        :param file_num: file number, defaults to 0
        :type file_num: int, optional
        :param pdfs: PDFs in which to save the figures, defaults to None
        :type pdfs: string, optional
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
                                             time_domain_pdf, freq_domain_pdf, optimal_filter_pdf))
        
        self.resonance_params = np.array(resonance_params)
        self.forces = np.array(forces)

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
            if self.pulse_amp_keV:
                ax.set_title('{:.0f} keV/c impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            else:
                ax.set_title('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
            ax.legend(loc='upper right')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return p, success

    def deconvolve_response(self, z_filt_win, params=None):
        """Deconvolve the response of the oscillator from the position data.

        :param z_filt_win: filtered z data around the impulse
        :type z_filt_win: numpy.ndarray
        :param params: resonance params to use. If None, uses global parameters
        :type params: numpy.ndarray
        """

        # window the data. Detrending will have been taken care of by the bandpass already
        z_filt_win *= self.window_func(len(z_filt_win))

        z_fft = np.fft.rfft(z_filt_win)*np.sqrt(2)/len(z_filt_win)
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
        return z_fft*optimal, freq_fft

    def compute_force(self, times_win, z_filt_win, file_num=None, impulse_num=None, params=None, \
                      time_domain_pdf=None, freq_domain_pdf=None, optimal_filter_pdf=None):
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
        :return: force imparted by the impulse
        :rtype: float
        """

        F_fft, freq_fft = self.deconvolve_response(times_win, params)

        # deconvolve to get force (Wiener filter picture) or get amplitude of signals matching template 
        # vs time (optimal filter picture)
        f_td = np.fft.irfft(F_fft, n=len(z_filt_win))*len(z_filt_win)/np.sqrt(2)

        lpf = sig.butter(3, self.f_cutoff[1], btype='lowpass', output='sos', fs=self.f_samp)
        f_filt = sig.sosfiltfilt(lpf, f_td)
        f_filt = np.copy(f_td)
        f_filt[:1000] = 0
        f_filt[-1000:] = 0
        f_td[:1000] = 0
        f_td[-1000:] = 0

        # find the maximum within the search window
        width = int(self.search_window*self.f_samp)
        force_ind = len(times_win)//2 - width//2 + np.argmax(np.abs(f_filt[len(times_win)//2 - width//2:\
                                                             len(times_win)//2 + width//2]))
        force = f_filt[force_ind]

        # save the deconvolved pulse as an attribute
        self.deconvolved_pulses.append(f_filt)
        self.recon_impulse_inds.append(force_ind)
        self.pulse_times.append(times_win[0])

        if optimal_filter_pdf:
            fig, ax = plt.subplots(2, figsize=(6, 6), layout='constrained')
            sort_inds = np.concat((np.arange(len(times_win)//2, len(times_win)), np.arange(0, len(times_win)//2)))
            ax[0].plot((times_win - times_win[len(times_win)//2])*1e6, np.fft.irfft(optimal)[sort_inds], \
                       label='$C$ from noise')
            ax[0].plot((times_win - times_win[len(times_win)//2])*1e6, np.fft.irfft(np.conj(chi)/(np.abs(chi)**2 + 10*C))[sort_inds], \
                       label=r'$C\times10$')
            ax[0].plot((times_win - times_win[len(times_win)//2])*1e6, np.fft.irfft(np.conj(chi)/(np.abs(chi)**2 + 0.1*C))[sort_inds], \
                       label=r'$C/10$')
            ax[0].set_xlim([-20, 20])
            ax[0].set_xlabel(r'Time [$\mu$s]')
            ax[0].set_ylabel('Filter magnitude [au]')
            ax[0].legend()
            if self.pulse_amp_keV:
                ax[0].set_title('{:.0f} keV/c impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            else:
                ax[0].set_title('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
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
            ax.plot((times_win - times_win[force_ind])*1e6, z_filt_win, label='$z$ position')
            ax.axvline((times_win[len(times_win)//2] - times_win[force_ind])*1e6, \
                       color='C4', lw=1, ls='--', label='Impulse time', zorder=10)
            ax2 = ax.twinx()
            ax2.plot((times_win - times_win[force_ind])*1e6, f_td, color='C1', label='Force')
            ax2.plot((times_win - times_win[force_ind])*1e6, f_filt, color='C2', label='Force filtered')
            ax2.plot(0, force, color='C3', marker='.', ms=5, label='Reconstructed', zorder=10)
            ax.set_xlabel(r'Time [$\mu$s]')
            ax.set_ylabel('$z$ response [V]')
            ax2.set_ylabel('Force [au]')
            if self.pulse_amp_keV:
                ax.set_title('{:.0f} keV/c impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            else:
                ax.set_title('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
            leg = ax.legend(loc='upper left')
            leg.remove()
            ax2.legend(loc='upper right')
            ax2.add_artist(leg)
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
            if self.pulse_amp_keV:
                ax.set_title('{:.0f} keV/c impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            else:
                ax.set_title('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
            freq_domain_pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return force

    def integrate_noise(self):
        """Integrate the force noise spectrum.
        """
        F_fft, freqs = self.deconvolve_response(self.z_filtered)

        plt.figure()
        plt.semilogy(freqs, np.abs(F_fft))
        plt.xlim([0, 1e5])
        return np.sqrt(trapezoid(np.abs(F_fft)**2, freqs))
    
class NanoDataset:
    """Class to handle a dataset containing multiple files of nanosphere data.
    """

    def __init__(self, path, plot_path=None, f_cutoff=[2e4, 1e5], t_window=1e-3, \
                 search_window=5e-5, max_files=1000, max_windows=1000000, verbose=False):
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
        :param max_windows: maximum number of windows to search in, defaults to 1000
        :type max_windows: int, optional
        :param verbose: whether to print verbose output, defaults to False
        :type verbose: bool, optional
        """
        if not isinstance(path, list):
            path = [path]
        self.file_paths = list(np.concat([glob(p + '_*.hdf5') for p in path]))
        self.file_inds = [int(f.split('_')[-1].split('.hdf5')[0]) for f in self.file_paths]
        sort_inds = np.argsort(self.file_inds)
        self.file_paths = np.array(self.file_paths)[sort_inds][:max_files]
        self.file_inds = np.array(self.file_inds)[sort_inds][:max_files]
        self.max_windows = max_windows
        self.plot_path = plot_path
        self.f_cutoff = f_cutoff
        self.t_window = t_window
        self.search_window = search_window
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
    
    def load_calibration_data(self, global_params=True, pulse_amps_1e=None, pulse_amps_V=None, \
                              noise=False):
        """Loads calibration data files as NanoFile objects and extracts relevant data from them.

        :param global_params: whether to use global resonance parameters (vs pulse-by-pulse), defaults to True
        :type global_params: bool, optional
        :param pulse_amps_1e: amplitudes of applied impulses in keV/c assuming 1e charge, defaults to None
        :type pulse_amps_1e: list, optional
        :param pulse_amps_V: amplitudes of applied impulses in V, defaults to None
        :type pulse_amps_V: list, optional
        :param noise: whether to use noise indices instead of impulse indices, defaults to False
        :type noise: bool, optional
        """
        forces = []
        res_params = []
        suscs = []
        pulses = []
        recon_impulse_inds = []
        pulse_times = []
        timestamps = []

        for i, fp in zip(self.file_inds, self.file_paths):
            if self.verbose:
                print('Loading file {}...'.format(i+1))
            nf = NanoFile(fp, f_cutoff=self.f_cutoff, t_window=self.t_window, search_window=self.search_window, \
                          verbose=self.verbose)
            nf.calibrate_pulse_amp(pulse_amps_1e, pulse_amps_V)
            nf.compute_and_fit_psd(file_num=i, pdf=self.spectra_pdf)
            impulse_inds = nf.get_impulse_inds(file_num=i, pdf=self.impulse_times_pdf)
            if noise:
                impulse_inds = nf.get_noise_inds(impulse_inds)
            nf.get_force_array(impulse_inds=impulse_inds, global_params=global_params, file_num=i, \
                               pdfs=(self.time_domain_pdf, self.freq_domain_pdf, self.optimal_filter_pdf, \
                               self.res_fit_pdf, self.impulse_win_pdf))
            forces.append(nf.forces.copy())
            suscs.append(nf.susc.copy())
            pulses.append(nf.deconvolved_pulses.copy())
            recon_impulse_inds.append(nf.recon_impulse_inds)
            pulse_times.append(nf.pulse_times)
            timestamps.append(nf.timestamp)
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
        self.pulses = np.concatenate(pulses)
        self.recon_impulse_inds = np.concatenate(recon_impulse_inds)
        self.pulse_times = np.concatenate(pulse_times)
        self.timestamps = np.array(timestamps)
        self.suscs = np.array(suscs) # can maybe get rid of this
        self.close_pdfs()

    def load_search_data(self, compute_sensitivity=False):
        """Loads impulse search data files as NanoFile objects and extracts relevant data from them.

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
        pulses = []
        recon_impulse_inds = []

        for i, fp in zip(self.file_inds, self.file_paths):
            if self.verbose:
                print('Loading file {}...'.format(i+1))
            nf = NanoFile(fp, f_cutoff=self.f_cutoff, t_window=self.t_window, verbose=self.verbose)
            nf.compute_and_fit_psd(file_num=i, pdf=self.spectra_pdf)
            nf.search_all_data(file_num=i, max_windows=self.max_windows, freq_domain_pdf=self.freq_domain_pdf, \
                               time_domain_pdf=self.time_domain_pdf)
            forces.append(nf.forces.copy())
            pulses.append(nf.deconvolved_pulses.copy())
            recon_impulse_inds.append(nf.recon_impulse_inds)
            if i == 0:
                self.freqs = nf.susc_freq.copy()
                if compute_sensitivity:
                    self.integrated_noise = nf.integrate_noise()

            del nf
            gc.collect()

        self.forces = np.concatenate(forces)
        self.pulses = np.concatenate(pulses)
        self.recon_impulse_inds = np.concatenate(recon_impulse_inds)
        self.close_pdfs()