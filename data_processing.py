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

from iminuit import Minuit
from iminuit.cost import LeastSquares

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
    # omega = np.asarray(omega)
    # if (gamma < 0) or (A < 0):
    #     return np.full_like(omega, -np.inf, dtype=float)
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
    # if gamma < 0:
    #     return -np.inf
    return A/(omega_0**2 - omega**2 - 1j*gamma*omega)

class NanoFile:
    """Class to handle an individual file containing nanosphere data.
    """

    def __init__(self, file_path, f_cutoff=[2e4, 1e5], t_window=1e-3, search_window=5e-5, \
                 d_sphere_nm=166., verbose=False):
        """Initializes a NanoFile object containing data from a single HDF5 file.

        :param file_path: path to the HDF5 file to load
        :type file_path: string
        :param f_cutoff: upper and lower cutoff frequencies, defaults to [2e4, 1e5]
        :type f_cutoff: list, optional
        :param t_window: pulse window in seconds, defaults to 1e-3
        :type t_window: float, optional
        :param d_sphere_nm: nanosphere diameter in nanometers, defaults to 166
        :type d_sphere_nm: float, optional
        :param verbose: whether to print verbose output, defaults to False
        :type verbose: bool, optional
        """
        self.file_path = file_path
        self.f_cutoff = f_cutoff
        self.t_window = t_window
        self.search_window = search_window
        self.window_func = 'tukey'
        self.verbose = verbose
        self.pulse_amp_keV = None
        V_ns = (4/3.)*np.pi*(d_sphere_nm*1e-9/2.)**3
        rho_silica = 2.65e3 # density of silica, kg/m^3
        self.mass_sphere = rho_silica*V_ns # mass of the sphere in kg
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

    def calibrate_spectrum(self, Pxx_mon_raw, Pxx_z_raw, *params):
        """Calibrate the position spectrum to Newtons

        :param Pxx_mon_raw: the PSD of the raw monitoring signal
        :type Pxx_mon_raw: numpy.ndarray
        """

        print('\n  Calibrating volts to meters...')

        Efield = 79. # V/m (when 1 V is applied to lens holder 2)
        e = 1.602176634e-19 # unit charge

        # bandpass filter around the monitoring frequency
        f_mon = self.freqs[np.argmax(Pxx_mon_raw)]
        bp_bw = 4e3
        bp = sig.butter(4, [f_mon - bp_bw/2, f_mon + bp_bw/2], btype='bandpass', output='sos', fs=self.f_samp)
        force_applied = np.copy(self.mon_raw) # force on the sphere in N
        drive = sig.sosfiltfilt(bp, force_applied)
        resp = sig.sosfiltfilt(bp, self.z_raw)

        # get in-phase and quadrature components of response
        z = sig.hilbert(drive)
        phi = np.unwrap(np.angle(z))
        i_raw = resp*np.cos(phi)
        q_raw = resp*np.sin(phi)
        
        # low-pass filter the demodulated data streams
        f_lpf = 1e6
        lpf = sig.butter(4, f_lpf, btype='low', output='sos', fs=self.f_samp)

        # demodulate the response
        i_filt = 2*sig.sosfiltfilt(lpf, i_raw)
        q_filt = 2*sig.sosfiltfilt(lpf, q_raw)
        R = i_filt + 1j*q_filt

        # demodulate the drive
        i_drive = 2*sig.sosfiltfilt(lpf, drive*np.cos(phi))
        q_drive = 2*sig.sosfiltfilt(lpf, drive*np.sin(phi))
        D = i_drive + 1j*q_drive

        print('  Mass of the nanosphere: \t\t{:.3e} kg'.format(self.mass_sphere))
        print('  Number of charges on the nanosphere: \t{}'.format(int(self.n_charges)))
        print('  Electric field per volt applied: \t{:.1f} V/m'.format(Efield))
        print('  Drive signal from demodulation: \t{:.3f} V'.format(np.mean(np.abs(D))))
        print('  Drive signal from Welch spectrum: \t{:.3f} V'.format(np.sqrt(np.abs(Pxx_mon_raw[np.argmin(np.abs(self.freqs - f_mon))]))))
        print('  Applied force from demodulation: \t{:.3e} N'.format(self.n_charges*e*Efield*np.mean(np.abs(D))))
        print('  Applied force from Welch spectrum: \t{:.3e} N'.format(self.n_charges*e*Efield*np.sqrt(np.abs(Pxx_mon_raw[np.argmin(np.abs(self.freqs - f_mon))]))))
        print('  Sensor response from demodulation: \t{:.3e} V'.format(np.mean(np.abs(R))))
        print('  Sensor response from Welch spectrum: \t{:.3e} V'.format(np.sqrt(np.abs(Pxx_z_raw[np.argmin(np.abs(self.freqs - f_mon))]))))

        # compute the susceptibility in meters/Newton
        self.susceptibility = lambda omega: susc(omega, 1, params[1], params[2])/self.mass_sphere

        print('  Susceptibility at {:.1f} kHz drive: \t{:.3e} m/V'.format(f_mon*1e-3, np.abs(self.susceptibility(2*np.pi*f_mon))))
        print('  Nanosphere response amplitude: \t{:.3e} m'.format(self.n_charges*e*Efield*np.mean(np.abs(D))*np.abs(self.susceptibility(2*np.pi*f_mon))))

        # conversion factor from volts to meters
        # self.meters_per_volt = np.mean(self.n_charges*e*Efield*np.abs(D/R))*np.abs(self.susceptibility(2*np.pi*f_mon))
        self.meters_per_volt = np.mean(self.n_charges*e*Efield*np.sqrt(np.abs(Pxx_mon_raw[np.argmin(np.abs(self.freqs - f_mon))])))/np.sqrt(np.abs(Pxx_z_raw[np.argmin(np.abs(self.freqs - f_mon))]))*np.abs(self.susceptibility(2*np.pi*f_mon))
        print('  Position response calibration factor: {:.3e} m/V\n'.format(self.meters_per_volt))

        # save the calibrated z position data
        # self.z_calibrated = np.copy(self.z_filtered)
        # print(self.z_calibrated.dtype)
        self.z_calibrated = self.z_filtered*self.meters_per_volt

        # self.meters_per_volt = meters_per_volt

        # construct the transfer function in the time domain
        # TF = R/(D + 1e-30)
        # self.volts_per_newton = np.abs(TF)

        # self.factor = np.abs(1/np.mean(np.abs(TF))*susc(2*np.pi*f_mon, *params)/susc(2*np.pi*self.freqs, *params))

        # fig, ax = plt.subplots(2, figsize=(6, 6))
        # # ax[0].plot(resp[:5000], label='Raw response')
        # # ax[0].plot(drive[:5000]*1e-2, label='Raw drive')
        # # ax[0].plot(np.abs(R[:5000]), label='Response')
        # # ax[0].plot(np.abs(D[:5000]*1e-6), label='Drive')
        # ax[0].plot(self.times[:50000], np.abs(TF[:50000]), label='TF')
        # ax[0].set_ylabel('Magnitude [V/m]')
        # ax[0].legend(loc='upper right')
        # # ax[1].plot(resp[:1000], label='Raw response')
        # # ax[1].plot(drive[:1000]*1e-4, label='Raw drive')
        # # ax[1].plot(np.rad2deg(np.angle(R[:5000])), label='Response')
        # # ax[1].plot(np.rad2deg(np.angle(D[:5000])), label='Drive')
        # ax[1].plot(self.times[:50000], np.rad2deg(np.angle(TF[:50000])), label='TF')
        # ax[1].set_ylabel(r'Phase [$^\circ$]')
        # ax[1].set_ylim([-200, 200])
        # ax[1].set_yticks([-180, -90, 0, 90, 180])

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

        nperseg = int(5e5)
        window_s1 = np.sum(sig.get_window(self.window_func, nperseg))
        window_s2 = np.sum(sig.get_window(self.window_func, nperseg)**2)
        spectrum_to_density = window_s1/np.sqrt(window_s2*self.f_samp)
        self.freqs, Pxx_z_raw = sig.welch(self.z_raw, fs=self.f_samp, window=self.window_func, nperseg=nperseg, scaling='spectrum')
        _, Pxx_z_filt = sig.welch(self.z_filtered, fs=self.f_samp, window=self.window_func, nperseg=nperseg, scaling='spectrum')
        _, Pxx_mon_raw = sig.welch(self.mon_raw, fs=self.f_samp, window=self.window_func, nperseg=nperseg, scaling='spectrum')

        impr_win = 1.5*self.freqs[np.argmax(Pxx_z_filt)] + np.array((0, 1e4))
        impr_inds = [np.argmin(np.abs(n - self.freqs)) for n in impr_win]
        noise_floor = np.mean(Pxx_z_raw[impr_inds[0]:impr_inds[1]])

        p = self.fit_voigt_profile(Pxx_z_raw, noise_floor)

        self.calibrate_spectrum(Pxx_mon_raw, Pxx_z_raw, p[0], p[1], p[3])

        self.sensing_noise = noise_floor*self.meters_per_volt**2*spectrum_to_density**2

        self.gamma = p[-1]
        self.omega_0 = p[1]
        self.amplitude = p[0]*self.meters_per_volt*spectrum_to_density

        if pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.semilogy(self.freqs*1e-3, Pxx_z_raw*spectrum_to_density**2, alpha=1.0, label='$z$ raw')
            ax.semilogy(self.freqs*1e-3, Pxx_z_filt*spectrum_to_density**2, alpha=1.0, label='$z$ filtered')
            ax.semilogy(self.freqs*1e-3, (p[0]**2*voigt_profile(2*np.pi*self.freqs - p[1], p[2], p[3]) \
                        + noise_floor)*spectrum_to_density**2, alpha=1.0, lw=1.0, label='Voigt fit')
            # ax.semilogy(self.freqs*1e-3, Pxx_mon_raw, alpha=1.0, label='Monitoring raw')
            if self.pulse_amp_keV:
                ax.set_title('{:.0f} keV/c impulse, file {}'.format(self.pulse_amp_keV, file_num + 1))
            else:
                ax.set_title('File {}'.format(file_num + 1))
            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel(r'PSD [$\mathrm{V^2/Hz}$]')
            ax.set_ylim([1e-16, 1e-4])
            ax.set_xlim([10, 100])
            ax.legend(loc='upper right')
            exp = np.floor(np.log10(p)).astype(int)
            mant = p / np.power(10., exp)
            ax.text(0.03, 0.97, '$A={:.3f}\\times10^{{{:.0f}}}$, \n'.format(mant[0], exp[0]) + \
                                '$\\omega_0={:.3f}\\times10^{{{:.0f}}}'.format(mant[1], exp[1])  + '~\\mathrm{s^{-1}}$\n' + \
                                '$\\sigma={:.3f}\\times10^{{{:.0f}}}'.format(mant[2], exp[2])  + '~\\mathrm{s^{-1}}$\n' + \
                                '$\\gamma={:.3f}\\times10^{{{:.0f}}}'.format(mant[3], exp[3]) + '~\\mathrm{s^{-1}}$', \
                                ha='left', va='top', transform=ax.transAxes)
            ax.grid(which='both')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.semilogy(self.freqs*1e-3, np.sqrt(Pxx_z_raw)*spectrum_to_density*self.meters_per_volt/\
                                         np.abs(self.susceptibility(2*np.pi*self.freqs)), alpha=1.0, label='$z$ raw')
            ax2 = ax.twinx()
            ax2.semilogy(self.freqs*1e-3, np.sqrt(Pxx_z_raw)*spectrum_to_density*self.meters_per_volt, color='C1')
            _, Pxx = sig.welch(self.z_raw, fs=self.f_samp, window=self.window_func, nperseg=nperseg)
            # ax2.semilogy(self.freqs*1e-3, np.sqrt(Pxx)*self.meters_per_volt, color='C2')
            ax2.set_ylabel(r'ASD [$\mathrm{m/\sqrt{Hz}}$]')
            if self.pulse_amp_keV:
                ax.set_title('{:.0f} keV/c impulse, file {}'.format(self.pulse_amp_keV, file_num + 1))
            else:
                ax.set_title('File {}'.format(file_num + 1))
            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel(r'ASD [$\mathrm{N/\sqrt{Hz}}$]')
            # ax.set_ylim([1e-16, 1e-4])
            ax.set_xlim([10, 200])
            # ax.legend(loc='upper right')
            ax.grid(which='both')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

    def fit_voigt_profile(self, Pxx_z_filt, C):
        """Fits a Voigt profile to the spectrum for a full file.

        :param Pxx_z_filt: the filtered PSD for the file
        :type Pxx_z_filt: numpy.ndarray
        :return: the best fit parameters
        :rtype: numpy.ndarray
        """
        f_0_range = self.freqs[np.argmax(Pxx_z_filt*np.array(self.freqs > 1e4, dtype=float))] + np.array([-5e3, 5e3])
        f_0_inds = [np.argmin(np.abs(self.freqs - f)) for f in f_0_range]

        omega_0_guess = 2*np.pi*self.freqs[f_0_inds[0] + np.argmax(Pxx_z_filt[f_0_inds[0]:f_0_inds[1]])]
        gamma_guess = 2*np.pi*1e-2
        sigma_guess = 2*np.pi*1e2
        A_guess = 5e-3
        p0 = [A_guess, omega_0_guess, sigma_guess, gamma_guess]

        fit_bw = 2*np.pi*2e4
        fit_inds = (2*np.pi*self.freqs > omega_0_guess - fit_bw/2.) & (2*np.pi*self.freqs < omega_0_guess + fit_bw/2.)

        voigt_scaled = lambda omega, A, omega_0, sigma, gamma: A**2*voigt_profile(omega - omega_0, sigma, gamma) + C
        try:
            p, _ = curve_fit(voigt_scaled, 2*np.pi*self.freqs[fit_inds], Pxx_z_filt[fit_inds], \
                             sigma=np.sqrt(Pxx_z_filt[fit_inds]), p0=p0)
            p = np.abs(p)
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

    def get_time_window(self, time_ind, t_window, centered=True, end_mode='wrap', file_num=None, \
                        impulse_num=None, pdf=None):
        """Gets the time window around an impulse.

        :param time_ind: index defining the time at which the window should be taken
        :type time_ind: int
        :param t_window: length of the time window in seconds
        :type t_window: float
        :param centered: center the window around the index vs taking it before or after
        :type centered: bool, optional
        :param end_mode: what to do at the end of the file; either 'wrap' from the other end or
        'pad' with zeros
        :type end_mode: str
        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param impulse_num: impulse number, defaults to None
        :type impulse_num: int, optional
        :param pdf: PDF in which the time series data should be saved, defaults to None
        :type pdf: PdfPages, optional
        :return: times and z data around the impulse
        :rtype: tuple
        """

        t_win_pm = int(t_window*self.n_samp/self.t_int/2)

        if end_mode not in ['wrap', 'pad']:
            raise ValueError('end_mode argument not recognized (must be either "wrap" or "pad").')

        if centered:
            # if the impulse is within the time window of the start/end of the file,
            # wrap around and pad with noise data from the other end
            if time_ind - t_win_pm < 0:
                if end_mode == 'wrap':
                    times_win = np.concat((self.times[time_ind - t_win_pm:], self.times[:time_ind + t_win_pm]))
                    z_win = np.concat((self.z_calibrated[time_ind - t_win_pm:], self.z_calibrated[:time_ind + t_win_pm]))
                elif end_mode == 'pad':
                    times_win = np.concat((np.zeros(t_win_pm - time_ind), self.times[:time_ind + t_win_pm]))
                    z_win = np.concat((np.zeros(t_win_pm - time_ind), self.z_calibrated[:time_ind + t_win_pm]))
            elif time_ind + t_win_pm > len(self.times):
                if end_mode == 'wrap':
                    times_win = np.concat((self.times[time_ind - t_win_pm:], self.times[:time_ind + t_win_pm - len(self.times)]))
                    z_win = np.concat((self.z_calibrated[time_ind - t_win_pm:], self.z_calibrated[:time_ind + t_win_pm - len(self.times)]))
                elif end_mode == 'pad':
                    times_win = np.concat((self.times[time_ind - t_win_pm:], np.zeros(time_ind + t_win_pm - len(self.times))))
                    z_win = np.concat((self.z_calibrated[time_ind - t_win_pm:], np.zeros(time_ind + t_win_pm - len(self.times))))
            else:
                times_win = self.times[time_ind - t_win_pm:time_ind + t_win_pm]
                z_win = self.z_calibrated[time_ind - t_win_pm:time_ind + t_win_pm]
        else:
            t_win_pm *= 2
            if time_ind - t_win_pm < 0:
                times_win = self.times[time_ind:time_ind + t_win_pm]
                z_win = self.z_calibrated[time_ind:time_ind + t_win_pm]
            else:
                times_win = self.times[time_ind - t_win_pm:time_ind]
                z_win = self.z_calibrated[time_ind - t_win_pm:time_ind]

        if pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            # ax.plot((times_win - np.mean(times_win))*1e6, z_raw_win, label='Raw')
            ax.plot((times_win - np.mean(times_win))*1e6, z_win*1e9)
            ax.set_xlabel(r'Time [$\mu$s]')
            ax.set_ylabel('$z$ response [nm]')
            ax.set_title('{:.0f} keV/c impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            # ax.legend()
            ax.grid(which='both')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return times_win, z_win

    def search_all_data(self, file_num=None, max_windows=1000000, freq_domain_pdf=None, time_domain_pdf=None):
        """Searches through all segments of the file and reconstructs the largest impulse in each.

        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param max_windows: maximum number of windows to search, defaults to 1000000
        :type max_windows: int, optional
        :param freq_domain_pdf: PDF in which to save the frequency domain figure, defaults to None
        :type freq_domain_pdf: PdfPages, optional
        :param time_domain_pdf: PDF in which to save the time domain figure, defaults to None
        :type time_domain_pdf: PdfPages, optional
        """
        search_win = int(self.search_window*self.f_samp)
        time_win = int(self.t_window*self.f_samp)
        impulses = []
        deconvolved_pulses = []
        recon_impulse_inds = []
        pulse_times = []
        
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
            z_win = np.copy(self.z_calibrated[start_idx:end_idx])
            pulse, imp_ind, imp_time = self.compute_impulse(times_win, z_win, file_num=file_num, impulse_num=i, \
                                                            freq_domain_pdf=freq_domain_pdf, time_domain_pdf=time_domain_pdf)
            impulses.append(pulse[imp_ind])
            deconvolved_pulses.append(pulse)
            recon_impulse_inds.append(imp_ind)
            pulse_times.append(imp_time)
        
        self.impulses = np.array(impulses)
        self.deconvolved_pulses = np.array(deconvolved_pulses)
        self.recon_impulse_inds = np.array(recon_impulse_inds)
        self.pulse_times = np.array(pulse_times)

    def get_impulse_array(self, impulse_inds, global_params=True, file_num=0, pdfs=None):
        """Computes the impulse at each index.

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
        impulses = []
        deconvolved_pulses = []
        recon_impulse_inds = []
        pulse_times = []
        resonance_params = []
        for i, ind in enumerate(impulse_inds):
            if self.verbose:
                print('    -> Computing impulse for kick at t={:.5f} seconds...'.format(self.times[ind]))
            times_win, z_win = self.get_time_window(ind, t_window=1e-1, centered=False)
            # times_win, z_raw_win, z_filt_win = self.get_time_window(ind, t_window=self.t_window, centered=True, \
            #                                                         file_num=file_num, impulse_num=i, pdf=impulse_win_pdf)
            p, success = self.fit_susceptibility(z_win, file_num, i, res_fit_pdf)
            if not success:
                print(f'Fit for impulse {i + 1} failed! Skipping')
                continue
            resonance_params.append(p)
            times_win, z_win = self.get_time_window(ind, t_window=self.t_window, centered=True, \
                                                    file_num=file_num, impulse_num=i, pdf=impulse_win_pdf, \
                                                    end_mode='pad')
            pulse, imp_ind, imp_time = self.compute_impulse(times_win, z_win, file_num, i, [p, None][int(global_params)], \
                                                            optimal_filter_pdf, time_domain_pdf, freq_domain_pdf)
            impulses.append(pulse[imp_ind])
            deconvolved_pulses.append(pulse)
            recon_impulse_inds.append(imp_ind)
            pulse_times.append(imp_time)
        
        self.resonance_params = np.array(resonance_params)
        self.impulses = np.array(impulses)
        self.deconvolved_pulses = np.array(deconvolved_pulses)
        self.recon_impulse_inds = np.array(recon_impulse_inds)
        self.pulse_times = np.array(pulse_times)

    def fit_susceptibility(self, z_win, file_num=None, impulse_num=None, pdf=None):
        """Fits the mechanical susceptibility of the trapped nanosphere.

        :param z_win: z data
        :type z_win: numpy.ndarray
        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param impulse_num: impulse number, defaults to None
        :type impulse_num: int, optional
        :param pdf: PDF in which to save the susceptiblity fit, defaults to None
        :type pdf: PdfPages, optional
        :return: best-fit parameters for the resonance and a success flag
        :rtype: tuple
        """
        nperseg = len(z_win)//8
        freq_filt, Pxx_filt = sig.welch(z_win, fs=self.f_samp, window='tukey', nperseg=nperseg)
        fit_bw_hz = 10*self.f_samp/nperseg
        f_win_pm = int(fit_bw_hz/2./np.diff(freq_filt)[0])
        peak_ind = f_win_pm + np.argmax(Pxx_filt[f_win_pm:])

        freq_fit = freq_filt[peak_ind - f_win_pm:peak_ind + f_win_pm + 1]
        Pxx_fit = Pxx_filt[peak_ind - f_win_pm:peak_ind + f_win_pm + 1]

        omega = 2*np.pi*freq_fit
        sigma = 0.1
        floor = np.finfo(float).tiny
        y = np.log(np.maximum(Pxx_fit, floor))

        def log_abs_susc2(omega, A, omega_0, gamma):
            model = abs_susc2(omega, A, omega_0, gamma)
            return np.log(np.maximum(model, floor))

        lsq = LeastSquares(omega, y, sigma, log_abs_susc2)

        m = Minuit(lsq, A=self.amplitude, omega_0=self.omega_0, gamma=self.gamma)
        m.limits['A'] = (0, np.inf)
        m.limits['omega_0'] = (np.amin(omega), np.amax(omega))
        m.limits['gamma'] = (0, np.inf)
        m.migrad()
        p = m.values
        success = m.valid
        # if not success:
        #     print('Error: fitting failed!')
        #     p = np.array((self.amplitude, self.omega_0, self.gamma))

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
                          '$\\omega_0={:.3f}\\times10^{{{:.0f}}}'.format(mant[1], exp[1])  + '~\\mathrm{s^{-1}}$\n' + \
                          '$\\gamma={:.3f}\\times10^{{{:.0f}}}'.format(mant[2], exp[2]) + '~\\mathrm{s^{-1}}$')#\n' \
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

    def deconvolve_response(self, times_win, z_win, file_num=None, impulse_num=None, params=None, \
                            optimal_filter_pdf=None, freq_domain_pdf=None):
        """Deconvolve the response of the oscillator from the position data.

        :param times_win: array of times around the impulse
        :type times_win: numpy.ndarray
        :param z__win: filtered z data around the impulse
        :type z_win: numpy.ndarray
        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param impulse_num: impulse_number, defaults to None
        :type impulse_num: int, optional
        :param params: resonance params to use. If None, uses global parameters
        :type params: numpy.ndarray
        :param optimal_filter_pdf: PDF in which to save the optimal filter figure, defaults to None
        :type optimal_filter_pdf: PdfPages, optional
        :param freq_domain_pdf: PDF in which to save the frequency domain figure, defaults to None
        :type freq_domain_pdf: PdfPages, optional
        :return: force spectrum and corresponding frequencies
        :rtype: tuple of numpy.ndarray
        """

        z_fft = np.fft.rfft(z_win)*np.sqrt(2)/len(z_win)
        freq_fft = np.fft.rfftfreq(n=len(z_win), d=1./self.f_samp)
        if params is None:
            chi = susc(2.*np.pi*freq_fft, 1., self.omega_0, self.gamma)
            # chi = self.susceptibility(2*np.pi*freq_fft)
        else:
            chi = susc(2.*np.pi*freq_fft, 1., params[1], self.gamma)

        C = 5e-23
        # C = self.noise_floor/1e5**2#params[0]**2
        # # print(self.noise_floor/params[0]**2)
        # C = self.noise_floor/(self.amplitude*self.gamma*self.omega_0)**2/2.
        # print(C)
        # C = 100*self.noise_floor/(5e5)**2#p[0]**2

        # model the noise PSD as proportional to the susceptibility plus constant imprecision noise
        J_psd = np.abs(chi)**2 + C

        # construct the optimal filter from the susceptibility and the noise
        optimal = np.conj(chi)/J_psd

        # apply the optimal filter to deconvolve the response and get the force spectrum
        F_fft = z_fft*optimal

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

        if freq_domain_pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.semilogy(freq_fft*1e-3, np.abs(z_fft)**2, label=r'$\tilde{z}(\omega)$')
            ax.semilogy(freq_fft*1e-3, np.abs(chi)**2, label=r'$\chi(\omega)$')
            ax.semilogy(freq_fft*1e-3, J_psd, label=r'$J(\omega)$')
            # ax.semilogy(freq_fft*1e-3, np.abs(F_fft)**2, label=r'$\tilde{F}(\omega)$')
            ax.legend(ncol=2)
            ax.set_xlabel('Frequency [kHz]')
            ax.set_xlim([0, 1e2])
            ax.set_ylabel('Magnitude [au]')
            # ax.set_ylim([1e-12, 2e0])
            if self.pulse_amp_keV:
                ax.set_title('{:.0f} keV/c impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            else:
                ax.set_title('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
            freq_domain_pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        # compute the force from the product of Z(omega) and the optimal filter
        return F_fft, freq_fft

    def compute_impulse(self, times_win, z_win, file_num=None, impulse_num=None, params=None, \
                        optimal_filter_pdf=None, time_domain_pdf=None, freq_domain_pdf=None):
        """Compute the impulse imparted by a kick.

        :param times_win: array of times around the impulse
        :type times_win: numpy.ndarray
        :param z_win: z data around the impulse
        :type z_win: numpy.ndarray
        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param impulse_num: impulse_number, defaults to None
        :type impulse_num: int, optional
        :param params: resonance params to use. If None, uses global parameters
        :type params: numpy.ndarray
        :param optimal_filter_pdf: PDF in which to save the optimal filter figure, defaults to None
        :type optimal_filter_pdf: PdfPages, optional
        :param time_domain_pdf: PDF in which to save the time domain figure, defaults to None
        :type time_domain_pdf: PdfPages, optional
        :param freq_domain_pdf: PDF in which to save the frequency domain figure, defaults to None
        :type freq_domain_pdf: PdfPages, optional
        :return: the waveform containing the pulse, the index of the impulse, and the time of the window
        :rtype: tuple
        """

        F_fft, _ = self.deconvolve_response(times_win, z_win, file_num=file_num, impulse_num=impulse_num, \
                                            params=params, optimal_filter_pdf=optimal_filter_pdf, \
                                            freq_domain_pdf=freq_domain_pdf)

        # deconvolve to get force (Wiener filter picture) or get amplitude of signals matching template 
        # vs time (optimal filter picture)
        f_td = np.fft.irfft(F_fft, n=len(z_win))*len(z_win)/np.sqrt(2)

        lpf = sig.butter(3, self.f_cutoff[1], btype='lowpass', output='sos', fs=self.f_samp)
        f_filt = sig.sosfiltfilt(lpf, f_td)
        f_filt = np.copy(f_td)
        f_filt[:1000] = 0
        f_filt[-1000:] = 0
        f_td[:1000] = 0
        f_td[-1000:] = 0

        # find the maximum within the search window
        width = int(self.search_window*self.f_samp)
        imp_ind = len(times_win)//2 - width//2 + np.argmax(np.abs(f_filt[len(times_win)//2 - width//2:\
                                                             len(times_win)//2 + width//2]))
        impulse = f_filt[imp_ind]

        if time_domain_pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.plot((times_win - times_win[imp_ind])*1e6, z_win*1e9, label='$z$ position')
            ax.axvline((times_win[len(times_win)//2] - times_win[imp_ind])*1e6, \
                       color='C4', lw=1, ls='--', label='Impulse time', zorder=10)
            ax2 = ax.twinx()
            ax2.plot((times_win - times_win[imp_ind])*1e6, f_td, color='C1', label='Force')
            ax2.plot((times_win - times_win[imp_ind])*1e6, f_filt, color='C2', label='Force filtered')
            ax2.plot(0, impulse, color='C3', marker='.', ms=5, label='Reconstructed', zorder=10)
            ax.set_xlabel(r'Time [$\mu$s]')
            ax.set_ylabel('$z$ response [nm]')
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

        return f_filt, imp_ind, times_win[0]

    def integrate_noise(self):
        """Integrate the force noise spectrum.
        """
        F_fft, freqs = self.deconvolve_response(self.times, self.z_calibrated)

        # plt.figure()
        # plt.semilogy(freqs, np.abs(F_fft))
        # plt.xlim([0, 1e5])
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
        self.path = path
        self.file_paths = glob(path + '_*.hdf5')
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
        impulses = []
        res_params = []
        pulses = []
        recon_impulse_inds = []
        pulse_times = []
        timestamps = []

        if self.verbose:
            print('Loading files starting with\n{}'.format(self.path))

        for i, fp in zip(self.file_inds, self.file_paths):
            if self.verbose:
                print('  Loading file {}...'.format(i+1))
            nf = NanoFile(fp, f_cutoff=self.f_cutoff, t_window=self.t_window, search_window=self.search_window, \
                          verbose=self.verbose)
            nf.calibrate_pulse_amp(pulse_amps_1e, pulse_amps_V)
            nf.compute_and_fit_psd(file_num=i, pdf=self.spectra_pdf)
            impulse_inds = nf.get_impulse_inds(file_num=i, pdf=self.impulse_times_pdf)
            if noise:
                impulse_inds = nf.get_noise_inds(impulse_inds)
            nf.get_impulse_array(impulse_inds=impulse_inds, global_params=global_params, file_num=i, \
                                 pdfs=(self.time_domain_pdf, self.freq_domain_pdf, self.optimal_filter_pdf, \
                               self.res_fit_pdf, self.impulse_win_pdf))
            impulses.append(nf.impulses.copy())
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

            del nf
            gc.collect()

        self.impulses = np.concatenate(impulses)
        self.resonance_params = np.concatenate(res_params)
        self.pulses = np.concatenate(pulses)
        self.recon_impulse_inds = np.concatenate(recon_impulse_inds)
        self.pulse_times = np.concatenate(pulse_times)
        self.timestamps = np.array(timestamps)
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
        impulses = []
        pulses = []
        recon_impulse_inds = []

        for i, fp in zip(self.file_inds, self.file_paths):
            if self.verbose:
                print('Loading file {}...'.format(i+1))
            nf = NanoFile(fp, f_cutoff=self.f_cutoff, t_window=self.t_window, verbose=self.verbose)
            nf.compute_and_fit_psd(file_num=i, pdf=self.spectra_pdf)
            nf.search_all_data(file_num=i, max_windows=self.max_windows, freq_domain_pdf=self.freq_domain_pdf, \
                               time_domain_pdf=self.time_domain_pdf)
            impulses.append(nf.impulses.copy())
            pulses.append(nf.deconvolved_pulses.copy())
            recon_impulse_inds.append(nf.recon_impulse_inds)
            if i == 0 and compute_sensitivity:
                    self.integrated_noise = nf.integrate_noise()

            del nf
            gc.collect()

        self.impulses = np.concatenate(impulses)
        self.pulses = np.concatenate(pulses)
        self.recon_impulse_inds = np.concatenate(recon_impulse_inds)
        self.close_pdfs()