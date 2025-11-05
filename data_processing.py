import h5py
import numpy as np
from glob import glob
from pathlib import Path
import gc

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

def abs_susc2(omega, A, omega_0, gamma):#, C):
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
    if (gamma < 0):# or (C < 0):
        return -np.inf
    return A**2*np.abs(1/(omega_0**2 - omega**2 - 1j*gamma*omega))**2# + C**2

def voigt_scaled(omega, A, omega_0, gamma, sigma, C=1.5e-12):
    return np.abs(A)*voigt_profile(omega - omega_0, gamma, sigma) + np.abs(C)

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
        """
        self.file_path = file_path
        charge_p = file_path.split('_p')[-1].split('e_')[0]
        charge_n = file_path.split('_n')[-1].split('e_')[0]
        self.n_charges = float([charge_p, charge_n][len(charge_p) > len(charge_n)])
        self.pulse_amp_V = float(file_path.split('v_')[0].split('_')[-1])
        self.f_cutoff = f_cutoff
        self.t_window = t_window
        # pulse height is undefined until the calibration is done
        self.pulse_amp_keV = np.empty(100)
        self.pulse_amp_keV[:] = np.nan 
        self.verbose = verbose

    def calibrate_pulse_amp(self, pulse_amps_1e, pulse_amps_V):
        """Loads the calibration factors to convert pulse amplitudes into keV/c.

        :param pulse_amps_1e: amplitude of pulses in keV/c assuming a 1e charge
        :type pulse_amps_1e: list
        :param pulse_amps_V: amplitude of pulses delivered in volts
        :type pulse_amps_V: list
        """
        self.pulse_amp_keV = pulse_amps_1e[np.argmin(np.abs(pulse_amps_V - self.pulse_amp_V))]*self.n_charges

    def get_force_array(self, file_num=0, pdfs=None, impulse_thresh=1., params=None):
        """Computes the force at each impulse in the data and returns the results in an array.

        :param file_num: file number, defaults to 0
        :type file_num: int, optional
        :param pdfs: PDFs in which to save the figures, defaults to None
        :type pdfs: string, optional
        :param impulse_thresh: bias threshold for finding impulses, defaults to 1.
        :type impulse_thresh: float, optional
        :param params: parameters defining the resonance, defaults to None
        :type params: list, optional
        """
        if pdfs:
            pulses_pdf, res_pdf, imp_pdf, spectra_pdf, time_pdf = pdfs
        else:
            pulses_pdf, res_pdf, imp_pdf, spectra_pdf, time_pdf = None, None, None
        self.load_file(file_num=file_num, pdf=spectra_pdf)
        impulse_inds = self.get_impulse_inds(file_num=file_num, pdf=time_pdf, impulse_thresh=impulse_thresh)
        forces = []
        resonance_params = []
        for i, ind in enumerate(impulse_inds):
            if self.verbose:
                print('-> Computing force for impulse at t={:.5f} seconds...'.format(self.times[ind]))
            times_win, z_raw_win, z_filt_win = self.get_impulse_window(ind, file_num, i, imp_pdf)
            if params is None:
                p, success = self.fit_susceptibility(z_filt_win, file_num, i, res_pdf)
                if not success:
                    continue
                resonance_params.append(p)
                forces.append(self.compute_force(p, times_win, z_filt_win, file_num, i, pulses_pdf))
            else:
                forces.append(self.compute_force(params, times_win, z_filt_win, file_num, i, False))
        
        self.resonance_params = np.array(resonance_params)
        self.forces = np.array(forces)

    def load_file(self, file_num=None, pdf=None):
        """Load data from the HDF5 file and do some preliminary processing.

        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param pdf: PDF in which to save the spectra figure, defaults to None
        :type pdf: PdfPages, optional
        """

        with h5py.File(self.file_path, 'r') as f:
            self.z_raw = np.array(f['data/channel_d'])*f['data/channel_d'].attrs['adc2mv']*1e-3
            self.imp_raw = np.array(f['data/channel_g'])*f['data/channel_g'].attrs['adc2mv']*1e-3
            self.mon_raw = np.array(f['data/channel_f'])*f['data/channel_f'].attrs['adc2mv']*1e-3

            self.f_samp = 1./f['data'].attrs['delta_t']
            self.n_samp = len(self.z_raw)
            self.t_int = self.n_samp/self.f_samp
            self.times = np.arange(0, self.t_int, 1/self.f_samp)

            bandpass = sig.butter(3, [self.f_cutoff[0], 2*self.f_cutoff[1]], btype='bandpass', output='sos', fs=self.f_samp)
            self.z_filtered = sig.sosfiltfilt(bandpass, self.z_raw)

            self.freqs, Pxx_z_raw = sig.welch(self.z_raw, fs=self.f_samp, noverlap=0, nperseg=self.f_samp*0.01)
            _, Pxx_z_filt = sig.welch(self.z_filtered, fs=self.f_samp, noverlap=0, nperseg=self.f_samp*0.01)
            _, Pxx_mon_raw = sig.welch(self.mon_raw, fs=self.f_samp, noverlap=0, nperseg=self.f_samp*0.01)

            p = self.fit_voigt_profile(Pxx_z_raw)

            if pdf:
                fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
                ax.semilogy(self.freqs*1e-3, Pxx_z_raw, alpha=1.0, label='$z$ raw')
                ax.semilogy(self.freqs*1e-3, Pxx_z_filt, alpha=0.5, label='$z$ filtered')
                ax.semilogy(self.freqs*1e-3, p[0]*voigt_profile(2*np.pi*self.freqs - p[1], p[2], p[3]) + p[4], alpha=0.5, label='Voigt fit')
                ax.semilogy(self.freqs*1e-3, Pxx_mon_raw, alpha=1.0, label='Monitoring raw')
                ax.set_title('File {}'.format(file_num + 1))
                ax.set_xlabel('Frequency [kHz]')
                ax.set_ylabel(r'PSD [$\mathrm{V^2/Hz}$]')
                ax.text(21, 1e-6, 'Test 1')
                ax.set_ylim([1e-14, 1e-4])
                ax.set_xlim([20, 40])
                ax.legend(loc='upper right')
                ax.grid(which='both')
                pdf.savefig(fig, dpi=150)
                fig.clf()
                plt.close()
                del fig, ax
                gc.collect()

    def fit_voigt_profile(self, Pxx_z_filt):
        f_0_range = [27.5e3, 32.5e3]
        f_0_inds = [np.argmin(np.abs(self.freqs - f)) for f in f_0_range]
        omega_0_guess = 2*np.pi*self.freqs[f_0_inds[0] + np.argmax(Pxx_z_filt[f_0_inds[0]:f_0_inds[1]])]
        gamma_guess = 2*np.pi*0.3
        sigma_guess = 2*np.pi*1e2
        A_guess = 1e-3 #np.amax(Pxx_z_filt)*gamma_guess**2*omega_0_guess**2
        C_guess = 1.5e-12
        p0 = [A_guess, omega_0_guess, sigma_guess, gamma_guess]

        fit_bw = 2*np.pi*5e3
        fit_inds = (self.freqs > omega_0_guess - fit_bw/2.) & (self.freqs < omega_0_guess + fit_bw/2.)

        voigt_scaled = lambda omega, A, omega_0, sigma, gamma, C: np.abs(A)*voigt_profile(omega - omega_0, sigma, gamma) + np.abs(C)
        try:
            p, _ = curve_fit(voigt_scaled, 2*np.pi*self.freqs[fit_inds], Pxx_z_filt[fit_inds], p0=p0, \
                             bounds=((0, 2*np.pi*f_0_range[0], 0, 0, 0), (np.inf, 2*np.pi*f_0_range[1], np.inf, np.inf, np.inf)), \
                             sigma=0.1*np.sqrt(Pxx_z_filt[fit_inds]))
            print('Fit succeeded')
            print(p)
        except:
            print('Fit failed')
            print(p0)
            return p0

        return p

    def integrate_noise(self):
        pass

    def get_impulse_inds(self, impulse_thresh=1., file_num=None, pdf=None):
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
        impulse_inds = np.hstack((impulse_inds[0, 0], impulse_inds[1:, 0][np.diff(impulse_inds[:,0]) > 5]))

        if pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.plot(self.times, self.imp_raw)
            ax.set_ylabel('Impulse [V]')
            ax.set_xlabel('Time [s]')
            ax.set_title('File {}'.format(file_num + 1))
            ax.plot(self.times[impulse_inds], self.imp_raw[impulse_inds], ls='none', marker='.')
            ax.grid(which='both')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return impulse_inds

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
        times_win = self.times[impulse_ind - t_win_pm:impulse_ind + t_win_pm]
        z_raw_win = self.z_raw[impulse_ind - t_win_pm:impulse_ind + t_win_pm]
        z_filt_win = self.z_filtered[impulse_ind - t_win_pm:impulse_ind + t_win_pm]

        if pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.plot(times_win, z_raw_win, label='Raw')
            ax.plot(times_win, z_filt_win, label='Filtered')
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('$z$ response [V]')
            ax.set_title('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
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
        freq_filt, Pxx_filt = sig.welch(z_filt_win[:len(z_filt_win)//2], \
                                        fs=self.f_samp, noverlap=0, nperseg=self.f_samp*0.001)

        peak_ind = np.argmax(Pxx_filt)
        fit_bw_hz = 4e3
        f_win_pm = int(fit_bw_hz/2./np.diff(freq_filt)[0])

        freq_fit = freq_filt[peak_ind - f_win_pm:peak_ind + f_win_pm + 1]
        Pxx_fit = Pxx_filt[peak_ind - f_win_pm:peak_ind + f_win_pm + 1]

        omega = 2*np.pi*freq_fit
        omega_0_guess = 2*np.pi*freq_fit[np.argmax(Pxx_fit)]
        gamma_guess = 2*np.pi*1e3
        C_guess = np.sqrt(Pxx_fit[-1])
        A_guess = np.amax(Pxx_fit)*gamma_guess**2*omega_0_guess**2
        p0 = (A_guess, omega_0_guess, gamma_guess)#, C_guess)
        fit_log = False
        try:
            if fit_log:
                fit_func = lambda omega, A, omega_0, gamma: np.log(abs_susc2(omega, A, omega_0, gamma))
                p, _ = curve_fit(fit_func, omega, np.log(Pxx_fit), p0=p0)
            else:
                p, _ = curve_fit(abs_susc2, omega, Pxx_fit, p0=p0)#, sigma=Pxx_fit, absolute_sigma=True)
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
            ax.set_title('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
            ax.legend(loc='upper right')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return p, success

    def compute_force(self, p, times_win, z_filt_win, file_num=None, impulse_num=None, pdf=None):
        """Compute the force imparted by an impulse.

        :param p: parameters describing the resonance
        :type p: numpy.ndarray
        :param times_win: array of times around the impulse
        :type times_win: numpy.ndarray
        :param z_filt_win: filtered z data around the impulse
        :type z_filt_win: numpy.ndarray
        :param file_num: file number, defaults to None
        :type file_num: int, optional
        :param impulse_num: impulse_number, defaults to None
        :type impulse_num: int, optional
        :param pdf: PDF in which to save the force figure, defaults to None
        :type pdf: PdfPages, optional
        :return: force imparted by the impulse
        :rtype: float
        """

        z_filt_win = sig.detrend(z_filt_win)
        # z_filt_win *= sig.windows.hann(len(z_filt_win))

        z_fft = np.fft.rfft(z_filt_win)
        freq_fft = np.fft.rfftfreq(n=len(z_filt_win), d=1./self.f_samp)
        # large variations in fitted amplitude can affect the results so we just use a constant
        chi = susc(2.*np.pi*freq_fft, 1., p[1], 2*np.pi*200.)
        # C = 1e-27
        # C = np.abs(p[-1]/p[0]**2)
        C = 0
        # print(C)
        F_fft = z_fft/(chi + C/np.conj(chi))

        f_td = np.fft.irfft(F_fft, n=len(z_filt_win))

        lpf = sig.butter(3, self.f_cutoff[1], btype='lowpass', output='sos', fs=self.f_samp)
        f_filt = sig.sosfiltfilt(lpf, f_td)
        f_filt[:1000] = 0
        f_filt[-1000:] = 0
        f_td[:1000] = 0
        f_td[-1000:] = 0

        force_ind = len(times_win)//2 - 10 + np.argmax(np.abs(f_filt[len(times_win)//2-10:len(times_win)//2+20]))
        force = f_filt[force_ind]

        if pdf:
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.plot(times_win, z_filt_win, label='$z$ position')
            ax2 = ax.twinx()
            ax2.plot(times_win, f_td, color='C1', label='Force')
            ax2.plot(times_win, f_filt, color='C2', label='Force filtered')
            ax2.plot(times_win[force_ind], force, color='C3', marker='.', zorder=100)
            ax.set_xlabel('Time [s]')
            ax.set_ylabel('$z$ response [V]')
            ax2.set_ylabel('Force [au]')
            ax.set_title('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax, ax2
            gc.collect()

            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.semilogy(freq_fft, np.abs(z_fft)**2/np.amax(np.abs(z_fft)**2), label=r'$\tilde{z}(\omega)$')
            ax.semilogy(freq_fft, np.abs(chi)**2/np.amax(np.abs(chi)**2), label=r'$\chi(\omega)$')
            ax.semilogy(freq_fft, np.abs(F_fft)**2/np.amax(np.abs(F_fft)**2), label=r'$\tilde{F}(\omega)$')
            ax.legend()
            ax.set_xlabel('Frequency [Hz]')
            ax.set_xlim([0, 1e5])
            ax.set_ylabel('Magnitude')
            ax.set_ylim([1e-12, 2e0])
            pdf.savefig(fig, dpi=150)
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
        """
        self.file_paths = glob(path + '_*.hdf5')[:max_files]
        self.file_inds = [int(f.split('_')[-1].split('.hdf5')[0]) for f in self.file_paths]
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
            self.pulses_pdf = PdfPages(self.plot_path + '_pulses.pdf')
            self.res_pdf = PdfPages(self.plot_path + '_res_fit.pdf')
            self.imp_pdf = PdfPages(self.plot_path + '_impulses.pdf')
            self.spectra_pdf = PdfPages(self.plot_path + '_spectra.pdf')
            self.time_pdf = PdfPages(self.plot_path + '_times.pdf')
        else:
            self.pulses_pdf, self.res_pdf, self.imp_pdf, self.spectra_pdf, self.time_pdf \
                = None, None, None, None, None

    def close_pdfs(self):
        """Close the PDF files.
        """
        if self.plot_path:
            self.pulses_pdf.close()
            self.res_pdf.close()
            self.imp_pdf.close()
            self.spectra_pdf.close()
            self.time_pdf.close()
    
    def load_files(self, resonance_params=None, pulse_amps_1e=None, pulse_amps_V=None):
        """Load the datasets as NanoFile objects and extract relevant data from them.

        :param resonance_params: global resonance parameters to use, defaults to None
        :type resonance_params: numpy.ndarray, optional
        :param pulse_amps_1e: amplitudes of applied impulses in keV/c assuming 1e charge, defaults to None
        :type pulse_amps_1e: list, optional
        :param pulse_amps_V: amplitudes of applied impulses in V, defaults to None
        :type pulse_amps_V: list, optional
        """
        forces = []
        res_params = []

        for i, fp in zip(self.file_inds, self.file_paths):
            if self.verbose:
                print('Loading file {}...'.format(i+1))
            nf = NanoFile(fp, f_cutoff=self.f_cutoff, t_window=self.t_window, verbose=self.verbose)
            nf.calibrate_pulse_amp(pulse_amps_1e, pulse_amps_V)
            nf.get_force_array(params=resonance_params, file_num=i, \
                               pdfs=(self.pulses_pdf, self.res_pdf, self.imp_pdf, self.spectra_pdf, self.time_pdf))
            forces.append(nf.forces.copy())
            if res_params is None:
                res_params = [nf.resonance_params.copy()]
            else:
                res_params.append(nf.resonance_params.copy())
            if i == 0:
                self.pulse_amp_keV = nf.pulse_amp_keV.copy()

            del nf
            gc.collect()

        self.forces = np.concatenate(forces)
        self.resonance_params = np.concatenate(res_params)
        self.close_pdfs()