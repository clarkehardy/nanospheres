import h5py
import numpy as np
from glob import glob
from pathlib import Path
import gc
import yaml

from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import scipy.signal as sig
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.integrate import trapezoid

from iminuit import Minuit
from iminuit.cost import LeastSquares

c = 299792458.  # speed of light, m/s
e = 1.602176634e-19 # unit charge, C

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
    return A/(omega_0**2 - omega**2 - 1j*gamma*omega)

class NanoFile:
    """Class to handle an individual file containing nanosphere data.
    """

    def __init__(self, file_path, f_cutoff=[2e4, 1e5], t_window=1e-3, search_window=5e-5, \
                 fit_window=1e1, d_sphere_nm=166., calibrate=False, cal_factors=[1, 0], \
                 verbose=False, apply_notch=False, config=None):
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
        :param apply_notch: whether to build and apply notch filters, defaults to False
        :type apply_notch: bool, optional
        """
        self.file_path = file_path
        self.f_cutoff = f_cutoff
        self.t_window = t_window
        self.fit_window = fit_window
        self.calibrate = calibrate
        self.search_window = search_window
        self.window_func = 'tukey'
        self.verbose = verbose
        self.apply_notch = apply_notch
        self.pulse_amp_keV = None
        self.cal_factors = cal_factors
        if config:
            self.load_config(config)
        V_ns = (4/3.)*np.pi*(d_sphere_nm*1e-9/2.)**3
        rho_silica = 2.65e3 # density of silica, kg/m^3
        self.mass_sphere = rho_silica*V_ns # mass of the sphere in kg
        self.load_file()

    def load_config(self, config):
        """
        Load class attributes from a config file. These will supersede any
        values passed as input arguments to the init method.
        
        :param config: Either a dictionary or a path to a YAML file
        :type config: dict or str or Path
        """
        # If config is a path (string or Path object), load the YAML file
        if isinstance(config, (str, Path)):
            with open(config, 'r') as f:
                config_dict = yaml.safe_load(f)
        # If config is already a dictionary, use it directly
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise TypeError(f"config must be a dictionary or a path (str/Path), got {type(config)}")
        
        # Set class attributes from the dictionary
        for key, value in config_dict.items():
            setattr(self, key, value)

    def load_file(self):
        """Load data from the HDF5 file and do some preliminary processing.
        """

        with h5py.File(self.file_path, 'r') as f:
            self.z_raw = np.array(f['data/channel_d'])*f['data/channel_d'].attrs['adc2mv']*1e-3
            try:
                self.imp_raw = np.array(f['data/channel_g'])*f['data/channel_g'].attrs['adc2mv']*1e-3
                self.mon_raw = np.array(f['data/channel_f'])*f['data/channel_f'].attrs['adc2mv']*1e-3
            except:
                self.imp_raw = None
                self.mon_raw = None

            self.f_samp = 1./f['data'].attrs['delta_t']

            self.timestamp = f['data'].attrs['timestamp']

        self.n_samp = len(self.z_raw)
        self.t_int = self.n_samp/self.f_samp
        self.times = np.arange(0, self.t_int, 1/self.f_samp)

    def compute_sensitivity(self, Fxx, freqs, freq_range=[30e3, 70e3]):
        """Computes the force sensitivity from a force spectrum.

        :param Fxx: the force power spectral density
        :type Fxx: numpy.ndarray
        :param freqs: the frequency array corresponding to Fxx
        :type freqs: numpy.ndarray
        :param freq_range: frequency range for sensitivity calculation in Hz, defaults to [20e3, 100e3]
        :type freq_range: list, optional
        :return: sensitivity in keV/c
        :rtype: float
        """
        ind_1 = np.argmin(np.abs(freqs - freq_range[0]))
        ind_2 = np.argmin(np.abs(freqs - freq_range[1]))
        p_sens = 1/(np.sqrt(trapezoid(1/Fxx[ind_1:ind_2], freqs[ind_1:ind_2])) + 1e-30) * c / (1e3 * e)
        return p_sens

    def calibrate_spectrum(self, Pxx_mon_raw, Pxx_z_raw, *params):
        """Calibrate the position spectrum to Newtons

        :param Pxx_mon_raw: the PSD of the raw monitoring signal
        :type Pxx_mon_raw: numpy.ndarray
        """

        if self.verbose:
            print('  Calibrating volts to meters...')

        Efield = 79. # V/m (when 1 V is applied to lens holder 2)

        if self.mon_raw is not None:
            f_mon = self.freqs[np.argmax(Pxx_mon_raw)]
            force_applied = np.copy(self.mon_raw)
        else:
            f_mon = np.copy(self.drive_freq)
            force_applied = self.drive_amp*np.sin(2*np.pi*f_mon*self.times)

        # bandpass filter around the monitoring frequency
        bp_bw = 1e3
        bp = sig.butter(4, [f_mon - bp_bw/2, f_mon + bp_bw/2], btype='bandpass', output='sos', fs=self.f_samp)
        drive = sig.sosfiltfilt(bp, force_applied)
        resp = sig.sosfiltfilt(bp, self.z_raw)

        # get in-phase and quadrature components of response
        z = sig.hilbert(drive)
        phi = np.unwrap(np.angle(z))
        i_raw = resp*np.cos(phi)
        q_raw = resp*np.sin(phi)
        
        # low-pass filter the demodulated data streams
        f_lpf = 1e2
        lpf = sig.butter(4, f_lpf, btype='low', output='sos', fs=self.f_samp)

        # demodulate the response
        i_filt = 2*sig.sosfiltfilt(lpf, i_raw)
        q_filt = 2*sig.sosfiltfilt(lpf, q_raw)
        R = i_filt + 1j*q_filt

        # demodulate the drive
        i_drive = 2*sig.sosfiltfilt(lpf, drive*np.cos(phi))
        q_drive = 2*sig.sosfiltfilt(lpf, drive*np.sin(phi))
        D = i_drive + 1j*q_drive

        # conversion factor from volts to meters
        meters_per_volt = np.mean(self.n_charges*e*Efield*np.abs(D/R))*np.abs(self.susceptibility(2*np.pi*f_mon))

        if self.verbose:
            print('    Mass of the nanosphere: \t\t{:.3e} kg'.format(self.mass_sphere))
            print('    Number of charges on the sphere: \t{}'.format(int(self.n_charges)))
            print('    Electric field per volt applied: \t{:.1f} V/m'.format(Efield))
            print('    Trap resonant frequency: \t\t{:.1f} kHz'.format(params[1]*1e-3/2/np.pi))
            print('    Drive signal from demodulation: \t{:.3f} V'.format(np.mean(np.abs(D))))
            print('    Applied force from demodulation: \t{:.3e} N'.format(self.n_charges*e*Efield*np.mean(np.abs(D))))
            print('    Sensor response from demodulation: \t{:.3e} V'.format(np.mean(np.abs(R))))
            print('    Susceptibility at {:.1f} kHz drive: \t{:.3e} m/V'.format(f_mon*1e-3, np.abs(self.susceptibility(2*np.pi*f_mon))))
            print('    Nanosphere response amplitude: \t{:.3e} m'.format(self.n_charges*e*Efield*np.mean(np.abs(D))*np.abs(self.susceptibility(2*np.pi*f_mon))))
            print('    Position calibration factor: \t{:.3e} m/V'.format(meters_per_volt))

        return meters_per_volt

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

        nperseg = 2*len(self.z_raw)//120
        window_s1 = np.sum(sig.get_window(self.window_func, nperseg))
        window_s2 = np.sum(sig.get_window(self.window_func, nperseg)**2)
        spectrum_to_density = window_s1/np.sqrt(window_s2*self.f_samp)

        self.freqs, Pxx_z_raw = sig.welch(self.z_raw, fs=self.f_samp, window=self.window_func, nperseg=nperseg, scaling='spectrum')
        _, Pxx_z_filt = sig.welch(self.z_filtered, fs=self.f_samp, window=self.window_func, nperseg=nperseg, scaling='spectrum')
        if self.mon_raw is not None:
            _, Pxx_mon_raw = sig.welch(self.mon_raw, fs=self.f_samp, window=self.window_func, nperseg=nperseg, scaling='spectrum')
        else:
            Pxx_mon_raw = None

        impr_win = 1.5*self.freqs[np.argmax(Pxx_z_filt)] + np.array((0, 1e4))
        impr_inds = [np.argmin(np.abs(n - self.freqs)) for n in impr_win]
        noise_floor = np.mean(Pxx_z_raw[impr_inds[0]:impr_inds[1]])

        p = self.fit_voigt_profile(Pxx_z_raw, noise_floor)

        # Compute fitted spectrum
        fitted_spectrum = p[0]**2*voigt_profile(2*np.pi*self.freqs - p[1], p[2], p[3]) + noise_floor
        
        # Build notch filters and apply them if requested
        if self.apply_notch:
            z_filtered_notched, Pxx_z_filt = self.build_notch_filters(Pxx_z_raw, fitted_spectrum, p, nperseg)
            # Update z_filtered to include notch filters
            self.z_filtered = z_filtered_notched
        else:
            # Set passthrough filters when notch filtering is disabled
            self.notch_freqs = np.array([]).reshape(0, 2)
            self.notch_filters = np.array([[1., 0., 0., 1., 0., 0.]])

        # compute the susceptibility in meters/Newton
        self.susceptibility = lambda omega: susc(omega, 1, p[1], p[3])/self.mass_sphere

        # calibrate from volts to meters
        if self.calibrate:
            self.meters_per_volt = self.calibrate_spectrum(Pxx_mon_raw, Pxx_z_raw, p[0], p[1], p[3])
        else:
            self.meters_per_volt = 5e-8 # default value; placeholder for now

        # apply calibration factor to z data
        self.z_calibrated = self.z_filtered * self.meters_per_volt

        # save resonance parameters as class attributes
        self.gamma = p[-1]
        self.omega_0 = p[1]
        self.amplitude = p[0]*self.meters_per_volt*spectrum_to_density

        self.susc_noise_floor = noise_floor*np.sqrt(2*np.pi)*spectrum_to_density**2/p[0]**2/self.mass_sphere**2/p[3]/p[1]**2

        if pdf:
            Fxx = Pxx_z_filt*(spectrum_to_density*self.meters_per_volt/np.abs(self.susceptibility(2*np.pi*self.freqs)))**2
            p_sens = self.compute_sensitivity(Fxx, self.freqs)

            if self.pulse_amp_keV:
                title = '{:.0f} keV/c impulse, file {}'.format(self.pulse_amp_keV, file_num + 1)
            else:
                title = 'File {}'.format(file_num + 1)
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.semilogy(self.freqs*1e-3, np.sqrt(Pxx_z_raw)*spectrum_to_density, alpha=1.0, label='$z$ raw')
            ax.semilogy(self.freqs*1e-3, np.sqrt(Pxx_z_filt)*spectrum_to_density, alpha=1.0, label='$z$ filtered')
            ax.semilogy(self.freqs*1e-3, np.sqrt((p[0]**2*voigt_profile(2*np.pi*self.freqs - p[1], p[2], p[3]) \
                        + noise_floor))*spectrum_to_density, alpha=1.0, lw=1.0, label='Voigt fit', zorder=100)
            ax.semilogy(self.freqs*1e-3, np.abs(susc(2*np.pi*self.freqs, 1/self.mass_sphere, p[1], p[3])) \
                        *p[0]*self.mass_sphere*np.sqrt(p[3]*p[1]**2/np.sqrt(2*np.pi))*spectrum_to_density, \
                        alpha=1.0, label=r'Scaled $\chi$')
            ax.set_title(title)
            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel(r'Raw ASD [V/$\sqrt{\mathrm{Hz}}$]')
            ax.set_ylim([1e-7, 1e-1])
            ax.set_xlim([20, 100])
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
            ax.semilogy(self.freqs*1e-3, np.sqrt(Pxx_z_filt)*spectrum_to_density*self.meters_per_volt/\
                                         np.abs(self.susceptibility(2*np.pi*self.freqs)), alpha=1.0, label='$z$ raw')
            ax.set_title(title)
            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel(r'Calibrated ASD [$\mathrm{N/\sqrt{Hz}}$]')
            ax.set_xlim([20, 100])
            ax.set_ylim([1e-21, 1e-18])
            ax.text(0.03, 0.97, '$\\Delta p={:.1f}$ keV/c'.format(p_sens), ha='left', va='top', transform=ax.transAxes)
            ax.grid(which='both')
            pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.semilogy(self.freqs*1e-3, np.sqrt(Pxx_z_filt)*spectrum_to_density*self.meters_per_volt)
            ax.set_title(title)
            ax.set_xlabel('Frequency [kHz]')
            ax.set_ylabel(r'Calibrated ASD [$\mathrm{m/\sqrt{Hz}}$]')
            ax.set_xlim([20, 100])
            ax.set_ylim([1e-14, 1e-9])
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

    def build_notch_filters(self, Pxx_z_raw, fitted_spectrum, p, nperseg):
        """Builds notch filters to remove excess frequencies from the power spectrum.

        Identifies frequencies where the power spectrum exceeds a threshold factor times
        the fitted Voigt profile, groups them into contiguous regions, and creates
        bandstop (notch) filters for each region. Filters are designed to avoid the
        resonance frequency and are combined into a single SOS filter.

        :param Pxx_z_raw: the raw power spectral density
        :type Pxx_z_raw: numpy.ndarray
        :param fitted_spectrum: the fitted Voigt profile spectrum
        :type fitted_spectrum: numpy.ndarray
        :param p: the Voigt profile fit parameters [A, omega_0, sigma, gamma]
        :type p: numpy.ndarray
        :param nperseg: number of samples per segment for Welch's method
        :type nperseg: int
        :return: filtered z data and updated power spectral density
        :rtype: tuple of numpy.ndarray
        """
        # Get resonance frequency in Hz (p[1] is omega_0 in rad/s)
        f_resonance = p[1] / (2 * np.pi)
        resonance_protection_band = 1e3  # 1 kHz protection band around resonance
        
        # Identify frequencies where power exceeds threshold (default 3x the fit)
        # This can be adjusted later by the user
        threshold_factor = 3
        excess_mask = Pxx_z_raw > threshold_factor * fitted_spectrum
        
        # Find contiguous regions of excess frequencies
        notch_regions = []
        if np.any(excess_mask):
            # Find where the mask changes (edges of regions)
            diff_mask = np.diff(excess_mask.astype(int))
            start_inds = np.where(diff_mask == 1)[0] + 1  # Start of excess regions
            end_inds = np.where(diff_mask == -1)[0] + 1   # End of excess regions
            
            # Handle edge cases
            if excess_mask[0]:
                start_inds = np.concatenate(([0], start_inds))
            if excess_mask[-1]:
                end_inds = np.concatenate((end_inds, [len(excess_mask)]))
            
            # Group into contiguous regions and create notch filters
            for start_idx, end_idx in zip(start_inds, end_inds):
                if end_idx > start_idx:  # Valid region
                    f_low = self.freqs[start_idx]
                    f_high = self.freqs[end_idx - 1]
                    # Add minimal padding to ensure we filter the entire region
                    f_width = (f_high - f_low) * 1.2  # 30% reduction from region width
                    # Ensure minimum width (at least 5 frequency bins)
                    min_width = 5 * np.diff(self.freqs)[0]
                    # f_width = max(f_width, min_width)
                    f_center = (f_low + f_high) / 2
                    f_notch_low = max(f_center - f_width/2, self.freqs[0])
                    f_notch_high = min(f_center + f_width/2, self.freqs[-1])
                    
                    # Check if notch region is within protection band of resonance
                    notch_center = (f_notch_low + f_notch_high) / 2
                    distance_from_resonance = np.abs(notch_center - f_resonance)
                    # Also check if the notch region overlaps with the protection band
                    notch_overlaps_resonance = not ((f_notch_high < f_resonance - resonance_protection_band) or 
                                                     (f_notch_low > f_resonance + resonance_protection_band))
                    
                    # Only create notch if it's within reasonable range and not near resonance
                    if f_notch_low < f_notch_high and not notch_overlaps_resonance:
                        notch_regions.append((f_notch_low, f_notch_high))
        
        # Save notch frequencies as numpy array
        if notch_regions:
            self.notch_freqs = np.array(notch_regions)
        else:
            self.notch_freqs = np.array([]).reshape(0, 2)
        
        # Construct combined notch filter from all notch regions
        notch_sos_list = []
        for f_low, f_high in notch_regions:
            # Create bandstop (notch) filter using butterworth with lower order for gentler filtering
            # Only apply if within the bandpass range
            if f_low >= self.f_cutoff[0] and f_high <= self.f_cutoff[1]:
                # Use 1st order filter for very gentle filtering
                notch_sos = sig.butter(1, [f_low, f_high], btype='bandstop', output='sos', fs=self.f_samp)
                notch_sos_list.append(notch_sos)
        
        # Combine all notch filters into a single SOS filter by concatenating
        if notch_sos_list:
            self.notch_filters = np.vstack(notch_sos_list)
        else:
            # Create a passthrough filter (identity) if no notches
            self.notch_filters = np.array([[1., 0., 0., 1., 0., 0.]])
        
        # Apply combined notch filter to z_filtered
        z_filtered_notched = sig.sosfiltfilt(self.notch_filters, self.z_filtered)
        
        # Recompute Pxx_z_filt with notch filters applied
        _, Pxx_z_filt = sig.welch(z_filtered_notched, fs=self.f_samp, window=self.window_func, \
                                  nperseg=nperseg, scaling='spectrum')
        
        return z_filtered_notched, Pxx_z_filt

    def calibrate_pulse_amp(self, pulse_amps_1e=None, pulse_amps_V=None):
        """Loads the calibration factors to convert pulse amplitudes into keV/c.

        :param pulse_amps_1e: amplitude of pulses in keV/c assuming a 1e charge
        :type pulse_amps_1e: list
        :param pulse_amps_V: amplitude of pulses delivered in volts
        :type pulse_amps_V: list
        """
        charge_p = self.file_path.split('_p')[-1].split('e_')[0]
        charge_n = self.file_path.split('_n')[-1].split('e_')[0]
        self.n_charges = float([charge_p, charge_n][len(charge_p) > len(charge_n)])
        if (pulse_amps_1e is not None) and (pulse_amps_V is not None):
            self.pulse_amp_V = float(self.file_path.split('v_')[0].split('_')[-1])
            self.pulse_amp_keV = pulse_amps_1e[np.argmin(np.abs(pulse_amps_V - self.pulse_amp_V))]*self.n_charges
        if 'khz' in self.file_path.split('/')[-1]:
            self.drive_freq = 1e3*float(self.file_path.split('e_')[-1].split('khz')[0])
            self.drive_amp = 0.5*float(self.file_path.split('khz_')[-1].split('vpp')[0])

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
        self.convert_to_keV()
        self.recon_impulse_inds = np.array(recon_impulse_inds)
        self.pulse_times = np.array(pulse_times)

    def convert_to_keV(self):
        """Convert the impulses in force units to keV using the calibration
        factors computed separately.
        """
        self.impulses = self.impulses * self.cal_factors[0] + self.cal_factors[1]
        self.deconvolved_pulses = self.deconvolved_pulses * self.cal_factors[0] + self.cal_factors[1]

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
        time_domain_pdf, freq_domain_pdf, optimal_filter_pdf, res_fit_pdf, noise_spectra_pdf, impulse_win_pdf = pdfs
        impulses = []
        deconvolved_pulses = []
        recon_impulse_inds = []
        pulse_times = []
        resonance_params = []
        for i, ind in enumerate(impulse_inds):
            if self.verbose:
                print('    -> Computing impulse for kick at t={:.5f} seconds...'.format(self.times[ind]))
            times_win, z_win = self.get_time_window(ind, t_window=self.fit_window, centered=False)
            p, success = self.fit_susceptibility(z_win, file_num, i, res_fit_pdf, noise_spectra_pdf)
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

    def fit_susceptibility(self, z_win, file_num=None, impulse_num=None, res_fit_pdf=None, noise_spectra_pdf=None):
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
        nperseg = 2*len(z_win)//12
        freq_filt, Pxx_filt = sig.welch(z_win, fs=self.f_samp, window=self.window_func, nperseg=nperseg)
        # Pxx_filt = np.abs(np.fft.rfft(z_win)*np.sqrt(2/len(z_win)/self.f_samp))**2
        # freq_filt = np.fft.rfftfreq(n=len(z_win), d=1./self.f_samp)
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

        plot_freq = np.linspace(freq_filt[peak_ind - 3*f_win_pm], freq_filt[peak_ind + 3*f_win_pm + 1], 1000)
        Fxx = np.abs(np.sqrt(Pxx_filt)/susc(2*np.pi*freq_filt, 1/self.mass_sphere, *p[1:]))**2

        if res_fit_pdf:
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
            res_fit_pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        if noise_spectra_pdf:
            fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True, layout='constrained')
            ax[0].semilogy(freq_filt*1e-3, np.sqrt(Pxx_filt))
            ax[1].semilogy(freq_filt*1e-3, np.sqrt(Fxx))
            ax[0].set_ylabel(r'ASD [m/$\sqrt{\mathrm{Hz}}$]')
            ax[1].set_ylabel(r'ASD [N/$\sqrt{\mathrm{Hz}}$]')
            ax[0].set_xlim([20, 100])
            ax[0].set_ylim([1e-14, 1e-9])
            ax[1].set_ylim([1e-21, 1e-18])
            ax[1].text(0.05, 0.95, '$\\Delta p={:.1f}$ keV/c'.format(self.compute_sensitivity(Fxx, freq_filt)), \
                       ha='left', va='top', transform=ax[1].transAxes)
            for i in range(2):
                ax[i].grid(which='both')
                ax[i].set_xlabel('Frequency [kHz]')
            if self.pulse_amp_keV:
                fig.suptitle('{:.0f} keV/c impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            else:
                fig.suptitle('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
            noise_spectra_pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return p, success

    def deconvolve_response(self, times_win, z_win, file_num=None, impulse_num=None, params=None, \
                            optimal_filter_pdf=None):
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
        :return: force spectrum and corresponding frequencies
        :rtype: tuple of numpy.ndarray
        """

        z_fft = np.fft.rfft(z_win)*np.sqrt(2)/len(z_win)
        freq_fft = np.fft.rfftfreq(n=len(z_win), d=1./self.f_samp)
        if params is None:
            chi = susc(2.*np.pi*freq_fft, 1./self.mass_sphere, self.omega_0, self.gamma)
        else: 
            chi = susc(2.*np.pi*freq_fft, 1./self.mass_sphere, params[1], self.gamma)
            
        C = 5e-23/self.mass_sphere**2

        # model the noise PSD as proportional to the susceptibility plus constant imprecision noise
        J_psd = np.abs(chi)**2 + C

        # construct the optimal filter from the susceptibility and the noise
        optimal = np.conj(chi)/J_psd

        # apply the optimal filter to deconvolve the response and get the force spectrum
        F_fft = z_fft * optimal

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

        return F_fft, z_fft, freq_fft

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

        F_fft, z_fft, freq_fft = self.deconvolve_response(times_win, z_win, file_num=file_num, impulse_num=impulse_num, \
                                                          params=params, optimal_filter_pdf=optimal_filter_pdf)

        # deconvolve to get force (Wiener filter picture) or get amplitude of signals matching template 
        # vs time (optimal filter picture)
        f_td = np.fft.irfft(F_fft, n=len(z_win))/np.sqrt(2)*len(z_win)

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
            ax2.plot((times_win - times_win[imp_ind])*1e6, f_filt*1e18, color='C1', label='Force')
            ax2.plot(0, impulse*1e18, color='C3', marker='.', ms=8, ls='none', label='Reconstructed', zorder=10)
            ax.set_xlabel(r'Time [$\mu$s]')
            ax.set_ylabel('$z$ response [nm]')
            ax2.set_ylabel('Force [aN]')
            ax.legend()
            ax2.legend()
            handles = []
            labels = []
            for axis in [ax, ax2]:
                h, l = axis.get_legend_handles_labels()
                axis.get_legend().remove()
                handles.extend(h)
                labels.extend(l)
            ax2.legend(handles, labels, ncol=2, loc='upper left')
            if self.pulse_amp_keV:
                ax.set_title('{:.0f} keV/c impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            else:
                ax.set_title('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
            time_domain_pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax, ax2
            gc.collect()

        if freq_domain_pdf:
            spectrum_to_density = np.sqrt(len(z_win)/self.f_samp)
            fig, ax = plt.subplots(figsize=(6, 4), layout='constrained')
            ax.semilogy(freq_fft*1e-3, np.abs(z_fft)*spectrum_to_density, color='C0', label=r'$\tilde{z}(\omega)$')
            ax.legend()
            ax2 = ax.twinx()
            ax2.semilogy(freq_fft*1e-3, np.abs(F_fft)*spectrum_to_density, color='C1', label=r'$\tilde{F}(\omega)$')
            ax2.legend()
            handles = []
            labels = []
            for axis in [ax, ax2]:
                h, l = axis.get_legend_handles_labels()
                axis.get_legend().remove()
                handles.extend(h)
                labels.extend(l)
            ax.legend(handles, labels)
            ax.set_ylabel(r'$z$ ASD [m/$\sqrt{\mathrm{Hz}}$]')
            ax2.set_ylabel(r'Force ASD [N/$\sqrt{\mathrm{Hz}}$]')
            ax.set_xlim([20, 100])
            ax.set_ylim([1e-14, 1e-9])
            ax2.set_ylim([1e-20, 1e-15])
            ax.grid(which='both')
            ax.set_xlabel('Frequency [kHz]')
            if self.pulse_amp_keV:
                ax.set_title('{:.0f} keV/c impulse, file {}, impulse {}'.format(self.pulse_amp_keV, file_num + 1, impulse_num + 1))
            else:
                ax.set_title('File {}, impulse {}'.format(file_num + 1, impulse_num + 1))
            freq_domain_pdf.savefig(fig, dpi=150)
            fig.clf()
            plt.close()
            del fig, ax
            gc.collect()

        return f_filt, imp_ind, times_win[0]
    
class NanoDataset:
    """Class to handle a dataset containing multiple files of nanosphere data.
    """

    def __init__(self, path, plot_path=None, f_cutoff=[2e4, 1e5], t_window=1e-3, \
                 search_window=5e-5, fit_window=1e-1, calibrate=False, max_files=1000, \
                 max_windows=1000000, cal_factors=[1, 0], verbose=False, apply_notch=False, config=None):
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
        :param apply_notch: whether to build and apply notch filters, defaults to False
        :type apply_notch: bool, optional
        :param config: configuration dictionary or path to YAML file to pass to NanoFile objects, defaults to None
        :type config: dict or str or Path, optional
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
        self.fit_window = fit_window
        self.calibrate = calibrate
        self.cal_factors = cal_factors
        self.verbose = verbose
        self.apply_notch = apply_notch
        self.config = config
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
            self.noise_spectra_pdf = PdfPages(self.plot_path + '_noise_spectra.pdf')
            self.impulse_win_pdf = PdfPages(self.plot_path + '_impulse_window.pdf')
            self.spectra_pdf = PdfPages(self.plot_path + '_spectra.pdf')
            self.impulse_times_pdf = PdfPages(self.plot_path + '_impulse_times.pdf')
        else:
            self.time_domain_pdf, self.freq_domain_pdf, self.optimal_filter_pdf, self.res_fit_pdf, \
            self.noise_spectra_pdf, self.impulse_win_pdf, self.spectra_pdf, self.impulse_times_pdf \
                = None, None, None, None, None, None, None, None

    def close_pdfs(self):
        """Close the PDF files.
        """
        if self.plot_path:
            self.time_domain_pdf.close()
            self.freq_domain_pdf.close()
            self.optimal_filter_pdf.close()
            self.res_fit_pdf.close()
            self.noise_spectra_pdf.close()
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
                          fit_window=self.fit_window, calibrate=self.calibrate, verbose=self.verbose, \
                          apply_notch=self.apply_notch, config=self.config)
            nf.calibrate_pulse_amp(pulse_amps_1e, pulse_amps_V)
            nf.compute_and_fit_psd(file_num=i, pdf=self.spectra_pdf)
            impulse_inds = nf.get_impulse_inds(file_num=i, pdf=self.impulse_times_pdf)
            if noise:
                impulse_inds = nf.get_noise_inds(impulse_inds)
            nf.get_impulse_array(impulse_inds=impulse_inds, global_params=global_params, file_num=i, \
                                 pdfs=(self.time_domain_pdf, self.freq_domain_pdf, self.optimal_filter_pdf, \
                                 self.res_fit_pdf, self.noise_spectra_pdf, self.impulse_win_pdf))
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
                self.freqs = nf.freqs.copy()

            del nf
            gc.collect()

        self.impulses = np.concatenate(impulses)
        self.resonance_params = np.concatenate(res_params)
        self.pulses = np.concatenate(pulses)
        self.recon_impulse_inds = np.concatenate(recon_impulse_inds)
        self.pulse_times = np.concatenate(pulse_times)
        self.timestamps = np.array(timestamps)
        self.close_pdfs()

    def load_search_data(self):
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
            nf = NanoFile(fp, f_cutoff=self.f_cutoff, search_window=self.search_window, \
                          fit_window=self.fit_window, calibrate=self.calibrate, verbose=self.verbose, \
                          apply_notch=self.apply_notch, cal_factors=self.cal_factors, config=self.config)
            nf.compute_and_fit_psd(file_num=i, pdf=self.spectra_pdf)
            nf.search_all_data(file_num=i, max_windows=self.max_windows, freq_domain_pdf=self.freq_domain_pdf, \
                               time_domain_pdf=self.time_domain_pdf)
            impulses.append(nf.impulses.copy())
            pulses.append(nf.deconvolved_pulses.copy())
            recon_impulse_inds.append(nf.recon_impulse_inds)

            if i == 0:
                self.freqs = nf.freqs.copy()

            del nf
            gc.collect()

        self.impulses = np.concatenate(impulses)
        self.pulses = np.concatenate(pulses)
        self.recon_impulse_inds = np.concatenate(recon_impulse_inds)
        self.close_pdfs()