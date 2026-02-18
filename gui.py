#!/usr/bin/env python3
"""Nanosphere Live Data GUI

Watches a folder for new HDF5 nanosphere data files, loads the most recent
one using NanoFile, and displays calibration readouts, an ASD spectra plot,
and a raw time-series trace — updating automatically at ~1 Hz.

Usage:
    python gui.py

Requires PySide6 (pip install PySide6).
"""

import os
import sys
import glob
import time
import traceback
import yaml

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.style as mplstyle
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox,
    QFileDialog, QGroupBox, QFrame, QScrollArea, QSplitter, QTabWidget,
)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QColor

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_processing import NanoFile
from plotting import plot_raw_asd, plot_force_asd, plot_position_asd

try:
    mplstyle.use('clarke-default')
except OSError:
    pass


# ---------------------------------------------------------------------------
# Background file-watcher thread
# ---------------------------------------------------------------------------

class FileWatcher(QThread):
    """Polls a folder for new .hdf5 files and emits new_file(path)."""

    new_file = Signal(str)

    def __init__(self, folder: str):
        super().__init__()
        self.folder = folder
        self.last_file: str | None = None
        self._running = True

    def run(self):
        while self._running:
            try:
                files = sorted(
                    glob.glob(os.path.join(self.folder, '*.hdf5')),
                    key=os.path.getmtime,
                )
                if files and files[-1] != self.last_file:
                    self.last_file = files[-1]
                    self.new_file.emit(files[-1])
            except Exception:
                pass
            time.sleep(1.0)

    def stop(self):
        self._running = False
        self.wait()


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class NanosphereGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nanosphere Live Data")
        self.resize(1100, 760)

        self._watcher: FileWatcher | None = None

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(6, 6, 6, 6)

        # Vertical splitter: top section (config + plots) / bottom (time trace)
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setChildrenCollapsible(False)

        # ---- Top section ----
        top_widget = QWidget()
        # Allow top section to shrink below its children's preferred size so
        # the splitter can honour the 60/40 ratio at any window height.
        top_widget.setMinimumHeight(0)
        top = QHBoxLayout(top_widget)
        top.setSpacing(6)
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._build_left_panel(), stretch=0)
        top.addWidget(self._build_spectra_panel(), stretch=1)

        # ---- Bottom section ----
        trace_panel = self._build_trace_panel()

        self._splitter.addWidget(top_widget)
        self._splitter.addWidget(trace_panel)
        root_layout.addWidget(self._splitter)

        # Apply initial 60/40 split after the layout has been computed
        QTimer.singleShot(0, self._enforce_split)

    # -----------------------------------------------------------------------
    # Panel builders
    # -----------------------------------------------------------------------

    def _build_left_panel(self) -> QWidget:
        """Config fields, start/stop button, and calibration readouts.

        Wrapped in a QScrollArea so its content height does not impose a
        minimum height on the top section of the splitter.
        """
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(6)

        # ---- Folder ----
        folder_group = QGroupBox("Folder")
        fg = QHBoxLayout(folder_group)
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("Select folder…")
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_folder)
        fg.addWidget(self.folder_edit)
        fg.addWidget(browse_btn)
        layout.addWidget(folder_group)

        # ---- Config ----
        cfg_group = QGroupBox("Configuration")
        cg = QGridLayout(cfg_group)
        cg.setVerticalSpacing(3)

        self._cfg: dict[str, QLineEdit] = {}
        fields = [
            ("f_cutoff low (Hz)",   "f_low",           "20000"),
            ("f_cutoff high (Hz)",  "f_high",          "100000"),
            ("search_window (s)",   "search_window",   "5e-5"),
            ("fit_window (s)",      "fit_window",      "0.1"),
            ("d_sphere_nm (nm)",    "d_sphere_nm",     "166.0"),
            ("meters_per_volt (m/V)", "meters_per_volt", "5e-8"),
            ("keV_per_N",           "keV_per_N",       "1.0"),
            ("ds_factor (int)",     "ds_factor",       "1"),
        ]
        for r, (label, key, default) in enumerate(fields):
            cg.addWidget(QLabel(label), r, 0)
            edit = QLineEdit(default)
            edit.setFixedWidth(90)
            cg.addWidget(edit, r, 1)
            self._cfg[key] = edit

        # Update meters_per_volt default when sphere diameter is changed
        self._cfg['d_sphere_nm'].editingFinished.connect(self._auto_update_mpv)

        r = len(fields)
        self.calibrate_cb = QCheckBox("calibrate")
        self.notch_cb = QCheckBox("apply_notch")
        cg.addWidget(self.calibrate_cb, r, 0)
        cg.addWidget(self.notch_cb,     r, 1)
        r += 1

        # Config-file override row
        cg.addWidget(QLabel("Config file:"), r, 0)
        self.cfg_file_edit = QLineEdit()
        self.cfg_file_edit.setPlaceholderText("optional .yaml…")
        cg.addWidget(self.cfg_file_edit, r, 1)
        r += 1
        cfg_browse_btn = QPushButton("Browse config…")
        cfg_browse_btn.clicked.connect(self._browse_config_file)
        cfg_load_btn   = QPushButton("Load config")
        cfg_load_btn.clicked.connect(self._load_config_file)
        cg.addWidget(cfg_browse_btn, r, 0)
        cg.addWidget(cfg_load_btn,   r, 1)

        layout.addWidget(cfg_group)

        # ---- Start/Stop button ----
        self.watch_btn = QPushButton("Start Watching")
        self.watch_btn.setCheckable(True)
        self.watch_btn.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; font-weight:bold; padding:4px; }"
            "QPushButton:checked { background:#c62828; }"
        )
        self.watch_btn.clicked.connect(self._toggle_watch)
        layout.addWidget(self.watch_btn)

        # ---- Readouts ----
        ro_group = QGroupBox("Calibration Readouts")
        rg = QGridLayout(ro_group)
        rg.setVerticalSpacing(2)

        self._readouts: dict[str, QLabel] = {}
        ro_fields = [
            ("ω₀/2π (kHz)",      "omega_0"),
            ("γ/2π (Hz)",         "gamma"),
            ("Mass (kg)",         "mass"),
            ("Cal. factor (m/V)", "meters_per_volt"),
            ("Drive freq (kHz)",  "drive_freq"),
            ("Drive amp (Vpp)",   "drive_amp"),
            ("N charges",         "n_charges"),
            ("Phonon occ. n̄",    "phonon_n"),
            ("T_eff (K)",         "T_eff"),
            ("Timestamp",         "timestamp"),
            ("Pressure (mbar)",   "pressure"),
        ]
        for r, (label, key) in enumerate(ro_fields):
            rg.addWidget(QLabel(label + ":"), r, 0, Qt.AlignLeft)
            val_lbl = QLabel("—")
            val_lbl.setFrameShape(QFrame.Shape.StyledPanel)
            val_lbl.setMinimumWidth(100)
            val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val_lbl.setStyleSheet("background:white; padding:1px 3px;")
            rg.addWidget(val_lbl, r, 1)
            self._readouts[key] = val_lbl

        layout.addWidget(ro_group)

        # ---- Status ----
        status_group = QGroupBox("Status")
        sg = QVBoxLayout(status_group)
        self.status_lbl = QLabel("Idle")
        self.status_lbl.setWordWrap(True)
        self.status_lbl.setStyleSheet("background:#fffde7; padding:3px;")
        sg.addWidget(self.status_lbl)
        layout.addWidget(status_group)

        layout.addStretch()

        # Wrap in a scroll area — this means the left panel's content height
        # does not set a hard minimum on the top section of the splitter.
        scroll = QScrollArea()
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(270)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(0)
        return scroll

    def _build_spectra_panel(self) -> QWidget:
        tabs = QTabWidget()

        # ---- Tab 1: Raw ASD ----
        tab1 = QWidget()
        t1 = QVBoxLayout(tab1)
        t1.setContentsMargins(2, 2, 2, 2)
        self.fig_spectra = Figure(tight_layout=True)
        self.ax_spectra = self.fig_spectra.add_subplot(111)
        self.ax_spectra.set_xlabel("Frequency [kHz]")
        self.ax_spectra.set_ylabel(r"Raw ASD [V/$\sqrt{\mathrm{Hz}}$]")
        self.ax_spectra.set_yscale('log')
        self.ax_spectra.grid(True, which='both', alpha=0.4)
        self.canvas_spectra = FigureCanvas(self.fig_spectra)
        self.canvas_spectra.setMinimumSize(0, 0)
        t1.addWidget(self.canvas_spectra)
        tabs.addTab(tab1, "Raw ASD")

        # ---- Tab 2: Calibrated ASDs ----
        tab2 = QWidget()
        t2 = QVBoxLayout(tab2)
        t2.setContentsMargins(2, 2, 2, 2)
        self.fig_calib = Figure(tight_layout=True)
        self.ax_force, self.ax_pos = self.fig_calib.subplots(1, 2)
        for ax, ylabel in [
            (self.ax_force, r"Calibrated ASD [$\mathrm{N/\sqrt{Hz}}$]"),
            (self.ax_pos,   r"Calibrated ASD [$\mathrm{m/\sqrt{Hz}}$]"),
        ]:
            ax.set_xlabel("Frequency [kHz]")
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.grid(True, which='both', alpha=0.4)
        self.canvas_calib = FigureCanvas(self.fig_calib)
        self.canvas_calib.setMinimumSize(0, 0)
        t2.addWidget(self.canvas_calib)
        tabs.addTab(tab2, "Calibrated ASD")

        return tabs

    def _build_trace_panel(self) -> QWidget:
        box = QGroupBox("Raw Time Trace (z_raw)")
        # Prevent the trace from being squashed to nothing
        box.setMinimumHeight(120)
        layout = QVBoxLayout(box)
        self.fig_trace = Figure(tight_layout=True)
        self.ax_trace = self.fig_trace.add_subplot(111)
        self.ax_trace.set_xlabel("Time (s)")
        self.ax_trace.set_ylabel("Signal (V)")
        self.ax_trace.grid(True, alpha=0.4)
        self.canvas_trace = FigureCanvas(self.fig_trace)
        self.canvas_trace.setMinimumSize(0, 0)
        layout.addWidget(self.canvas_trace)
        return box

    # -----------------------------------------------------------------------
    # Config helpers
    # -----------------------------------------------------------------------

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select data folder")
        if path:
            self.folder_edit.setText(path)

    def _auto_update_mpv(self):
        """Log-linearly interpolate a meters_per_volt default from d_sphere_nm.

        Anchor points: 100 nm → 2e-7 m/V, 166 nm → 5e-8 m/V.
        """
        try:
            d = float(self._cfg['d_sphere_nm'].text())
            log_mpv = np.interp(d, [100.0, 166.0],
                                [np.log(2e-7), np.log(5e-8)])
            self._cfg['meters_per_volt'].setText(f'{np.exp(log_mpv):.3e}')
        except (ValueError, TypeError):
            pass

    def _browse_config_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select config file", "", "YAML files (*.yaml *.yml);;All files (*)")
        if path:
            self.cfg_file_edit.setText(path)

    def _load_config_file(self):
        """Load a NanoFile-compatible YAML config and populate the GUI fields."""
        path = self.cfg_file_edit.text().strip()
        if not path or not os.path.isfile(path):
            self.status_lbl.setText("Config file not found.")
            return
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            self.status_lbl.setText(f"Config load error: {e}")
            return

        # Map YAML keys → GUI fields
        simple = {
            'search_window': 'search_window',
            'fit_window':    'fit_window',
            'd_sphere_nm':   'd_sphere_nm',
            'keV_per_N':     'keV_per_N',
            'ds_factor':     'ds_factor',
            'meters_per_volt': 'meters_per_volt',
        }
        for yaml_key, cfg_key in simple.items():
            if yaml_key in cfg:
                self._cfg[cfg_key].setText(str(cfg[yaml_key]))

        if 'f_cutoff' in cfg:
            self._cfg['f_low'].setText(str(cfg['f_cutoff'][0]))
            self._cfg['f_high'].setText(str(cfg['f_cutoff'][1]))

        if 'calibrate' in cfg:
            self.calibrate_cb.setChecked(bool(cfg['calibrate']))
        if 'apply_notch' in cfg:
            self.notch_cb.setChecked(bool(cfg['apply_notch']))

        # If d_sphere_nm was loaded but meters_per_volt was not, auto-update
        if 'd_sphere_nm' in cfg and 'meters_per_volt' not in cfg:
            self._auto_update_mpv()

        self.status_lbl.setText(f"Config loaded: {os.path.basename(path)}")

    def _get_nanofile_kwargs(self) -> dict:
        return dict(
            f_cutoff=[float(self._cfg['f_low'].text()),
                      float(self._cfg['f_high'].text())],
            search_window=float(self._cfg['search_window'].text()),
            fit_window=float(self._cfg['fit_window'].text()),
            d_sphere_nm=float(self._cfg['d_sphere_nm'].text()),
            calibrate=self.calibrate_cb.isChecked(),
            apply_notch=self.notch_cb.isChecked(),
            keV_per_N=float(self._cfg['keV_per_N'].text()),
            ds_factor=int(self._cfg['ds_factor'].text()),
        )

    # -----------------------------------------------------------------------
    # Splitter sizing
    # -----------------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._enforce_split)

    def _enforce_split(self):
        h = self._splitter.height()
        if h > 0:
            self._splitter.setSizes([int(h * 0.6), int(h * 0.4)])

    # -----------------------------------------------------------------------
    # Watch / stop
    # -----------------------------------------------------------------------

    def _toggle_watch(self):
        if self.watch_btn.isChecked():
            folder = self.folder_edit.text().strip()
            if not folder or not os.path.isdir(folder):
                self.status_lbl.setText("Error: select a valid folder first.")
                self.watch_btn.setChecked(False)
                return
            self.watch_btn.setText("Stop Watching")
            self.status_lbl.setText("Watching for new files…")
            self._watcher = FileWatcher(folder)
            self._watcher.new_file.connect(self._load_and_plot)
            self._watcher.start()
        else:
            self.watch_btn.setText("Start Watching")
            self.status_lbl.setText("Stopped.")
            if self._watcher:
                self._watcher.stop()
                self._watcher = None

    def closeEvent(self, event):
        if self._watcher:
            self._watcher.stop()
        super().closeEvent(event)

    # -----------------------------------------------------------------------
    # Load & plot  (runs in the main thread via Qt signal)
    # -----------------------------------------------------------------------

    def _load_and_plot(self, file_path: str):
        fname = os.path.basename(file_path)
        self.status_lbl.setText(f"Loading: {fname}…")
        QApplication.processEvents()

        try:
            nf = NanoFile(file_path, **self._get_nanofile_kwargs())

            # Apply manual meters_per_volt (overwritten by compute_and_fit_psd
            # only when calibrate=True, which is the desired behaviour).
            try:
                nf.meters_per_volt = float(self._cfg['meters_per_volt'].text())
            except (ValueError, KeyError):
                pass

            # Parse filename for n_charges / drive params (may fail — that's ok)
            try:
                nf.calibrate_pulse_amp()
            except Exception:
                pass

            nf.compute_and_fit_psd()

            # -----------------------------------------------------------
            # Update readout labels
            # -----------------------------------------------------------
            def _fmt(val, spec='.4g', na='N/A'):
                try:
                    return format(float(val), spec)
                except Exception:
                    return na

            self._readouts['omega_0'].setText(
                _fmt(nf.omega_0 / (2 * np.pi * 1e3)))
            self._readouts['gamma'].setText(
                _fmt(nf.gamma / (2 * np.pi)))
            self._readouts['mass'].setText(
                _fmt(nf.mass_sphere, '.3e'))
            self._readouts['meters_per_volt'].setText(
                _fmt(nf.meters_per_volt, '.3e'))

            drive_freq = getattr(nf, 'drive_freq', None)
            self._readouts['drive_freq'].setText(
                _fmt(drive_freq / 1e3) if drive_freq is not None else 'N/A')

            drive_amp = getattr(nf, 'drive_amp', None)
            self._readouts['drive_amp'].setText(
                _fmt(drive_amp) if drive_amp is not None else 'N/A')

            n_charges = getattr(nf, 'n_charges', None)
            self._readouts['n_charges'].setText(
                _fmt(n_charges, '.1f') if n_charges is not None else 'N/A')

            self._readouts['phonon_n'].setText(_fmt(nf.n_avg, '.3g'))
            self._readouts['T_eff'].setText(_fmt(nf.T_eff, '.4g'))

            ts = getattr(nf, 'timestamp', None)
            self._readouts['timestamp'].setText(str(ts)[:19] if ts else 'N/A')

            pressure = getattr(nf, 'pressure', None)
            self._readouts['pressure'].setText(
                _fmt(pressure, '.3g') if pressure is not None else 'N/A')

            # -----------------------------------------------------------
            # Tab 1: Raw ASD
            # -----------------------------------------------------------
            self.ax_spectra.cla()
            plot_raw_asd(nf, self.ax_spectra)
            self.fig_spectra.tight_layout()
            self.canvas_spectra.draw()

            # -----------------------------------------------------------
            # Tab 2: Calibrated ASDs
            # -----------------------------------------------------------
            try:
                self.ax_force.cla()
                plot_force_asd(nf, self.ax_force)
                self.ax_pos.cla()
                plot_position_asd(nf, self.ax_pos)
                self.fig_calib.tight_layout()
                self.canvas_calib.draw()
            except Exception:
                pass

            # -----------------------------------------------------------
            # Time trace (downsampled to ≤10 k points)
            # -----------------------------------------------------------
            z = nf.z_raw
            t = nf.times
            stride = max(1, len(z) // 10000)
            self.ax_trace.cla()
            self.ax_trace.plot(t[::stride], z[::stride],
                               lw=0.5, alpha=0.85, color='C0')
            self.ax_trace.set_xlabel("Time (s)")
            self.ax_trace.set_ylabel("Signal (V)")
            self.ax_trace.grid(True, alpha=0.4)
            self.fig_trace.tight_layout()
            self.canvas_trace.draw()

            self.status_lbl.setText(f"Loaded: {fname}")

        except Exception as e:
            tb = traceback.format_exc()
            self.status_lbl.setText(
                f"{type(e).__name__}: {e}\n…{tb[-300:]}")


def main():
    app = QApplication(sys.argv)
    win = NanosphereGUI()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
