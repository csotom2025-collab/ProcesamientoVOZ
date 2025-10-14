import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np

class SegmentAnalysisTab(ttk.Frame):
    def pitch_cepstrum(self, signal, fs):
        spectrum = np.fft.fft(signal)
        log_spectrum = np.log(np.abs(spectrum) + 1e-10)
        cepstrum = np.fft.ifft(log_spectrum).real
        min_pitch = int(fs / 500)
        max_pitch = int(fs / 50)
        if max_pitch > len(cepstrum):
            max_pitch = len(cepstrum) - 1
        if min_pitch >= max_pitch:
            return 0
        peak_index = np.argmax(cepstrum[min_pitch:max_pitch]) + min_pitch
        pitch_freq = fs / peak_index if peak_index > 0 else 0
        return pitch_freq

    def calcular_espectrograma(self, data, fs, win_len=0.04, hop_len=0.01, nfft=1024):
        win_samples = int(win_len * fs)
        hop_samples = int(hop_len * fs)
        n_frames = int(np.floor((len(data) - win_samples) / hop_samples)) + 1
        S = []
        t_spec = []
        for i in range(n_frames):
            start = i * hop_samples
            end = start + win_samples
            frame = data[start:end] * np.hamming(win_samples)
            spectrum = np.abs(np.fft.rfft(frame, nfft))
            S.append(spectrum)
            t_spec.append((start + end) / 2 / fs)
        S = np.array(S).T
        f_spec = np.fft.rfftfreq(nfft, 1/fs)
        t_spec = np.array(t_spec)
        return S, t_spec, f_spec

    def calcular_mfcc(self, data, fs, n_mfcc=13, win_len=0.025, hop_len=0.01):
        from scipy.fftpack import dct
        emphasized = np.append(data[0], data[1:] - 0.97 * data[:-1])
        frame_len = int(win_len * fs)
        frame_step = int(hop_len * fs)
        signal_length = len(emphasized)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_len)) / frame_step))
        pad_signal_length = num_frames * frame_step + frame_len
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized, z)
        indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_len)
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
        nfilt = 26
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (fs / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        bin = np.floor((NFFT + 1) * hz_points / fs).astype(int)
        fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
        for m in range(1, nfilt + 1):
            f_m_minus = bin[m - 1]
            f_m = bin[m]
            f_m_plus = bin[m + 1]
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)
        mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
        t_mfcc = np.arange(mfccs.shape[0]) * hop_len
        return mfccs, t_mfcc
    def __init__(self, master, get_signal_callback):
        super().__init__(master)
        # Frame horizontal para nombre y controles
        self.top_controls_frame = ttk.Frame(self)
        self.filename_label = ttk.Label(self.top_controls_frame, text="Archivo: ---")
        self.filename_label.pack(side=tk.LEFT, padx=10)
        ttk.Label(self.top_controls_frame, text="Inicio (s):").pack(side=tk.LEFT, padx=5)
        self.segment_start_var = tk.StringVar(value="0")
        self.segment_start_entry = ttk.Entry(self.top_controls_frame, textvariable=self.segment_start_var, width=8)
        self.segment_start_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(self.top_controls_frame, text="Duración (s):").pack(side=tk.LEFT, padx=5)
        self.segment_dur_var = tk.StringVar(value="0.04")
        self.segment_dur_entry = ttk.Entry(self.top_controls_frame, textvariable=self.segment_dur_var, width=8)
        self.segment_dur_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(self.top_controls_frame, text="Vista:").pack(side=tk.LEFT, padx=10)
        self.current_view = tk.StringVar(value="Frecuencia")
        self.view_menu = ttk.OptionMenu(
            self.top_controls_frame, self.current_view,
            "Frecuencia", "Frecuencia", "Espectrograma", "MFCC", "Pitch (Cepstrum)", "Forma de onda", "Energía", "Cruces por cero"
        )
        self.view_menu.pack(side=tk.LEFT, padx=5)
        self.segment_plot_btn = ttk.Button(self.top_controls_frame, text="Graficar", command=self.plot_segment_spectrum)
        self.segment_plot_btn.pack(side=tk.LEFT, padx=10)
        self.top_controls_frame.pack(fill='x', padx=10, pady=10)

        self.segment_fig = plt.Figure(figsize=(5,3), dpi=100)
        self.segment_ax = self.segment_fig.add_subplot(111)
        self.segment_canvas = FigureCanvasTkAgg(self.segment_fig, master=self)
        self.segment_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.get_signal_callback = get_signal_callback

    def plot_segment_spectrum(self):
        signal_data = self.get_signal_callback()
        if signal_data is None:
            self.segment_ax.clear()
            self.segment_ax.text(0.5, 0.5, "No hay señal cargada", ha='center')
            self.segment_canvas.draw()
            self.filename_label.config(text="Archivo: ---")
            return
        data, fs, filename = signal_data
        self.filename_label.config(text=f"Archivo: {filename}")
        try:
            start = float(self.segment_start_var.get())
            dur = float(self.segment_dur_var.get())
        except ValueError:
            self.segment_ax.clear()
            self.segment_ax.text(0.5, 0.5, "Valores inválidos", ha='center')
            self.segment_canvas.draw()
            return
        start_idx = int(start * fs)
        end_idx = int((start + dur) * fs)
        if start_idx < 0 or end_idx > len(data) or end_idx <= start_idx:
            self.segment_ax.clear()
            self.segment_ax.text(0.5, 0.5, "Rango fuera de la señal", ha='center')
            self.segment_canvas.draw()
            return
        segment = data[start_idx:end_idx]
        view = self.current_view.get()
        self.segment_ax.clear()
        if view == "Frecuencia":
            N = len(segment)
            freqs = np.fft.rfftfreq(N, 1/fs)
            fft_vals = np.abs(np.fft.rfft(segment))
            self.segment_ax.plot(freqs, fft_vals)
            self.segment_ax.set_xlabel("Frecuencia [Hz]")
            self.segment_ax.set_ylabel("Magnitud")
            self.segment_ax.set_title(f"Espectro: inicio={start}s, dur={dur}s")
        elif view == "Espectrograma":
            S, t_spec, f_spec = self.calcular_espectrograma(segment, fs)
            im = self.segment_ax.pcolormesh(t_spec, f_spec, 10 * np.log10(S + 1e-10), shading='auto', cmap="viridis")
            self.segment_ax.set_xlabel("Tiempo [s]")
            self.segment_ax.set_ylabel("Frecuencia [Hz]")
            self.segment_ax.set_title(f"Espectrograma segmento: inicio={start}s, dur={dur}s")
        elif view == "MFCC":
            mfccs, t_mfcc = self.calcular_mfcc(segment, fs)
            im = self.segment_ax.imshow(mfccs.T, aspect='auto', origin='lower', cmap="viridis", extent=[t_mfcc[0], t_mfcc[-1], 0, mfccs.shape[1]])
            self.segment_ax.set_xlabel("Tiempo [s]")
            self.segment_ax.set_ylabel("Coeficiente MFCC")
            self.segment_ax.set_title(f"MFCC segmento: inicio={start}s, dur={dur}s")
        elif view == "Pitch (Cepstrum)":
            win_len = int(0.04 * fs)
            hop_len = int(0.01 * fs)
            pitches = []
            times = []
            for i in range(0, len(segment) - win_len, hop_len):
                seg = segment[i:i+win_len]
                pitch = self.pitch_cepstrum(seg, fs)
                pitches.append(pitch)
                times.append((start * fs + i) / fs)
            self.segment_ax.plot(times, pitches, label="Pitch (Hz)")
            self.segment_ax.set_xlabel("Tiempo [s]")
            self.segment_ax.set_ylabel("Pitch [Hz]")
            self.segment_ax.set_title(f"Pitch segmento: inicio={start}s, dur={dur}s")
            self.segment_ax.legend()
        elif view == "Forma de onda":
            t = np.arange(len(segment)) / fs + start
            self.segment_ax.plot(t, segment)
            self.segment_ax.set_xlabel("Tiempo [s]")
            self.segment_ax.set_ylabel("Amplitud")
            self.segment_ax.set_title(f"Forma de onda segmento: inicio={start}s, dur={dur}s")
        elif view == "Energía":
            energy = np.sum(segment ** 2)
            self.segment_ax.bar([0], [energy])
            self.segment_ax.set_ylabel("Energía")
            self.segment_ax.set_title(f"Energía segmento: inicio={start}s, dur={dur}s")
        elif view == "Cruces por cero":
            zero_crossings = np.where(np.diff(np.sign(segment)))[0]
            t = np.arange(len(segment)) / fs + start
            self.segment_ax.plot(t, segment)
            self.segment_ax.plot(t[zero_crossings], segment[zero_crossings], 'ro')
            self.segment_ax.set_xlabel("Tiempo [s]")
            self.segment_ax.set_ylabel("Amplitud")
            self.segment_ax.set_title(f"Cruces por cero segmento: inicio={start}s, dur={dur}s")
        self.segment_canvas.draw()
