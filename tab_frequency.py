import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import os

class FrequencyAnalysisTab(ttk.Frame):
    def __init__(self, master, folder_var):
        super().__init__(master)
        self.pack(fill='both', expand=True)
        self.folder_var = folder_var
        top_frame = ttk.Frame(self)
        top_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(top_frame, text="Archivo WAV:").pack(side=tk.LEFT, padx=10)
        self.analysis_file_var = tk.StringVar()
        self.analysis_file_menu = ttk.OptionMenu(top_frame, self.analysis_file_var, "", *[], command=self.update_analysis_file)
        self.analysis_file_menu.pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, text="Vista:").pack(side=tk.LEFT, padx=10)
        self.current_view = tk.StringVar(value="Frecuencia")
        self.view_menu = ttk.OptionMenu(top_frame, self.current_view, "Frecuencia", "Frecuencia", "Pitch (Cepstrum)", "Espectrograma", "MFCC", command=self.update_view)
        self.view_menu.pack(side=tk.LEFT, padx=5)
        self.analysis_btn = ttk.Button(top_frame, text="Graficar", command=self.apply_analysis)
        self.analysis_btn.pack(side=tk.LEFT, padx=10)
        self.spectrogram_cmap = tk.StringVar(value="viridis")
        self.cmap_menu = ttk.OptionMenu(top_frame, self.spectrogram_cmap, "viridis", "viridis", "jet", "gray", "Greys")
        self.cmap_menu.pack_forget()  # Oculto por defecto
        # Paneles de control debajo de la gráfica
        self.move_frame = ttk.Frame(self)
        self.interval_frame = ttk.Frame(self)
        self.options_frame = ttk.Frame(self)
        self.analysis_result_label = ttk.Label(self, text="")
        # Inicializar gráfica principal
        self.figure = plt.Figure(figsize=(6,3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        self.move_frame.pack(fill='x', padx=10, pady=5)
        self.interval_frame.pack(fill='x', padx=10, pady=5)
        self.options_frame.pack(fill='x', padx=10, pady=5)
        self.analysis_result_label.pack(fill='x', padx=10, pady=5)
        # Paneles de control
        self.left_btn = ttk.Button(self.move_frame, text="<", width=3)
        self.left_btn.pack(side=tk.LEFT, padx=2)
        self.left_btn.bind('<ButtonPress-1>', lambda e: self._start_repeat(self.move_view, -1))
        self.left_btn.bind('<ButtonRelease-1>', lambda e: self._stop_repeat())
        self.right_btn = ttk.Button(self.move_frame, text=">", width=3)
        self.right_btn.pack(side=tk.LEFT, padx=2)
        self.right_btn.bind('<ButtonPress-1>', lambda e: self._start_repeat(self.move_view, 1))
        self.right_btn.bind('<ButtonRelease-1>', lambda e: self._stop_repeat())
        self.zoom_in_btn = ttk.Button(self.move_frame, text="+", width=3)
        self.zoom_in_btn.pack(side=tk.LEFT, padx=2)
        self.zoom_in_btn.bind('<ButtonPress-1>', lambda e: self._start_repeat(self.zoom_in))
        self.zoom_in_btn.bind('<ButtonRelease-1>', lambda e: self._stop_repeat())
        self.zoom_out_btn = ttk.Button(self.move_frame, text="-", width=3)
        self.zoom_out_btn.pack(side=tk.LEFT, padx=2)
        self.zoom_out_btn.bind('<ButtonPress-1>', lambda e: self._start_repeat(self.zoom_out))
        self.zoom_out_btn.bind('<ButtonRelease-1>', lambda e: self._stop_repeat())
        self.label_inicio = ttk.Label(self.interval_frame, text="Inicio (Hz):")
        self.label_inicio.pack(side=tk.LEFT)
        self.zoom_start_var = tk.StringVar(value="0")
        self.zoom_start_entry = ttk.Entry(self.interval_frame, textvariable=self.zoom_start_var, width=8)
        self.zoom_start_entry.pack(side=tk.LEFT, padx=5)
        self.label_fin = ttk.Label(self.interval_frame, text="Fin (Hz):")
        self.label_fin.pack(side=tk.LEFT)
        self.zoom_end_var = tk.StringVar(value="5000")
        self.zoom_end_entry = ttk.Entry(self.interval_frame, textvariable=self.zoom_end_var, width=8)
        self.zoom_end_entry.pack(side=tk.LEFT, padx=5)
        self.apply_zoom_btn = ttk.Button(self.interval_frame, text="Aplicar zoom", command=self.apply_manual_zoom)
        self.apply_zoom_btn.pack(side=tk.LEFT, padx=10)
        self.reset_zoom_btn = ttk.Button(self.options_frame, text="Reiniciar zoom", command=self.reset_zoom)
        self.reset_zoom_btn.pack(side=tk.LEFT, padx=10)
    def show_segment_spectrum_dialog(self):
        # Ventana para pedir inicio y duración
        dialog = tk.Toplevel(self)
        dialog.title("Espectro de segmento")
        ttk.Label(dialog, text="Inicio (s):").pack(padx=10, pady=5)
        start_var = tk.StringVar(value="0")
        start_entry = ttk.Entry(dialog, textvariable=start_var, width=8)
        start_entry.pack(padx=10, pady=5)
        ttk.Label(dialog, text="Duración (s):").pack(padx=10, pady=5)
        dur_var = tk.StringVar(value="0.04")
        dur_entry = ttk.Entry(dialog, textvariable=dur_var, width=8)
        dur_entry.pack(padx=10, pady=5)
        def plot_segment():
            try:
                start = float(start_var.get())
                dur = float(dur_var.get())
            except ValueError:
                ttk.Label(dialog, text="Valores inválidos").pack()
                return
            if self.current_data is None or self.current_fs is None:
                ttk.Label(dialog, text="No hay señal cargada").pack()
                return
            fs = self.current_fs
            data = self.current_data
            start_idx = int(start * fs)
            end_idx = int((start + dur) * fs)
            if start_idx < 0 or end_idx > len(data) or end_idx <= start_idx:
                ttk.Label(dialog, text="Rango fuera de la señal").pack()
                return
            segment = data[start_idx:end_idx]
            # Calcular espectro
            N = len(segment)
            freqs = np.fft.rfftfreq(N, 1/fs)
            fft_vals = np.abs(np.fft.rfft(segment))
            # Ventana nueva para mostrar
            spec_win = tk.Toplevel(self)
            spec_win.title("Espectro de segmento")
            fig = plt.Figure(figsize=(5,3), dpi=100)
            ax = fig.add_subplot(111)
            ax.plot(freqs, fft_vals)
            ax.set_xlabel("Frecuencia [Hz]")
            ax.set_ylabel("Magnitud")
            ax.set_title(f"Espectro: inicio={start}s, dur={dur}s")
            canvas = FigureCanvasTkAgg(fig, master=spec_win)
            canvas.get_tk_widget().pack(fill='both', expand=True)
            canvas.draw()
        ttk.Button(dialog, text="Graficar", command=plot_segment).pack(padx=10, pady=10)
        # Empaquetar los paneles de control después de la gráfica
        self.figure = plt.Figure(figsize=(6,3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        # Empaquetar la gráfica primero, luego los controles
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        self.move_frame.pack(fill='x', padx=10, pady=5)
        self.interval_frame.pack(fill='x', padx=10, pady=5)
        self.options_frame.pack(fill='x', padx=10, pady=5)
        self.analysis_result_label.pack(fill='x', padx=10, pady=5)
        self.refresh_analysis_file_list()
        self.current_data = None
        self.current_fs = None
        self._repeat_job = None
        self._repeat_func = None
        self._repeat_args = None

    def update_view(self, *args):
        view = self.current_view.get()
        if view in ["Espectrograma", "MFCC", "Pitch (Cepstrum)"]:
            self.cmap_menu.pack(side=tk.LEFT, padx=5) if view in ["Espectrograma", "MFCC"] else self.cmap_menu.pack_forget()
            self.label_inicio.config(text="Inicio (s):")
            self.label_fin.config(text="Fin (s):")
        else:
            self.cmap_menu.pack_forget()
            self.label_inicio.config(text="Inicio (Hz):")
            self.label_fin.config(text="Fin (Hz):")
        self.apply_analysis()

    def apply_manual_zoom(self):
        try:
            start = float(self.zoom_start_var.get())
            end = float(self.zoom_end_var.get())
        except ValueError:
            self.analysis_result_label.config(text="Intervalo inválido.")
            return
        view = self.current_view.get()
        xlim = self.ax.get_xlim()
        if start < 0 or end <= start:
            self.analysis_result_label.config(text="Intervalo fuera de rango.")
            return
        # Solo para vista de frecuencia, el zoom es en Hz
        if view == "Frecuencia":
            self.ax.set_xlim(start, end)
        else:
            # Para otras vistas, el zoom es en tiempo (segundos)
            self.ax.set_xlim(start, end)
        self.canvas.draw()

    def reset_zoom(self):
        xlim = self.ax.get_xlim()
        self.ax.set_xlim(0, xlim[1])
        self.canvas.draw()

    def move_view(self, direction):
        xlim = self.ax.get_xlim()
        window = xlim[1] - xlim[0]
        shift = window * 0.1 * direction
        new_start = max(0, xlim[0] + shift)
        new_end = xlim[1] + shift
        if new_end - new_start < window:
            new_start = new_end - window
            if new_start < 0:
                new_start = 0
                new_end = window
        self.ax.set_xlim(new_start, new_end)
        self.canvas.draw()

    def zoom_in(self):
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        width = (xlim[1] - xlim[0]) * 0.9
        new_start = max(0, center - width/2)
        new_end = center + width/2
        self.ax.set_xlim(new_start, new_end)
        self.canvas.draw()

    def zoom_out(self):
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        width = (xlim[1] - xlim[0]) * 1.1
        total = xlim[1]
        new_start = max(0, center - width/2)
        new_end = min(total, center + width/2)
        self.ax.set_xlim(new_start, new_end)
        self.canvas.draw()

    def _start_repeat(self, func, *args):
        self._repeat_func = func
        self._repeat_args = args
        self._repeat_job = self.after(100, self._repeat_action)

    def _repeat_action(self):
        if self._repeat_func:
            self._repeat_func(*self._repeat_args)
            self._repeat_job = self.after(100, self._repeat_action)

    def _stop_repeat(self):
        if self._repeat_job:
            self.after_cancel(self._repeat_job)
            self._repeat_job = None
            self._repeat_func = None
            self._repeat_args = None

    def refresh_analysis_file_list(self):
        folder = self.folder_var.get()
        menu = self.analysis_file_menu['menu']
        menu.delete(0, 'end')
        if not folder or not os.path.isdir(folder):
            self.analysis_file_var.set("")
            return
        wav_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        for f in wav_files:
            menu.add_command(label=f, command=lambda value=f: self.analysis_file_var.set(value))
        if wav_files:
            self.analysis_file_var.set(wav_files[0])
        else:
            self.analysis_file_var.set("")

    def update_analysis_file(self, value):
        self.analysis_file_var.set(value)

    def apply_analysis(self):
        folder = self.folder_var.get()
        filename = self.analysis_file_var.get()
        if not filename:
            self.analysis_result_label.config(text="Selecciona un archivo WAV.")
            return
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            self.analysis_result_label.config(text="El archivo no existe.")
            return
        fs, data = wav.read(path)
        if data.ndim > 1:
            data = data[:,0]
        data = data.astype(float)
        self.current_data = data
        self.current_fs = fs
        view = self.current_view.get()
        # Eliminar colorbar anterior si existe y es válida
        if hasattr(self, 'colorbar') and self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None
        # Para espectrograma y MFCC: borrar todos los ejes y crear uno nuevo, pero NO volver a empaquetar el canvas
        if view == "Espectrograma":
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            S, t_spec, f_spec = self.calcular_espectrograma(data, fs)
            cmap = self.spectrogram_cmap.get()
            im = self.ax.pcolormesh(t_spec, f_spec, 10 * np.log10(S + 1e-10), shading='auto', cmap=cmap)
            self.ax.set_xlabel("Tiempo [s]")
            self.ax.set_ylabel("Frecuencia [Hz]")
            self.ax.set_title(f"Espectrograma: {filename}")
            self.colorbar = self.figure.colorbar(im, ax=self.ax, label="dB", orientation='vertical')
            self.figure.subplots_adjust(right=0.85)
            self.analysis_result_label.config(text="")
        elif view == "MFCC":
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            mfccs, t_mfcc = self.calcular_mfcc(data, fs)
            im = self.ax.imshow(mfccs.T, aspect='auto', origin='lower', cmap=self.spectrogram_cmap.get(), extent=[t_mfcc[0], t_mfcc[-1], 0, mfccs.shape[1]])
            self.ax.set_xlabel("Tiempo [s]")
            self.ax.set_ylabel("Coeficiente MFCC")
            self.ax.set_title(f"MFCC: {filename}")
            self.colorbar = self.figure.colorbar(im, ax=self.ax, label="Valor", orientation='vertical')
            self.figure.subplots_adjust(right=0.85)
            self.analysis_result_label.config(text="")
        else:
            self.ax.clear()
            if view == "Frecuencia":
                N = len(data)
                freqs = np.fft.rfftfreq(N, 1/fs)
                fft_vals = np.abs(np.fft.rfft(data))
                self.ax.plot(freqs, fft_vals)
                self.ax.set_xlabel("Frecuencia [Hz]")
                self.ax.set_ylabel("Magnitud")
                self.ax.set_title(f"Espectro de Frecuencia: {filename}")
                self.analysis_result_label.config(text="")
            elif view == "Pitch (Cepstrum)":
                # Parámetros de ventana
                win_len = int(0.04 * fs)  # 40 ms
                hop_len = int(0.01 * fs)  # 10 ms
                pitches = []
                times = []
                for start in range(0, len(data) - win_len, hop_len):
                    segment = data[start:start+win_len]
                    pitch = self.pitch_cepstrum(segment, fs)
                    pitches.append(pitch)
                    times.append(start/fs)
                self.ax.plot(times, pitches, label="Pitch (Hz)")
                self.ax.set_xlabel("Tiempo [s]")
                self.ax.set_ylabel("Pitch [Hz]")
                self.ax.set_title(f"Pitch estimado (Cepstrum): {filename}")
                self.ax.legend()
                if pitches:
                    avg_pitch = np.mean([p for p in pitches if 50 < p < 500])
                    self.analysis_result_label.config(text=f"Pitch promedio: {avg_pitch:.2f} Hz")
                else:
                    self.analysis_result_label.config(text="No se pudo estimar el pitch.")
        self.canvas.draw()

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
        """
        Calcula el espectrograma de una señal de audio sin usar scipy.signal.spectrogram.
        Args:
            data: señal de audio (numpy array)
            fs: frecuencia de muestreo
            win_len: longitud de ventana en segundos
            hop_len: salto entre ventanas en segundos
            nfft: tamaño de FFT
        Returns:
            S: matriz de espectrograma (frecuencia x tiempo)
            t_spec: vector de tiempo
            f_spec: vector de frecuencia
        """
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
        S = np.array(S).T  # (frecuencia x tiempo)
        f_spec = np.fft.rfftfreq(nfft, 1/fs)
        t_spec = np.array(t_spec)
        return S, t_spec, f_spec

    def calcular_mfcc(self, data, fs, n_mfcc=13, win_len=0.025, hop_len=0.01):
        """
        Calcula los coeficientes MFCC de una señal de audio sin librerías externas.
        Args:
            data: señal de audio (numpy array)
            fs: frecuencia de muestreo
            n_mfcc: número de coeficientes MFCC
            win_len: longitud de ventana en segundos
            hop_len: salto entre ventanas en segundos
        Returns:
            mfccs: matriz de coeficientes MFCC (ventanas x coeficientes)
            t_mfcc: vector de tiempo de cada ventana
        """
        # 1. Pre-emphasis
        emphasized = np.append(data[0], data[1:] - 0.97 * data[:-1])
        # 2. framentación
        frame_len = int(win_len * fs)
        frame_step = int(hop_len * fs)
        signal_length = len(emphasized)
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_len)) / frame_step))
        pad_signal_length = num_frames * frame_step + frame_len
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(emphasized, z)
        indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        # 3. Ventanamiento
        frames *= np.hamming(frame_len)
        # 4. FFT y Power Spectrum
        NFFT = 512
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
        # 5. Filtro de bancos Mel
        nfilt = 28
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (fs / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        bin = np.floor((NFFT + 1) * hz_points / fs).astype(int)
        fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
        #Ventanas triangulares
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
        # 6. Coficientes para DCT para obtener MFCC
        mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
        # 7. Tiempo de cada ventana
        t_mfcc = np.arange(mfccs.shape[0]) * hop_len
        return mfccs, t_mfcc