from functools import partial
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import get_window
from scipy.fftpack import dct
import os
from scipy.signal import resample_poly
from fractions import Fraction


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
        self.save_image_btn = ttk.Button(self.options_frame, text="Guardar imagen", command=self.save_current_view)
        self.save_image_btn.pack(side=tk.LEFT, padx=10)

    
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

    def save_current_view(self):
        """Open a file dialog and save the current figure to the chosen path."""
        # Suggest a default filename using current file and view
        filename = self.analysis_file_var.get() or "figure"
        view = self.current_view.get() if hasattr(self, 'current_view') else 'view'
        default_name = f"{os.path.splitext(filename)[0]}_{view}.png"
        filetypes = [("PNG image","*.png"), ("JPEG image","*.jpg"), ("PDF file","*.pdf"), ("SVG file","*.svg")]
        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=filetypes, initialfile=default_name, title='Guardar imagen como')
        if not path:
            return
        try:
            # Save the current Matplotlib figure
            self.figure.savefig(path, bbox_inches='tight')
            self.analysis_result_label.config(text=f"Imagen guardada: {os.path.basename(path)}")
        except Exception as e:
            self.analysis_result_label.config(text=f"Error al guardar: {e}")

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
        fs1, data = wav.read(path)
        data, fs = self.load_with_scipy(path,sr=fs1)

        
        
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
            #mfccs, t_mfcc = self.calcular_mfcc(data, fs)
            mfcchtk=self.mfcc_htk(data, fs)
            #im = self.ax.imshow(mfccs.T, aspect='auto', origin='lower', cmap=self.spectrogram_cmap.get(), extent=[t_mfcc[0], t_mfcc[-1], 0, mfccs.shape[1]])
            im= self.ax.imshow(mfcchtk.T, aspect='auto', origin='lower', cmap=self.spectrogram_cmap.get(), extent=[0, len(data)/fs, 0, mfcchtk.shape[1]])
            self.ax.set_xlabel("Tiempo [s]")
            self.ax.set_ylabel("Coeficiente MFCC HTK")
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
        print('MFCC shape:', mfccs.shape)
        print('MFCC (frames):\n', mfccs)
        print('Tiempo MFCC (s):\n', t_mfcc)
        return mfccs, t_mfcc
    ##################### MFCC HTK IMPLEMENTATION #####################
    ############################################################################
    def frame_audio(self, audio, FFT_size=2048, hop_size=10, sample_rate=44100):
        # hop_size in ms
        frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num,FFT_size))
        for n in range(frame_num):
            frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
        return frames
    
    def freq_to_mel(self, freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)


    def met_to_freq(self, mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    def Melk(self, k, fres):
        return 1127.0 * np.log(1.0 + (k - 1) * fres)


    def do_melk(self,filter_points, FFT_size, num_chan, mels, sample_rate):
        fres = sample_rate/(FFT_size*700)
        #print (fres)
        chan = 0
        Nby2 = int (FFT_size / 2)
        maxChan = num_chan
        fblowChan = np.zeros((Nby2,), dtype=int)
        
        for k in range (0, Nby2 -1):
            melk = self.Melk (k,fres)
    #        print (melk)
            if (k < 2 or k > Nby2):
                fblowChan [k] = -1
            else:
            # print (melk)
                #print (mels[chan])
                while (mels[chan] < melk and chan <= maxChan):
                    chan+=1
                fblowChan[k] = chan -1


        for k in range (0, Nby2 -2):
            fblowChan [k] = fblowChan [k+1]
        fblowChan[Nby2-1] = num_chan
        
        return fblowChan


    def create_vector_lowerChanWeights(self,filter_points, FFT_size, lowChan, mels, sample_rate):
        Nby2 = int(FFT_size / 2)
        fblowWt = np.zeros(Nby2)
        fres = sample_rate / (FFT_size * 700.0)
        fbklo = 2
        fbkhi = Nby2
        for k in range(0, Nby2):
            chan = lowChan[k]
            if (k < (fbklo - 1) or k > fbkhi):
                fblowWt[k] = 0.0
            else:
                if chan > 0:
                    fblowWt[k]=(( mels[chan+1]) - self.Melk (k+1,fres)) / ((mels[chan+1]) - (mels[chan]))
                else:
                    fblowWt[1] = (( mels [1]) - self.Melk (k+1,fres)) / (( mels[1] - 0))
        return fblowWt


    def fill_bins(self,ek, lowchan, lowWt, numChans, FFT_size):
        valor = ek.shape[0]
        Nby2 = int(FFT_size / 2)
        fbank = np.zeros((valor, numChans + 1))
        fbklo = 2
        fbkhi = Nby2
        for j in range(0, valor):
            for k in range(fbklo, fbkhi):
                bin_idx = lowchan[k]
                t1 = lowWt[k] * ek[j, k]
                if bin_idx > 0:
                    fbank[j, bin_idx] += t1
                if bin_idx < numChans:
                    fbank[j, bin_idx + 1] += ek[j, k] - t1
        return fbank


    def take_logs(self, fbankrec, numChans):
        valor = fbankrec.shape[0]
        fbankdev = np.zeros((valor, numChans + 1))
        for j in range(0, valor):
            for bin in range(1, numChans):
                t1 = fbankrec[j, bin]
                if t1 < 1.0:
                    t1 = 1.0
                fbankdev[j, bin] = np.log(t1)
        return fbankdev


    def dct_htk(self, audio_logfbank, dct_filter_num, filter_len):
        mfnorm = np.sqrt(2.0 / filter_len)
        pi_factor = np.pi / filter_len
        segmentos = audio_logfbank.shape[0]
        numChan = audio_logfbank.shape[1]
        coefs = np.zeros((dct_filter_num - 1, segmentos))
        for j in range(1, dct_filter_num - 1):
            x = j * pi_factor
            for ele in range(1, segmentos):
                resultado = 0.0
                for k in range(1, numChan - 1):
                    resultado += audio_logfbank[ele, k] * np.cos(x * (k - 0.5))
                coefs[j, ele - 1] = resultado * mfnorm
        for ele in range(1, segmentos):
            suma = 0.0
            for k in range(1, numChan - 1):
                suma += audio_logfbank[ele, k]
            coefs[0, ele - 1] = suma * mfnorm
        return coefs

    
    def get_filter_points(self,fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
        fmin_mel = self.freq_to_mel(fmin)
        fmax_mel = self.freq_to_mel(fmax)
        mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
        freqs = self.met_to_freq(mels)
        return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs, mels

    
    
    def mfcc_htk(self,signal=None,sample_rate=None):
        """_summary_

        Args:
            sample_rate : fs. Defaults to None.
            signal: data. Defaults to None.
        """
        audio = signal
        print(f"Sample rate: {sample_rate}Hz")
        print(f"Audio duration: {len(signal) / sample_rate}s")
        # Framing
        pre_emphasis = 0.97
        signal=audio
        emphasized = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
        audio=emphasized
        hop_size = 10  # ms
        FFT_size = 256
        audio_framed = self.frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
        print('Framed audio shape:', audio_framed.shape)
        # Pre-emphasis (applied per-frame)
        frame_len = int(np.round(sample_rate * hop_size / 1000.0))
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        for n in range(frame_num):
            for i in range(FFT_size - 1, 0, -1):
                audio_framed[n, i] = audio_framed[n, i] - (audio_framed[n, i - 1] * pre_emphasis)

        # Windowing
        window = get_window('hamm', FFT_size, fftbins=True)
        audio_win = audio_framed * window

        # FFT
        audio_winT = np.transpose(audio_win)
        audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
        for n in range(audio_fft.shape[1]):
            audio_fft[:, n] = np.fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
        audio_fft = np.transpose(audio_fft)
        
        
        # Mel filter bank
        freq_min = 0
        freq_high = sample_rate / 2
        mel_filter_num = 28
        filter_points, mel_freqs, mels = self.get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=sample_rate)

        lowChan = self.do_melk(filter_points, FFT_size, mel_filter_num, mels, sample_rate)
        lowWt = self.create_vector_lowerChanWeights(filter_points, FFT_size, lowChan, mels, sample_rate)
        audio_window = audio_win
        mag_frames = np.absolute(np.fft.fft((audio_window), FFT_size))  # Magnitude of the FFT
        fbank = self.fill_bins(mag_frames, lowChan, lowWt, mel_filter_num, FFT_size)
        fbanklog = self.take_logs(fbank, mel_filter_num)

        dct_filter_num = 13
        dct_mfcc = self.dct_htk(fbanklog, dct_filter_num, mel_filter_num)
        mfcc_htk = np.transpose(dct_mfcc)
        print('MFCC HTK shape:', mfcc_htk.shape)
        print('MFCC HTK (frames):\n', mfcc_htk)
        return mfcc_htk
    ############################################################################
    ###################### Carga simulada a librosa ##########################
    def load_with_scipy(self, path, sr=None, mono=True, dtype=np.float32, resample_limit_denominator=1000):
        """
        Leer WAV con scipy y comportarse como librosa.load:
        - devuelve (y, sr)
        - y es float32 normalizado en [-1, 1]
        - opcionalmente mezcla a mono (mono=True)
        - opcionalmente re-muestrea a sr (si sr is not None)

        Nota: usa resample_poly (polyphase) para re-muestreo con buena calidad.
        Si prefieres la máxima calidad posible, usa librosa.resample o resampy.
        """
        sr_orig, data = wav.read(path)

        # Convertir a float32 y normalizar si viene en enteros
        if np.issubdtype(data.dtype, np.integer):
            iinfo = np.iinfo(data.dtype)
            # dividir por el valor máximo positivo (igual que librosa/soundfile en la práctica)
            data = data.astype(np.float32) / float(iinfo.max)
        else:
            data = data.astype(np.float32)

        # Mezclar a mono si se requiere
        if mono and data.ndim > 1:
            # librosa hace una mezcla promediando canales
            data = np.mean(data, axis=1)

        # Re-muestrear si se pide una sr distinta
        if sr is not None and sr != sr_orig:
            # Obtener fracción racional aproximada sr/sr_orig para resample_poly
            frac = Fraction(sr, sr_orig).limit_denominator(resample_limit_denominator)
            up, down = frac.numerator, frac.denominator
            # resample_poly espera 1D arrays; si multi-canal, habría que procesar por canal
            if data.ndim > 1:
                # aplicar por canal
                chans = []
                for c in range(data.shape[1]):
                    chans.append(resample_poly(data[:, c], up, down))
                data = np.stack(chans, axis=1)
            else:
                data = resample_poly(data, up, down)
            out_sr = sr
        else:
            out_sr = sr_orig

        # Forzar dtype de salida
        data = data.astype(dtype, copy=False)
        return data, out_sr