import os
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import get_window,resample_poly, freqz
from scipy.linalg import solve_toeplitz
from fractions import Fraction
from scipy.fftpack import dct


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
        # Añadida la opción "Filter Banks HTK" para visualizar el espectro mel (bank log)
        self.view_menu = ttk.OptionMenu(top_frame, self.current_view, "Frecuencia", "Frecuencia", "Pitch (Cepstrum)", 
                                "Espectrograma", "MFCC", "MFCC (Custom)", "Filter Banks (Custom)", "Filter Banks HTK", "LPC", command=self.update_view)
        self.view_menu.pack(side=tk.LEFT, padx=5)
        self.analysis_btn = ttk.Button(top_frame, text="Graficar", command=self.apply_analysis)
        self.analysis_btn.pack(side=tk.LEFT, padx=10)
        self.spectrogram_cmap = tk.StringVar(value="viridis")
        self.cmap_menu = ttk.OptionMenu(top_frame, self.spectrogram_cmap, "viridis", "viridis", "jet", "gray", "Greys")
        self.cmap_menu.pack_forget()  # Oculto por defecto
        # Paneles de control debajo de la gráfica (crearlos antes de referenciarlos)
        self.move_frame = ttk.Frame(self)
        self.interval_frame = ttk.Frame(self)
        self.options_frame = ttk.Frame(self)
        self.analysis_result_label = ttk.Label(self, text="")
        # Control para ordenar LPC (solo visible cuando se selecciona la vista LPC)
        self.lpc_order_var = tk.IntVar(value=12)
        self.lpc_order_label = ttk.Label(self.options_frame, text="Orden LPC:")
        # Usamos tk.Spinbox por compatibilidad
        self.lpc_order_spin = tk.Spinbox(self.options_frame, from_=1, to=100, textvariable=self.lpc_order_var, width=5)
        # Ocultar por defecto
        self.lpc_order_label.pack_forget()
        self.lpc_order_spin.pack_forget()
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
    ############################################################################
    ##################### Control de la vista ##################################
    ############################################################################
    def update_view(self, *args):
        view = self.current_view.get()
        # Mostrar/ocultar colormap solo para espectrograma/MFCC/Mel/Filter Banks
        if view in ["Espectrograma", "MFCC", "MFCC (Custom)", "Filter Banks (Custom)", "Filter Banks HTK"]:
            self.cmap_menu.pack(side=tk.LEFT, padx=5)
        else:
            self.cmap_menu.pack_forget()

        # Ajustar etiquetas de intervalo: tiempo para espectrograma/MFCC/Pitch, frecuencia para el resto
        if view in ["Espectrograma", "MFCC", "MFCC (Custom)", "Pitch (Cepstrum)", "Filter Banks HTK"]:
            self.label_inicio.config(text="Inicio (s):")
            self.label_fin.config(text="Fin (s):")
        else:
            self.label_inicio.config(text="Inicio (Hz):")
            self.label_fin.config(text="Fin (Hz):")

        # Mostrar control de orden LPC solo en vista LPC
        if view == "LPC":
            self.lpc_order_label.pack(side=tk.LEFT, padx=5)
            self.lpc_order_spin.pack(side=tk.LEFT, padx=5)
        else:
            self.lpc_order_label.pack_forget()
            self.lpc_order_spin.pack_forget()

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
        view = self.current_view.get()
        if view == "Frecuencia":
            # Para vista de frecuencia, usar la frecuencia de muestreo/2
            if hasattr(self, 'current_fs'):
                self.ax.set_xlim(0, self.current_fs/2)
            else:
                self.ax.set_xlim(0, 5000)  # valor por defecto
        elif view in ["Espectrograma", "MFCC", "MFCC (Custom)", "Filter Banks (Custom)", "Filter Banks HTK"]:
            # Para espectrograma y MFCC, usar la duración total de la señal
            if hasattr(self, 'current_data') and hasattr(self, 'current_fs'):
                self.ax.set_xlim(0, len(self.current_data)/self.current_fs)
            else:
                self.ax.set_xlim(0, 1)  # valor por defecto
        else:
            # Para otras vistas, restaurar a los valores por defecto
            self.ax.set_xlim(0, 1)
            
        # También restaurar los límites Y si es necesario
        if view in ["Espectrograma", "MFCC", "MFCC (Custom)", "Filter Banks (Custom)", "Filter Banks HTK"]:
            if hasattr(self, 'current_fs'):
                self.ax.set_ylim(0, self.current_fs/2)
                
        self.canvas.draw()

    def save_current_view(self):
        """Open a file dialog and save the current figure to the chosen path."""
        # Suggest a default filename using current file and view
        filename = self.analysis_file_var.get() or "figure"
        view = self.current_view.get() if hasattr(self, 'current_view') else 'view'
        default_name = f"{os.path.splitext(filename)[0]}_{view}.png"
        filetypes = [("PNG image","*.png"), ("JPEG image","*.jpg"), ("PDF file","*.pdf"), ("SVG file","*.svg")]
        # Guardar en carpeta ImgFrecuencia dentro de la carpeta seleccionada si es posible
        folder = None
        try:
            folder = self.folder_var.get() if hasattr(self, 'folder_var') else None
        except Exception:
            folder = None
        if folder and os.path.isdir(folder):
            img_dir = os.path.join(folder, 'ImgFrecuencia')
            try:
                os.makedirs(img_dir, exist_ok=True)
            except Exception:
                img_dir = folder
        else:
            img_dir = os.getcwd()

        path = filedialog.asksaveasfilename(defaultextension='.png', filetypes=filetypes, initialfile=default_name, initialdir=img_dir, title='Guardar imagen como')
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
    ############################################################################
    ##################### Analisis de la señal  ################################
    ############################################################################
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
        data, fs = self.cargaAudio(path)
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
            self.ax.set_xlim(0, len(data)/fs)  # Establecer límites iniciales
            self.ax.set_ylim(0, fs/2)
            self.colorbar = self.figure.colorbar(im, ax=self.ax, label="dB", orientation='vertical')
            self.figure.subplots_adjust(right=0.85)
            self.analysis_result_label.config(text="")
        elif view == "MFCC":
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            mfccs, fbanklog = self.mfcc_htk(data, fs)

            # Calcular frecuencias centrales de los filtros Mel (Hz) para eje Y
            freq_min = 0
            freq_high = fs / 2
            mel_filter_num = 28
            filter_points, mel_freqs, mels = self.get_filter_points(freq_min, freq_high, mel_filter_num, 256, sample_rate=fs)
            # Centros de los filtros Mel (excluyendo los bordes)
            hz_centers = self.met_to_freq(mels[1:-1])

            t = np.linspace(0, len(data)/fs, mfccs.shape[0])
            im = self.ax.imshow(mfccs.T, aspect='auto', origin='lower',
                                cmap=self.spectrogram_cmap.get(),
                                extent=[0, t[-1], hz_centers[0], hz_centers[-1]])

            # Ticks del eje Y en Hz
            yticks = np.linspace(hz_centers[0], hz_centers[-1], 6)
            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels([f'{int(f)} Hz' for f in yticks])

            self.ax.set_xlabel("Tiempo [s]")
            self.ax.set_ylabel("Frecuencia [Hz]")
            self.ax.set_title(f"MFCC HTK: {filename}")
            self.colorbar = self.figure.colorbar(im, ax=self.ax, label="Amplitud", orientation='vertical')
            self.figure.subplots_adjust(right=0.85)
            self.analysis_result_label.config(text="")
        elif view == "MFCC (Custom)":
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            mfccs, t_mfcc, filter_banks = self.calcular_mfcc(data, fs)
            
            # Calcular las frecuencias Mel para el eje Y
            fmin_mel = self.freq_to_mel(0)
            fmax_mel = self.freq_to_mel(fs/2)
            mel_freqs = np.linspace(fmin_mel, fmax_mel, mfccs.shape[1])
            freq_ticks = self.met_to_freq(mel_freqs)  # Convertir a Hz
            
            im = self.ax.imshow(mfccs.T, aspect='auto', origin='lower', 
                            cmap=self.spectrogram_cmap.get(), 
                            extent=[t_mfcc[0], t_mfcc[-1], freq_ticks[0], freq_ticks[-1]])
            
            # Configurar ticks del eje Y en frecuencias
            yticks = np.linspace(freq_ticks[0], freq_ticks[-1], 6)
            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels([f'{int(f)} Hz' for f in yticks])
            
            self.ax.set_xlabel("Tiempo [s]")
            self.ax.set_ylabel("Frecuencia [Hz]")
            self.ax.set_title(f"MFCC Custom : {filename}")
            self.colorbar = self.figure.colorbar(im, ax=self.ax, label="Valor", orientation='vertical')
            self.figure.subplots_adjust(right=0.85)
            self.analysis_result_label.config(text="")
        elif view == "Filter Banks (Custom)":
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            mfccs, t_mfcc, filter_banks = self.calcular_mfcc(data, fs)
            
            # Calcular las frecuencias Mel para el eje Y
            fmin_mel = self.freq_to_mel(0)
            fmax_mel = self.freq_to_mel(fs/2)
            mel_freqs = np.linspace(fmin_mel, fmax_mel, filter_banks.shape[1])
            freq_ticks = self.met_to_freq(mel_freqs)  # Convertir a Hz
            
            # Los filter_banks ya están en dB desde calcular_mfcc
            im = self.ax.imshow(filter_banks.T, aspect='auto', origin='lower', 
                            cmap=self.spectrogram_cmap.get(), 
                            extent=[t_mfcc[0], t_mfcc[-1], freq_ticks[0], freq_ticks[-1]])
            
            # Configurar ticks del eje Y en frecuencias
            yticks = np.linspace(freq_ticks[0], freq_ticks[-1], 6)
            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels([f'{int(f)} Hz' for f in yticks])
            
            self.ax.set_xlabel("Tiempo [s]")
            self.ax.set_ylabel("Frecuencia [Hz]")
            self.ax.set_title(f"Filter Banks Custom: {filename}")
            self.colorbar = self.figure.colorbar(im, ax=self.ax, label="Energía (dB)", orientation='vertical')
            self.figure.subplots_adjust(right=0.85)
            self.analysis_result_label.config(text="")
        elif view == "Filter Banks HTK":
            # Nueva vista: muestra el espectro mel (log-mel filterbank)
            self.figure.clf()
            self.ax = self.figure.add_subplot(111)
            mfccs, fbanklog = self.mfcc_htk(data, fs)
            
            # Calcular las frecuencias Mel para el eje Y
            fmin_mel = self.freq_to_mel(0)
            fmax_mel = self.freq_to_mel(fs/2)
            mel_freqs = np.linspace(fmin_mel, fmax_mel, fbanklog.shape[1])
            freq_ticks = self.met_to_freq(mel_freqs)  # Convertir a Hz
            
            im = self.ax.imshow(fbanklog.T, aspect='auto', origin='lower', 
                            cmap=self.spectrogram_cmap.get(), 
                            extent=[0, len(data)/fs, freq_ticks[0], freq_ticks[-1]])
            
            # Configurar ticks del eje Y en frecuencias
            yticks = np.linspace(freq_ticks[0], freq_ticks[-1], 6)
            self.ax.set_yticks(yticks)
            self.ax.set_yticklabels([f'{int(f)} Hz' for f in yticks])
            
            self.ax.set_xlabel("Tiempo [s]")
            self.ax.set_ylabel("Frecuencia [Hz]")
            self.ax.set_title(f"Filter Banks HTK : {filename}")
            self.colorbar = self.figure.colorbar(im, ax=self.ax, label="Log energía", orientation='vertical')
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
            elif view == "LPC":
                # Mostrar solo la comparación espectral FFT vs Sobre LPC (frame central)
                NFFT = 2048
                LPC_ORDER = int(self.lpc_order_var.get())
                # Framing (25 ms windows, 10 ms hop)
                frame_len = int(0.025 * fs)
                hop_len = int(0.01 * fs)

                # Obtener frames (ventaneo ya aplica ventana)
                try:
                    frames = self.ventaneo(data, frame_len, hop_len, window_type='hamming')
                except Exception:
                    # Fallback: simple framing sin ventana aplicada
                    num_frames = 1 + int((len(data) - frame_len) / hop_len)
                    frames = np.zeros((max(0, num_frames), frame_len))
                    for i in range(num_frames):
                        start = i * hop_len
                        frames[i] = data[start:start+frame_len]

                if frames.size == 0:
                    self.analysis_result_label.config(text="No hay suficientes muestras para LPC.")
                else:
                    # Calcular LPC usando la implementación interna (sin librosa)
                    lpc_matrix = self.calcular_lpc(data, fs, frame_size_ms=25.0, lpc_order=LPC_ORDER, hop_size_ms=10.0)
                    if lpc_matrix is None or lpc_matrix.size == 0:
                        self.analysis_result_label.config(text="No se pudieron calcular LPCs.")
                        self.canvas.draw()
                        return

                    n_frames = lpc_matrix.shape[0]
                    # Limpiar figura y usar un solo eje
                    self.figure.clf()
                    self.ax = self.figure.add_subplot(111)

                    # Elegir frame central
                    frame_index = min(n_frames - 1, n_frames // 2)
                    if frame_index < frames.shape[0]:
                        signal_frame = frames[frame_index]
                    else:
                        signal_frame = np.zeros(frame_len)

                    coeffs = lpc_matrix[frame_index]
                    lpc_coeffs = np.concatenate(([1.0], coeffs))

                    # Respuesta en frecuencia del filtro LPC (sobre espectral)
                    try:
                        w, h_lpc = freqz(b=1.0, a=lpc_coeffs, worN=NFFT, fs=fs)
                    except Exception:
                        w, h_lpc = freqz(b=1.0, a=lpc_coeffs, worN=NFFT)
                        w = w * fs / (2 * np.pi)

                    spectrum_lpc_db = 20 * np.log10(np.abs(h_lpc) + 1e-10)

                    # Espectro de la señal (FFT)
                    signal_fft = np.fft.rfft(signal_frame, n=NFFT)
                    frequencies_fft = np.fft.rfftfreq(NFFT, 1.0/fs)
                    spectrum_signal_db = 20 * np.log10(np.abs(signal_fft) + 1e-10)

                    # Escalar LPC para visualización
                    scaling_factor = np.max(spectrum_signal_db) - np.max(spectrum_lpc_db)
                    spectrum_lpc_db = spectrum_lpc_db + scaling_factor

                    # Dibujar comparación en un único plot
                    self.ax.plot(frequencies_fft, spectrum_signal_db, label='Espectro Original (FFT)', alpha=0.7)
                    self.ax.plot(w, spectrum_lpc_db, label=f'Sobre Espectral (LPC Orden {LPC_ORDER})', color='darkorange', linewidth=2.0)
                    self.ax.set_title(f'Comparación de Espectro: FFT vs. LPC (Frame {frame_index})')
                    self.ax.set_xlabel('Frecuencia (Hz)')
                    self.ax.set_ylabel('Magnitud (dB)')
                    self.ax.set_xlim(0, fs/2)
                    self.ax.legend()
                    self.ax.grid(True, which='both', linestyle='--', alpha=0.5)
                    self.figure.tight_layout()
                    self.analysis_result_label.config(text="")
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
    ############################################################################
    ##################### Preparacion de la señal ##############################
    ############################################################################
    def preenfasis(self, signal, pre_emphasis_coeff=0.97):
        """Aplica un filtro de preénfasis a la señal de audio."""
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis_coeff * signal[:-1])
        return emphasized_signal
    
    def ventaneo(self, signal, frame_size, hop_size, window_type='hamming'):
        """Divide la señal en frames con ventana aplicada."""
        num_frames = 1 + int((len(signal) - frame_size) / hop_size)
        frames = np.zeros((num_frames, frame_size))
        window = get_window(window_type, frame_size, fftbins=True)
        for i in range(num_frames):
            start = i * hop_size
            frames[i] = signal[start:start + frame_size] * window
        return frames
    
    def audioFFT(self, audio_win, FFT_size=2048):
        audio_winT = np.transpose(audio_win)
        audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')
        for n in range(audio_fft.shape[1]):
            audio_fft[:, n] = np.fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
        audio_fft = np.transpose(audio_fft)
        return audio_fft
    
    def segmentacionhtk(self, audio, FFT_size=2048, hop_size=10, sample_rate=44100):
        frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        frames = np.zeros((frame_num,FFT_size))
        for n in range(frame_num):
            frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
        return frames
    
    def segmentacion(self, audio, frame_size, hop_size, sample_rate=None):
        """
        Divide la señal en frames usando el mismo método que HTK.
        Args:
            audio: señal de audio
            frame_size: tamaño del frame en muestras
            hop_size: salto entre frames en muestras
            sample_rate: frecuencia de muestreo (opcional, no usado)
        Returns:
            frames: matriz de frames
        """
        # Calcular número de frames como HTK
        signal_length = len(audio)
        num_frames = int(np.ceil(float(signal_length - frame_size + hop_size) / hop_size))
        
        # Crear matriz de frames con padding si es necesario
        pad_length = (num_frames - 1) * hop_size + frame_size
        if pad_length > signal_length:
            pad_signal = np.append(audio, np.zeros(pad_length - signal_length))
        else:
            pad_signal = audio
            
        frames = np.zeros((num_frames, frame_size))
        for i in range(num_frames):
            start = i * hop_size
            frames[i] = pad_signal[start:start + frame_size]
            
        return frames
    
    def cargaAudio(self, path, sr=None, mono=True, dtype=np.float32, resample_limit_denominator=1000):
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


    ############################################################################
    ##################### MFCC HTK IMPLEMENTACION ##############################
    ############################################################################
    
    def freq_to_mel(self, freq):
        return 2595.0 * np.log10(1.0 + freq / 700.0)


    def met_to_freq(self, mels):
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)

    def Melk(self, k, fres):
        return 1127.0 * np.log(1.0 + (k - 1) * fres)
    
    def omaFreq(self, k, fres):
        return 'Ecuacion de homar'

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
                #print (melk)
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
        filter_points = np.floor((FFT_size + 1) * freqs / sample_rate).astype(int)
        return filter_points, freqs, mels

    
    
    def mfcc_htk(self, signal=None, sample_rate=None):
        """
        Calcula los coeficientes MFCC usando el método HTK.
        Args:
            sample_rate : fs. Defaults to None.
            signal: data. Defaults to None.
        """
        print(f"Sample rate: {sample_rate}Hz")
        print(f"Audio duration: {len(signal) / sample_rate}s")
        # Framing
        pre_emphasis = 0.97
        emphasized = self.preenfasis(signal, pre_emphasis_coeff=pre_emphasis)
        audio = emphasized
        hop_size = 10  # ms
        FFT_size = 256
        audio_framed = self.segmentacionhtk(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
        print('Framed audio shape:', audio_framed.shape)
        frame_len = int(np.round(sample_rate * hop_size / 1000.0))
        frame_num = int((len(audio) - FFT_size) / frame_len) + 1
        for n in range(frame_num):
            for i in range(FFT_size - 1, 0, -1):
                audio_framed[n, i] = self.preenfasis(audio_framed[n, :], pre_emphasis_coeff=pre_emphasis)[i]

        # Windowing
        audio_win = self.ventaneo(audio, FFT_size, frame_len, window_type='hamming')

        # FFT (usar audioFFT y su magnitud)
        audio_fft = self.audioFFT(audio_win, FFT_size=FFT_size)
        mag_frames = np.abs(audio_fft)
        print('FFT (magnitude):\n', mag_frames)

        # Mel filter bank
        freq_min = 0
        freq_high = sample_rate / 2
        mel_filter_num = 28
        filter_points, mel_freqs, mels = self.get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=sample_rate)

        lowChan = self.do_melk(filter_points, FFT_size, mel_filter_num, mels, sample_rate)
        lowWt = self.create_vector_lowerChanWeights(filter_points, FFT_size, lowChan, mels, sample_rate)
        # Usar mag_frames en lugar de recalcular la FFT
        fbank = self.fill_bins(mag_frames, lowChan, lowWt, mel_filter_num, FFT_size)
        fbanklog = self.take_logs(fbank, mel_filter_num)

        dct_filter_num = 13
        dct_mfcc = self.dct_htk(fbanklog, dct_filter_num, mel_filter_num)
        mfcc_htk = np.transpose(dct_mfcc)
        # Retornar tanto los MFCCs como el log-mel filterbank para visualización adicional
        # mfcc_htk: (n_frames, n_mfcc)
        # fbanklog: (n_frames, n_mel_bins+1)
        print('########### MFCC HTK & Log-Mel Filterbank Calculos ###########')
        print('MFCC HTK shape:', mfcc_htk.shape)
        print('MFCC HTK (frames):\n', mfcc_htk)
        print('Log-Mel Filterbank shape:', fbanklog.shape)
        print('Log-Mel Filterbank (frames):\n', fbanklog)
        return mfcc_htk, fbanklog
    
    
    def calcular_mfcc(self, data, fs, n_mfcc=12, win_len=0.025, hop_len=0.01):
        """
        Calcula los coeficientes MFCC de una señal de audio sin librerías externas.
        Args:
            data: señal de audio (numpy array)
            fs: frecuencia de muestreo
            n_mfcc: número de coeficientes MFCC (por defecto 12 como HTK)
            win_len: longitud de ventana en segundos (por defecto 25ms como HTK)
            hop_len: salto entre ventanas en segundos (por defecto 10ms como HTK)
        Returns:
            mfccs: matriz de coeficientes MFCC (ventanas x coeficientes)
            t_mfcc: vector de tiempo de cada ventana
            filter_banks: matriz de filter banks (ventanas x filtros)
        """
        # 1. Pre-emphasis
        emphasized = self.preenfasis(data, pre_emphasis_coeff=0.97)
        # 2. Framentación
        frame_len = int(win_len * fs)  # 25ms * fs
        frame_step = int(hop_len * fs)  # 10ms * fs
        frames = self.segmentacion(emphasized, frame_len, frame_step, fs)
        # 3. Ventanamiento
        frames *= self.ventaneo(np.ones(frame_len), frame_len, frame_len, window_type='hamming')[0]
        # 4. FFT y Power Spectrum
        NFFT = 256  # Como HTK
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
        pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))

        # 5. Crear y aplicar filtros Mel
        low_freq_mel = self.freq_to_mel(0)
        high_freq_mel = self.freq_to_mel(fs/2)
        n_filt = 29  # Como HTK
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filt + 2)
        hz_points = self.met_to_freq(mel_points)
        bin_points = np.floor((NFFT + 1) * hz_points / fs).astype(int)

        # Crear matriz de filter banks
        fbank = np.zeros((n_filt, NFFT // 2 + 1))
        # Construir los filtros triangulares
        for i in range(n_filt):
            for j in range(int(bin_points[i]), int(bin_points[i + 1])):
                fbank[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
            for j in range(int(bin_points[i + 1]), int(bin_points[i + 2])):
                fbank[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])
        # Normalizar los filtros
        fbank = fbank / np.maximum(np.sum(fbank, axis=1)[:, np.newaxis], 1e-8)
        # Aplicar los filter banks
        filter_banks = np.dot(pow_frames, fbank.T)
        # Convertir a dB y manejar valores pequeños
        filter_banks = np.where(filter_banks <= 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)
        # 6. Coeficientes DCT para obtener MFCC (sin transponer)
        mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, :n_mfcc]
        # 7. Tiempo de cada ventana
        t_mfcc = np.arange(len(mfccs)) * hop_len
        print('##############MFCC Custom calculated###############')
        print('MFCC Custom shape:', mfccs.shape)
        print('MFCC Custom (frames):\n', mfccs)
        print('Filter Banks Custom shape:', filter_banks.shape)
        print('Filter Banks Custom (frames):\n', filter_banks)
        return mfccs, t_mfcc, filter_banks
    ############################################################################
    ##################### CALCULO LPC ##########################################
    ############################################################################
    def lpc_costum(self, frame: np.ndarray, order: int):
        """
        Extrae los coeficientes LPC de un frame usando el método de autocorrelación.
        Resuelve las ecuaciones de Yule-Walker: R·a = r
        """
        N = len(frame)
        frame_autocorr_full = np.correlate(frame, frame, mode='full')
        mid_point = N - 1
        autocorr = frame_autocorr_full[mid_point : mid_point + order + 1]
        # En las ecuaciones de Yule-Walker la forma es R·a = -r (nota el signo -)
        # donde r = autocorr[1:order+1]. Por tanto aplicamos el signo negativo al RHS.
        r_col = autocorr[0:order]
        r_rhs = -autocorr[1:order+1]

        # Regularización numérica: si r_col[0] es muy pequeño o la matriz parece
        # cercana a singuralidad, añadimos un pequeño término en la diagonal (equivalente
        # a incrementar r_col[0]) para estabilizar la solución.
        eps = 1e-8
        if np.abs(r_col[0]) < eps:
            r_col = r_col.copy()
            r_col[0] += eps

        try:
            lpc_coeffs = solve_toeplitz(r_col, r_rhs)
            # Asegurar tipo float
            return np.asarray(lpc_coeffs, dtype=float)
        except np.linalg.LinAlgError:
            # Si falla, devolver ceros para no romper el flujo
            return np.zeros(order, dtype=float)
        
    def calcular_lpc(self, signal: np.ndarray, fs: int, frame_size_ms: float = 25.0, lpc_order: int = 12, hop_size_ms: float = 10.0):
        """
        Extrae coeficientes LPC por frame usando framing de frame_size_ms (ms) y hop_size_ms (ms).
        Esta versión procura reproducir la segmentación y ventana usadas en LpcComparacion.py
        (25 ms frames, 10 ms hop, ventana Hamming).
        Retorna una matriz de forma (n_frames, lpc_order) con los coeficientes (sin el 1.0 inicial).
        """
        if signal is None or len(signal) == 0:
            return np.empty((0, lpc_order))

        # Pre-énfasis (igual aproximación que en el ejemplo)
        pre_emphasis = 0.97
        emphasized = self.preenfasis(signal, pre_emphasis_coeff=pre_emphasis)

        # Calcular tamaños de ventana en muestras
        frame_len = int(np.round(frame_size_ms * fs / 1000.0))
        hop_len = int(np.round(hop_size_ms * fs / 1000.0))
        if frame_len <= 0:
            frame_len = 1
        if hop_len <= 0:
            hop_len = 1

        # Ventaneo (hamming)
        try:
            frames = self.ventaneo(emphasized, frame_len, hop_len, window_type='hamming')
        except Exception:
            # Fallback manual
            num_frames = 1 + int((len(emphasized) - frame_len) / hop_len)
            frames = np.zeros((max(0, num_frames), frame_len))
            for i in range(num_frames):
                start = i * hop_len
                frames[i] = emphasized[start:start + frame_len] * np.hamming(frame_len)

        if frames.size == 0:
            return np.empty((0, lpc_order))

        m = frames.shape[0]
        LPC_COf = np.zeros((m, lpc_order))

        for i, frm in enumerate(frames):
            LPC_COf[i, :] = self.lpc_costum(frm, lpc_order)

        return LPC_COf