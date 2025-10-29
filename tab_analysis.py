import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import resample_poly
import numpy as np
import scipy.io.wavfile as wav
import os
from fractions import Fraction

class AnalysisTab(ttk.Frame):
    
    def __init__(self, master, folder_var):
        super().__init__(master)
        self.pack(fill='both', expand=True)
        self.folder_var = folder_var
        analysis_frame = ttk.Frame(self)
        analysis_frame.pack(fill='x', padx=10, pady=5)
        top_frame = ttk.Frame(self)
        top_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(top_frame, text="Archivo WAV:").pack(side=tk.LEFT, padx=10)
        self.analysis_file_var = tk.StringVar()
        self.analysis_file_menu = ttk.OptionMenu(top_frame, self.analysis_file_var, "", *[], command=self.update_analysis_file)
        self.analysis_file_menu.pack(side=tk.LEFT, padx=5)
        ttk.Label(top_frame, text="Vista:").pack(side=tk.LEFT, padx=10)
        self.current_view = tk.StringVar(value="Forma de onda")
        self.view_menu = ttk.OptionMenu(top_frame, self.current_view, "Forma de onda", "Forma de onda", "Energía", "Cruces por cero", command=self.update_view)
        self.view_menu.pack(side=tk.LEFT, padx=5)
        self.analysis_btn = ttk.Button(top_frame, text="Aplicar y Graficar", command=self.apply_analysis)
        self.analysis_btn.pack(side=tk.LEFT, padx=10)
        self.play_btn = ttk.Button(top_frame, text="Reproducir audio", command=self.play_audio)
        self.play_btn.pack(side=tk.LEFT, padx=10)
        self.figure = plt.Figure(figsize=(6,3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        self.analysis_result_label = ttk.Label(self, text="")
        self.analysis_result_label.pack(fill='x', padx=10, pady=5)
        self.show_cruces_var = tk.BooleanVar(value=True)
        self.cruces_check = ttk.Checkbutton(
            self, text="Mostrar cruces por cero",
            variable=self.show_cruces_var,
            command=self.toggle_cruces_check
            )
        move_frame = ttk.Frame(self)
        move_frame.pack(fill='x', padx=10, pady=5)
        self.left_btn = ttk.Button(move_frame, text="<", width=3)
        self.left_btn.pack(side=tk.LEFT, padx=2)
        self.left_btn.bind('<ButtonPress-1>', lambda e: self._start_repeat(self.move_view, -1))
        self.left_btn.bind('<ButtonRelease-1>', lambda e: self._stop_repeat())
        self.right_btn = ttk.Button(move_frame, text=">", width=3)
        self.right_btn.pack(side=tk.LEFT, padx=2)
        self.right_btn.bind('<ButtonPress-1>', lambda e: self._start_repeat(self.move_view, 1))
        self.right_btn.bind('<ButtonRelease-1>', lambda e: self._stop_repeat())
        self.zoom_in_btn = ttk.Button(move_frame, text="+", width=3)
        self.zoom_in_btn.pack(side=tk.LEFT, padx=2)
        self.zoom_in_btn.bind('<ButtonPress-1>', lambda e: self._start_repeat(self.zoom_in))
        self.zoom_in_btn.bind('<ButtonRelease-1>', lambda e: self._stop_repeat())
        self.zoom_out_btn = ttk.Button(move_frame, text="-", width=3)
        self.zoom_out_btn.pack(side=tk.LEFT, padx=2)
        self.zoom_out_btn.bind('<ButtonPress-1>', lambda e: self._start_repeat(self.zoom_out))
        self.zoom_out_btn.bind('<ButtonRelease-1>', lambda e: self._stop_repeat())
        interval_frame = ttk.Frame(self)
        interval_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(interval_frame, text="Inicio (s):").pack(side=tk.LEFT)
        self.zoom_start_var = tk.StringVar(value="0")
        self.zoom_start_entry = ttk.Entry(interval_frame, textvariable=self.zoom_start_var, width=8)
        self.zoom_start_entry.pack(side=tk.LEFT, padx=5)
        ttk.Label(interval_frame, text="Fin (s):").pack(side=tk.LEFT)
        self.zoom_end_var = tk.StringVar(value="5")
        self.zoom_end_entry = ttk.Entry(interval_frame, textvariable=self.zoom_end_var, width=8)
        self.zoom_end_entry.pack(side=tk.LEFT, padx=5)
        self.apply_zoom_btn = ttk.Button(interval_frame, text="Aplicar zoom", command=self.apply_manual_zoom)
        self.apply_zoom_btn.pack(side=tk.LEFT, padx=10)
        options_frame = ttk.Frame(self)
        options_frame.pack(fill='x', padx=10, pady=5)
        self.reset_zoom_btn = ttk.Button(options_frame, text="Reiniciar zoom", command=self.reset_zoom)
        self.reset_zoom_btn.pack(side=tk.LEFT, padx=10)
        self.save_image_btn = ttk.Button(options_frame, text="Guardar imagen", command=self.save_current_view)
        self.save_image_btn.pack(side=tk.LEFT, padx=10)
        self.refresh_analysis_file_list()
        self._repeat_job = None
        self._repeat_func = None
        self._repeat_args = None
        

    
    def toggle_cruces_check(self):
        # Guarda el zoom actual antes de actualizar la gráfica
        prev_xlim = self.ax.get_xlim()
        self.apply_analysis(keep_xlim=True)
        self.ax.set_xlim(prev_xlim)
        self.canvas.draw()

    def play_audio(self):
        import sounddevice as sd
        folder = self.folder_var.get()
        filename = self.analysis_file_var.get()
        if not filename:
            self.analysis_result_label.config(text="Selecciona un archivo WAV para reproducir.")
            return
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            self.analysis_result_label.config(text="El archivo no existe.")
            return
        fs, data = wav.read(path)
        if data.ndim > 1:
            data = data[:,0]
        sd.stop()
        sd.play(data, fs)
        self.analysis_result_label.config(text=f"Reproduciendo: {filename}")

    def update_view(self, *args):
        if self.current_view.get() == "Cruces por cero":
            self.cruces_check.pack(fill='x', padx=10, pady=5)
        else:
            self.cruces_check.pack_forget()
        self.apply_analysis(keep_xlim=True)

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
            img_dir = os.path.join(folder, 'ImgTiempo')
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

    def update_analysis_file(self, value):
        self.analysis_file_var.set(value)

    def calcular_energia(self, data):
        """Calcula la energía de la señal de audio."""
        return np.sum(data**2)

    def cruces_por_cero(self, data):
        """Calcula los índices de los cruces por cero en la señal de audio."""
        return np.where(np.abs(np.diff(np.sign(data))) > 0)[0]

    def apply_analysis(self, keep_xlim=True):
        folder = self.folder_var.get()
        filename = self.analysis_file_var.get()
        if not filename:
            self.analysis_result_label.config(text="Selecciona un archivo WAV.")
            return
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            self.analysis_result_label.config(text="El archivo no existe.")
            return
        data,fs= self.cargaAudio(path, sr=None, mono=True, dtype=np.float32)
        if data.ndim > 1:
            data = data[:,0]
        data = data.astype(float)
        self.current_data = data
        self.current_fs = fs
        t = np.arange(len(data))/fs

        
        self.ax.clear()
        view = self.current_view.get()
        energia = self.calcular_energia(data)
        cruces = self.cruces_por_cero(data)
        if view == "Forma de onda":
            xlim = self.ax.get_xlim() if keep_xlim else (0, min(5, t[-1]))
            self.ax.plot(t, data, label="Señal")
            self.ax.set_xlim(xlim)
            self.ax.set_xlabel("Tiempo [s]")
            self.ax.set_ylabel("Amplitud")
            self.ax.set_title(f"Forma de onda: {filename}")
            self.ax.legend()
        elif view == "Energía":
            self.ax.plot(t, data**2, color='g', label="Energía instantánea")
            self.ax.set_xlabel("Tiempo [s]")
            self.ax.set_ylabel("Energía")
            self.ax.set_title("Energía de la señal")
            self.ax.legend()
        elif view == "Cruces por cero":
            self.ax.plot(t, data, label="Señal")
            if self.show_cruces_var.get():
                self.ax.plot(t[cruces], data[cruces], 'ro', label="Cruces por cero")
            self.ax.set_xlabel("Tiempo [s]")
            self.ax.set_ylabel("Amplitud")
            self.ax.set_title("Cruces por cero")
            self.ax.legend()
        self.analysis_result_label.config(text=f"Energía total: {energia:.4f} | Cruces por cero: {len(cruces)}")
        self.canvas.draw()

    def apply_manual_zoom(self):
        try:
            start = float(self.zoom_start_var.get())
            end = float(self.zoom_end_var.get())
        except ValueError:
            self.analysis_result_label.config(text="Intervalo inválido.")
            return
        if hasattr(self, 'current_data') and hasattr(self, 'current_fs'):
            data = self.current_data
            fs = self.current_fs
            t = np.arange(len(data))/fs
            if start < 0 or end > t[-1] or start >= end:
                self.analysis_result_label.config(text="Intervalo fuera de rango.")
                return
            self.ax.set_xlim(start, end)
            self.canvas.draw()

    def reset_zoom(self):
        if hasattr(self, 'current_data') and hasattr(self, 'current_fs'):
            data = self.current_data
            fs = self.current_fs
            t = np.arange(len(data))/fs
            self.ax.set_xlim(0, t[-1])
            self.canvas.draw()

    def move_view(self, direction):
        if hasattr(self, 'current_data') and hasattr(self, 'current_fs'):
            data = self.current_data
            fs = self.current_fs
            t = np.arange(len(data))/fs
            xlim = self.ax.get_xlim()
            window = xlim[1] - xlim[0]
            shift = window * 0.1 * direction  # 10% del tamaño de ventana
            new_start = max(0, xlim[0] + shift)
            new_end = min(t[-1], xlim[1] + shift)
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
        width = (xlim[1] - xlim[0]) * 0.9  # Reduce solo 10%
        new_start = max(0, center - width/2)
        new_end = center + width/2
        self.ax.set_xlim(new_start, new_end)
        self.canvas.draw()

    def zoom_out(self):
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        width = (xlim[1] - xlim[0]) * 1.1  # Aumenta solo 10%
        total = self.current_data.shape[0] / self.current_fs if hasattr(self, 'current_data') and hasattr(self, 'current_fs') else xlim[1]
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
