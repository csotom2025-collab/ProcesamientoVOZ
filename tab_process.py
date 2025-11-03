import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import resample_poly
import datetime
import tab_analysis as tim
import tab_frequency as frequency
from fractions import Fraction

class ProcessTab(ttk.Frame):
    def __init__(self, master, folder_var):
        super().__init__(master)
        self.pack(fill='both', expand=True)
        self.folder_var = folder_var
        
        # Lista de archivos
        self.file_listbox = tk.Listbox(self, selectmode=tk.MULTIPLE)
        self.file_listbox.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Frame para los botones principales
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', padx=10, pady=5)
        
        self.refresh_btn = ttk.Button(button_frame, text="Actualizar lista", command=self.refresh_file_list)
        self.refresh_btn.pack(side=tk.LEFT, fill='x', expand=True, padx=5)
        
        self.process_btn = ttk.Button(button_frame, text="Analizar Audio", command=self.process_audio)
        self.process_btn.pack(side=tk.LEFT, fill='x', expand=True, padx=5)
        
        # Frame para los checkboxes
        checks_frame = ttk.LabelFrame(self, text="Seleccionar análisis a realizar")
        checks_frame.pack(fill='x', padx=5, pady=5)
        
        # Variables para los checkboxes
        self.check_vars = {
            'freq': tk.BooleanVar(value=False),
            'energia': tk.BooleanVar(value=False),
            'cruces': tk.BooleanVar(value=False),
            'mfcc': tk.BooleanVar(value=False),
            'mfcc_custom': tk.BooleanVar(value=False),
            'lpc': tk.BooleanVar(value=False),
            'espectro': tk.BooleanVar(value=False)
        }
        
        # Crear checkboxes
        ttk.Checkbutton(checks_frame, text="Pitch (Cepstrum)", 
                variable=self.check_vars['freq']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checks_frame, text="Energía", 
                variable=self.check_vars['energia']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checks_frame, text="Cruces por cero", 
                variable=self.check_vars['cruces']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checks_frame, text="MFCC estándar", 
                variable=self.check_vars['mfcc']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checks_frame, text="MFCC custom", 
                variable=self.check_vars['mfcc_custom']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checks_frame, text="LPC", 
                variable=self.check_vars['lpc']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checks_frame, text="Espectro", 
                variable=self.check_vars['espectro']).pack(side=tk.LEFT, padx=5)
            
        # Área de resultados
        self.result_text = tk.Text(self, height=10)
        self.result_text.pack(fill='both', padx=10, pady=10)
        
        self.refresh_file_list()

    def refresh_file_list(self):
        self.file_listbox.delete(0, tk.END)
        folder = self.folder_var.get()
        if not os.path.isdir(folder):
            return
        wav_files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        for f in wav_files:
            self.file_listbox.insert(tk.END, f)
    def choose_folder(self):
        folder_selected = filedialog.askdirectory(initialdir=self.folder_var.get(), title="Selecciona carpeta de archivos WAV")
        if folder_selected:
            self.folder_var.set(folder_selected)
            self.refresh_file_list()
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

    def process_audio(self):
        selected = [self.file_listbox.get(i) for i in self.file_listbox.curselection()]
        if not selected:
            messagebox.showwarning("Advertencia", "Selecciona al menos un archivo WAV.")
            return
        
        folder = self.folder_var.get()
        results = ""
        
        for wav_file in selected:
            wav_path = os.path.join(folder, wav_file)
            try:
                # Leer el archivo WAV
                data,fs=self.cargaAudio(wav_path)
                if len(data.shape) > 1:  # Si es estéreo, convertir a mono
                    data = np.mean(data, axis=1)
                
                # Formatear resultados
                results += f"=== Análisis de {wav_file} ===\n"
                
                # Calcular pitch_cepstrum si está seleccionada
                if self.check_vars['freq'].get():
                    win_len = int(0.04 * fs)  # 40 ms
                    hop_len = int(0.01 * fs)  # 10 ms
                    pitches = []
                    times = []
                    for start in range(0, len(data) - win_len, hop_len):
                        segment = data[start:start+win_len]
                        pitch = frequency.FrequencyAnalysisTab.pitch_cepstrum(None,segment, fs)
                        pitches.append(pitch)
                        times.append(start/fs)

                    # Calcular pitch promedio de manera segura (evitar listas vacías/NaN)
                    valid_pitches = [p for p in pitches if np.isfinite(p) and 50 < p < 500]
                    if valid_pitches:
                        avg_pitch = float(np.mean(valid_pitches))
                        results += f"Pitch promedio: {avg_pitch:.2f} Hz\n"
                    else:
                        results += "Pitch promedio: N/A\n"
                
                # Calcular energía si está seleccionada
                if self.check_vars['energia'].get():
                    energia = tim.AnalysisTab.calcular_energia(None, data)
                    results += f"Energía: {energia:.4f}\n"
                
                # Calcular cruces por cero si está seleccionado
                if self.check_vars['cruces'].get():
                    cruces = tim.AnalysisTab.cruces_por_cero(None, data)
                    results+=f"Cruces por cero: {len(cruces)}\n puntos: ["
                    for punto in cruces:
                        results += f"{punto}, "
                    results = results[:-2]  
                    results += "]\n"
                
                # Calcular MFCC estándar si está seleccionado
                if self.check_vars['mfcc'].get():
                    helper = frequency.FrequencyAnalysisTab.__new__(frequency.FrequencyAnalysisTab)
                    mfccs_return = frequency.FrequencyAnalysisTab.mfcc_htk(helper, data, fs)
                    if mfccs_return is not None:
                        if isinstance(mfccs_return, (tuple, list)):
                            mfccs = mfccs_return[0]
                        else:
                            mfccs = mfccs_return
                        mfccs = np.asarray(mfccs)
                        if mfccs.size == 0:
                            results += "MFCC: no hay coeficientes disponibles\n"
                        else:
                            results += "MFCC Coeficientes estándar:[num ventana [coeficientes ventana]]\n"
                            for i in range(mfccs.shape[0]):
                                coef_ventana = mfccs[i]
                                coef_ventana = np.asarray(coef_ventana).ravel()
                                results += f"[{i + 1} ["
                                for k, coef in enumerate(coef_ventana):
                                    try:
                                        results += f"{float(coef):.4f}, "
                                    except Exception:
                                        row_str = np.array2string(coef_ventana, precision=4, separator=', ')
                                        results += row_str + ", "
                                        break
                                if results.endswith(', '):
                                    results = results[:-2]
                                results += f"]]\n"
                            results += "\n"
                # Calcular MFCC custom si está seleccionado
                if self.check_vars['mfcc_custom'].get():
                    try:
                        helper = frequency.FrequencyAnalysisTab.__new__(frequency.FrequencyAnalysisTab)
                        mfccs_custom = frequency.FrequencyAnalysisTab.calcular_mfcc(helper, data, fs)

                        # `calcular_mfcc` puede devolver un ndarray o una tupla/lista
                        # similar a la rama MFCC estándar; manejar ambos casos.
                        if isinstance(mfccs_custom, (tuple, list)):
                            if len(mfccs_custom) == 0:
                                mfccs_arr = np.asarray([])
                            else:
                                mfccs_arr = np.asarray(mfccs_custom[0])
                        else:
                            mfccs_arr = np.asarray(mfccs_custom)

                        if mfccs_arr.size == 0:
                            results += "MFCC custom: no hay coeficientes disponibles\n"
                        else:
                            results += "MFCC Coeficientes custom:[num ventana [coeficientes ventana]]\n"
                            for i in range(mfccs_arr.shape[0]):
                                coef_ventana = mfccs_arr[i]
                                coef_ventana = np.asarray(coef_ventana).ravel()
                                results += f"[{i + 1} ["
                                for k, coef in enumerate(coef_ventana):
                                    try:
                                        results += f"{float(coef):.4f}, "
                                    except Exception:
                                        row_str = np.array2string(coef_ventana, precision=4, separator=', ')
                                        results += row_str + ", "
                                        break
                                if results.endswith(', '):
                                    results = results[:-2]
                                results += f"]]\n"
                            results += "\n"
                    except Exception as e:
                        results += f"MFCC custom: error al calcular: {e}\n"
                
                # Calcular LPC si está seleccionado
                if self.check_vars['lpc'].get():
                    # Intentar obtener el valor del control de orden LPC de forma robusta.
                    try:
                        app = getattr(self.master, 'master', None)
                        if app is not None and hasattr(app, 'tab_frequency'):
                            lpc_order = int(app.tab_frequency.lpc_order_var.get())
                        else:
                            # Fallback: intentar obtener el widget por su path en el notebook
                            try:
                                lpc_order = int(self.master.nametowidget('.!notebook.!frequencyanalysistab').lpc_order_var.get())
                            except Exception:
                                # Último recurso: usar un valor por defecto razonable
                                lpc_order = 12
                    except Exception:
                        lpc_order = 12
                    try:
                        helper = frequency.FrequencyAnalysisTab.__new__(frequency.FrequencyAnalysisTab)
                        lpc_coeffs = frequency.FrequencyAnalysisTab.calcular_lpc(helper, data, fs, lpc_order)
                        if lpc_coeffs is not None:
                            results += f"LPC Coeficientes :\n"
                            for i, coef_ventana in enumerate(lpc_coeffs):
                                results += f"[{i + 1} ["
                                for j, coef in enumerate(coef_ventana):
                                    results += f"{coef:.4f}, "
                                results = results[:-2]  
                                results += f"]]"
                        results += "\n"
                    except Exception as e:
                        results += f"LPC: error calculando LPC: {e}\n"
                
                # Calcular espectro si está seleccionado
                if self.check_vars['espectro'].get():
                    # Crear una instancia ligera sin __init__ y llamar al método de instancia
                    helper = frequency.FrequencyAnalysisTab.__new__(frequency.FrequencyAnalysisTab)
                    S, t_spec, f_spec = frequency.FrequencyAnalysisTab.calcular_espectrograma(helper, data, fs)
                    if S is not None:
                        # S tiene forma (frecuencia x tiempo); tomar la energía promedio por frecuencia
                        spectrum = np.mean(S, axis=1)
                        # Formatear el array de forma segura para evitar errores de format()
                        try:
                            spectrum_str = np.array2string(spectrum, precision=4, separator=', ')
                        except Exception:
                            spectrum_str = str(spectrum.tolist())
                        results += f"Espectro: {spectrum_str}\n"
                
                results += "\n"
                
            except Exception as e:
                results += f"Error procesando {wav_file}: {str(e)}\n\n"
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, results)
        # Guardar todos los resultados en un archivo de texto dentro de la carpeta AnalisisTXT
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            analysis_dir = os.path.join(folder, 'AnalisisTXT')
            os.makedirs(analysis_dir, exist_ok=True)
            base_name = os.path.splitext(wav_file)[0]
            out_name = f"analysis_results_{base_name}_{timestamp}.txt"
            out_path = os.path.join(analysis_dir, out_name)
            with open(out_path, 'w', encoding='utf-8') as out_f:
                out_f.write(results)
            # Informar al usuario dónde se guardó
            messagebox.showinfo("Resultados guardados", f"Resultados guardados en:\n{out_path}")
        except Exception as e:
            messagebox.showwarning("Error guardando", f"No se pudo guardar el archivo de resultados: {e}")
