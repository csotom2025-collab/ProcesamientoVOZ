import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import dct
import datetime
import tab_analysis as tim
import tab_frequency as frequency

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
            'freq': tk.BooleanVar(value=True),
            'energia': tk.BooleanVar(value=True),
            'cruces': tk.BooleanVar(value=True),
            'mfcc': tk.BooleanVar(value=True),
            'lpc': tk.BooleanVar(value=True),
            'espectro': tk.BooleanVar(value=True)
        }
        
        # Crear checkboxes
        ttk.Checkbutton(checks_frame, text="Frecuencia fundamental", 
                        variable=self.check_vars['freq']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checks_frame, text="Energía", 
                        variable=self.check_vars['energia']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checks_frame, text="Cruces por cero", 
                        variable=self.check_vars['cruces']).pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(checks_frame, text="MFCC", 
                        variable=self.check_vars['mfcc']).pack(side=tk.LEFT, padx=5)
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
                fs, data = wav.read(wav_path)
                if len(data.shape) > 1:  # Si es estéreo, convertir a mono
                    data = np.mean(data, axis=1)
                
                # Formatear resultados
                results += f"=== Análisis de {wav_file} ===\n"
                
                # Calcular frecuencia fundamental si está seleccionada
                if self.check_vars['freq'].get():
                    frame_length = int(0.025 * fs)  # 25ms
                    frames = np.array_split(data, len(data) // frame_length)
                    freqs = []
                    for frame in frames:
                        if len(frame) >= frame_length:
                            autocorr = np.correlate(frame, frame, mode='full')
                            autocorr = autocorr[len(autocorr)//2:]
                            peaks = np.where((autocorr[1:-1] > autocorr[0:-2]) & (autocorr[1:-1] > autocorr[2:]))[0] + 1
                            if len(peaks) > 0:
                                first_peak = peaks[0]
                                if first_peak > 0:
                                    freq = fs / first_peak
                                    if 50 <= freq <= 500:  # Rango típico de voz
                                        freqs.append(freq)
                    freq_fundamental = np.mean(freqs) if freqs else 0
                    results += f"Frecuencia fundamental: {freq_fundamental:.2f} Hz\n"
                
                # Calcular energía si está seleccionada
                if self.check_vars['energia'].get():
                    energia = np.sum(data**2)
                    results += f"Energía: {energia:.4f}\n"
                
                # Calcular cruces por cero si está seleccionado
                if self.check_vars['cruces'].get():
                    cruces = np.sum(np.diff(np.signbit(data)))
                    results += f"Cruces por cero: {cruces}\n"
                
                # Calcular MFCC si está seleccionado
                if self.check_vars['mfcc'].get():
                    # llamar al método de instancia definiéndole self=None porque
                    # la implementación de calcular_mfcc no utiliza atributos de instancia
                    mfccs, t_mfcc = frequency.FrequencyAnalysisTab.calcular_mfcc(None, data, fs)
                    if mfccs is not None:
                        results += "MFCC Coeficientes:\n"
                        # Para cada ventana de tiempo
                        for i, coef_ventana in enumerate(mfccs):
                            results += f"Ventana {i + 1}:\n"
                            # Para cada coeficiente en la ventana
                            for j, coef in enumerate(coef_ventana):
                                results += f"  Coef {j + 1}: {coef:.4f}\n"
                        results += "\n"
                
                # Calcular LPC si está seleccionado
                #if self.check_vars['lpc'].get():
                    
                    #lpc_coeffs = calculate_lpc(wav_path)
                    #if lpc_coeffs is not None:
                        #results += f"LPC (primeros 5 coeficientes): {lpc_coeffs[:5]}\n"
                
                # Calcular espectro si está seleccionado
                if self.check_vars['espectro'].get():
                    # llamar al método de instancia pasando None como self
                    S, t_spec, f_spec = frequency.FrequencyAnalysisTab.calcular_espectrograma(None, data, fs)
                    if S is not None:
                        # S tiene forma (frecuencia x tiempo); tomar la energía promedio por frecuencia
                        spectrum = np.mean(S, axis=1)
                        results += f"Espectro: {spectrum}\n"
                
                results += "\n"
                
            except Exception as e:
                results += f"Error procesando {wav_file}: {str(e)}\n\n"
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, results)
        # Guardar todos los resultados en un archivo de texto con marca temporal
        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            out_name = f"analysis_results_{wav_file}.txt"
            out_path = os.path.join(folder, out_name)
            with open(out_path, 'w', encoding='utf-8') as out_f:
                out_f.write(results)
            # Informar al usuario dónde se guardó
            messagebox.showinfo("Resultados guardados", f"Resultados guardados en:\n{out_path}")
        except Exception as e:
            messagebox.showwarning("Error guardando", f"No se pudo guardar el archivo de resultados: {e}")
