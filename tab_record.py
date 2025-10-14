import tkinter as tk
from tkinter import ttk, filedialog
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import threading, time

class RecordTab(ttk.Frame):
    def __init__(self, master, refresh_callback, folder_var):
        super().__init__(master)
        self.pack(fill='both', expand=True)
        self.folder_var = folder_var
        self.input_devices = [d['name'] for d in sd.query_devices() if d['max_input_channels'] > 0]
        self.device_var = tk.StringVar()
        self.device_var.set(self.input_devices[0] if self.input_devices else "")
        ttk.Label(self, text="Dispositivo de entrada:").pack(pady=5)
        self.device_menu = ttk.OptionMenu(self, self.device_var, self.device_var.get(), *self.input_devices)
        self.device_menu.pack(pady=5)
        folder_frame = ttk.Frame(self)
        folder_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(folder_frame, text="Carpeta de archivos WAV:").pack(side=tk.LEFT)
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.folder_var, width=40)
        self.folder_entry.pack(side=tk.LEFT, padx=5)
        self.folder_btn = ttk.Button(folder_frame, text="Escoger carpeta", command=self.choose_folder)
        self.folder_btn.pack(side=tk.LEFT)
        duration_frame = ttk.Frame(self)
        duration_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(duration_frame, text="Duración (segundos):").pack(side=tk.LEFT)
        self.duration_var = tk.StringVar(value="1")
        self.duration_entry = ttk.Entry(duration_frame, textvariable=self.duration_var, width=10)
        self.duration_entry.pack(side=tk.LEFT, padx=5)
        self.record_btn = ttk.Button(self, text="Grabar", command=self.record_audio)
        self.record_btn.pack(fill='x', padx=10, pady=10)
        self.timer_label = ttk.Label(self, text="Tiempo: 0.0 s")
        self.timer_label.pack(fill='x', padx=10, pady=5)
        self.save_label = ttk.Label(self, text="")
        self.save_label.pack(fill='x', padx=10, pady=10)
        self.refresh_callback = refresh_callback

    def choose_folder(self):
        folder_selected = filedialog.askdirectory(initialdir=self.folder_var.get(), title="Selecciona carpeta de archivos WAV")
        if folder_selected:
            self.folder_var.set(folder_selected)
            if self.refresh_callback:
                self.refresh_callback()

    def record_audio(self):
        fs = 44100
        try:
            duration = float(self.duration_var.get())
        except ValueError:
            self.save_label.config(text="Duración inválida.")
            return
        device_name = self.device_var.get()
        device_index = None
        for idx, d in enumerate(sd.query_devices()):
            if d['name'] == device_name:
                device_index = idx
                break
        self.save_label.config(text="Grabando...")
        self.timer_label.config(text="Tiempo: 0.0 s")
        self.update()
        audio = np.zeros((int(duration * fs), 1), dtype='float32')
        def update_timer(elapsed):
            self.timer_label.config(text=f"Tiempo: {elapsed:.1f} s")
            self.update()
        def record_thread():
            sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32', device=device_index, out=audio)
            for t in range(int(duration * 10) + 1):
                elapsed = t / 10.0
                update_timer(elapsed)
                time.sleep(0.1)
            sd.wait()
            update_timer(duration)
            norm_audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            sd.play(norm_audio, fs)
            sd.wait()
            filename = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
            if filename:
                wav.write(filename, fs, (norm_audio * 32767).astype(np.int16))
                self.save_label.config(text=f"Guardado: {filename}")
                if self.refresh_callback:
                    self.refresh_callback()
            else:
                self.save_label.config(text="Grabación cancelada.")
        threading.Thread(target=record_thread).start()
