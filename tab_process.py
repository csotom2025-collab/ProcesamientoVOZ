import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import subprocess

class ProcessTab(ttk.Frame):
    def __init__(self, master, folder_var):
        super().__init__(master)
        self.pack(fill='both', expand=True)
        self.folder_var = folder_var
        self.file_listbox = tk.Listbox(self, selectmode=tk.MULTIPLE)
        self.file_listbox.pack(fill='both', expand=True, padx=10, pady=10)
        self.refresh_btn = ttk.Button(self, text="Actualizar lista", command=self.refresh_file_list)
        self.refresh_btn.pack(fill='x', padx=10, pady=5)
        self.process_btn = ttk.Button(self, text="Procesar en MATLAB", command=self.process_in_matlab)
        self.process_btn.pack(fill='x', padx=10, pady=5)
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
    def process_in_matlab(self):
        selected = [self.file_listbox.get(i) for i in self.file_listbox.curselection()]
        if not selected:
            messagebox.showwarning("Advertencia", "Selecciona al menos un archivo WAV.")
            return
        folder = self.folder_var.get()
        results = ""
        for wav_file in selected:
            wav_path = os.path.join(folder, wav_file)
            cmd = f"matlab -batch \"process_audio('{wav_path.replace('\\', '/')}')\""
            subprocess.run(cmd, shell=True)
            result_file = f"{wav_path}_result.txt"
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    results += f.read() + "\n"
            else:
                results += f"No se encontró resultado para {wav_file}\n"
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, results)
