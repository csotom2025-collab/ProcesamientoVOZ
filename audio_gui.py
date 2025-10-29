import tkinter as tk
from tkinter import ttk
from tab_record import RecordTab
from tab_process import ProcessTab
from tab_analysis import AnalysisTab
from tab_frequency import FrequencyAnalysisTab
from tab_segment_analysis import SegmentAnalysisTab



class AudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Grabador y Procesador de Audio")
        self.geometry("1208x720")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)
        self.folder_var = tk.StringVar(value="")
        self.tab_record = RecordTab(self.notebook, self.refresh_all, self.folder_var)
        self.notebook.add(self.tab_record, text="Grabar Audio")
        self.tab_process = ProcessTab(self.notebook, self.folder_var)
        self.notebook.add(self.tab_process, text="Procesar en Audio")
        self.tab_analysis = AnalysisTab(self.notebook, self.folder_var)
        self.notebook.add(self.tab_analysis, text="Análisis en Tiempo")
        self.tab_frequency = FrequencyAnalysisTab(self.notebook, self.folder_var)
        self.notebook.add(self.tab_frequency, text="Análisis en Frecuencia")
        self.tab_segment = SegmentAnalysisTab(self.notebook, self.get_signal_callback)
        self.notebook.add(self.tab_segment, text="Análisis por Segmento")

    # Callback para obtener la señal actual
    def get_signal_callback(self):
        if hasattr(self.tab_frequency, 'current_data') and hasattr(self.tab_frequency, 'current_fs') and hasattr(self.tab_frequency, 'analysis_file_var'):
            if self.tab_frequency.current_data is not None and self.tab_frequency.current_fs is not None:
                filename = self.tab_frequency.analysis_file_var.get() if self.tab_frequency.analysis_file_var.get() else "---"
                return self.tab_frequency.current_data, self.tab_frequency.current_fs, filename
        return None

    def refresh_all(self):
        if hasattr(self, 'tab_process'):
            self.tab_process.refresh_file_list()
        if hasattr(self, 'tab_analysis'):
            self.tab_analysis.refresh_analysis_file_list()
        if hasattr(self, 'tab_frequency'):
            self.tab_frequency.refresh_analysis_file_list()

if __name__ == "__main__":
    app = AudioApp()
    app.mainloop()
