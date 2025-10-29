import librosa
import numpy as np
import scipy.signal
from scipy.linalg import solve_toeplitz
import matplotlib.pyplot as plt  # <-- Importamos matplotlib

# --- Parámetros Configurables ---
LPC_ORDER = 100
FRAME_SEC = 0.025
STEP_SEC = 0.010
PRE_EMPHASIS = 0.97
N_FFT = 2048  # <-- Definimos un tamaño para la FFT (para visualización)

# --- Ruta al archivo de audio ---
# !! IMPORTANTE: Cambia esto a la ruta de TU archivo de voz del dataset !!
try:
    FILE_PATH = 'AudiosVoz/A_1.wav'  # <-- Ruta al archivo de audio de voz
except Exception:
    print("No se pudo cargar la muestra de voz, usando 'trumpet' como ejemplo.")
    FILE_PATH = librosa.example('trumpet')


def extract_lpc_custom(frame: np.ndarray, order: int) -> np.ndarray:
    """
    Extrae los coeficientes LPC de un frame usando el método de autocorrelación.
    Resuelve las ecuaciones de Yule-Walker: R·a = r
    """
    N = len(frame)
    frame_autocorr_full = np.correlate(frame, frame, mode='full')
    mid_point = N - 1
    autocorr = frame_autocorr_full[mid_point : mid_point + order + 1]
    
    try:
        lpc_coeffs = solve_toeplitz(
            autocorr[0:order],
            autocorr[1:order+1]
        )
        return lpc_coeffs
    except np.linalg.LinAlgError:
        return np.zeros(order)

def main():
    """
    Función principal para cargar audio, procesarlo, extraer LPC y visualizar.
    """
    print(f"Procesando archivo: {FILE_PATH}")
    
    # 1. Cargar la señal de voz
    try:
        y, sr = librosa.load(FILE_PATH, sr=None)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {FILE_PATH}")
        return

    print(f"Señal cargada: {len(y)} muestras, Frecuencia de muestreo: {sr} Hz")

    # 2. Pre-énfasis
    y_preemphasized = scipy.signal.lfilter([1, -PRE_EMPHASIS], 1, y)

    # 3. Calcular parámetros de segmentación (framing)
    frame_length_samples = int(FRAME_SEC * sr)
    step_length_samples = int(STEP_SEC * sr)
    
    print(f"Configuración de LPC: Orden p={LPC_ORDER}")
    print(f"Segmentación: Frames de {frame_length_samples} muestras ({FRAME_SEC*1000} ms)")

    # 4. Segmentación (Framing)
    frames = librosa.util.frame(
        y_preemphasized,
        frame_length=frame_length_samples,
        hop_length=step_length_samples
    )
    frames = frames.T

    # 5. Enventanado (Windowing)
    window = scipy.signal.get_window('hamming', frame_length_samples)
    frames = frames * window

    # --- Extracción de Características ---
    
    m = frames.shape[0]
    n = LPC_ORDER        
    
    lpc_matrix_custom = np.zeros((m, n))
    lpc_matrix_librosa = np.zeros((m, n + 1))

    print(f"\nExtrayendo LPCs para {m} segmentos...")

    for i, frame in enumerate(frames):
        lpc_matrix_custom[i, :] = extract_lpc_custom(frame, LPC_ORDER)
        lpc_matrix_librosa[i, :] = librosa.lpc(frame, order=LPC_ORDER)

    print("¡Extracción completada!")

    # --- Verificación y Comparación (como antes) ---
    
    print("\n--- Verificación de Dimensiones ---")
    print(f"Forma de la matriz LPC (Custom): {lpc_matrix_custom.shape}")
    
    # ... (código de comparación omitido por brevedad) ...
    
    # -----------------------------------------------
    # --- VISUALIZACIÓN DE RESULTADOS ---
    # -----------------------------------------------

    print("\nGenerando visualizaciones...")

    # --- Visualización 1: Matriz LPC (m x n) como Heatmap ---
    
    plt.figure(figsize=(10, 6))
    # Usamos .T para transponer: (n x m)
    # Así, el eje Y son los 'n' coeficientes y el eje X son los 'm' frames (tiempo)
    plt.imshow(
        lpc_matrix_custom.T, 
        aspect='auto', 
        origin='lower', 
        extent=[0, m * STEP_SEC, 0, n]
    )
    plt.colorbar(label='Valor del Coeficiente')
    plt.title(f'Matriz de Coeficientes LPC (Orden $p={LPC_ORDER}$)')
    plt.xlabel('Tiempo (segundos)')
    plt.ylabel('Índice del Coeficiente LPC ($n$)')
    plt.tight_layout()

    # --- Visualización 2: Sobre Espectral (LPC vs FFT) ---
    
    # Elegimos un frame del medio (suele tener más señal)
    frame_index = m // 2
    
    # Obtenemos el frame original (ya enventanado)
    signal_frame = frames[frame_index]
    
    # Obtenemos los coeficientes LPC de librosa para ESE frame
    # (Usamos los de librosa [1, a1, a2...] porque freqz los necesita así)
    lpc_coeffs_frame = lpc_matrix_librosa[frame_index]

    # 1. Calcular el espectro del filtro LPC (el "sobre")
    # El filtro LPC es 1/A(z), donde A(z) son los lpc_coeffs
    # 'freqz' nos da la respuesta en frecuencia de este filtro
    w, h_lpc = scipy.signal.freqz(
        b=1.0,  # Numerador (Ganancia G, aquí simplificada a 1)
        a=lpc_coeffs_frame, # Denominador (los coeficientes LPC)
        worN=N_FFT, 
        fs=sr
    )
    # Convertimos a decibelios (dB)
    # Usamos 1e-10 para evitar log(0)
    spectrum_lpc_db = 20 * np.log10(np.abs(h_lpc) + 1e-10)

    # 2. Calcular el espectro de la señal original (FFT)
    # Usamos rfft (FFT real) ya que la señal es real
    signal_fft = np.fft.rfft(signal_frame, n=N_FFT)
    frequencies_fft = np.fft.rfftfreq(N_FFT, 1.0/sr)
    
    # Convertimos a decibelios (dB)
    spectrum_signal_db = 20 * np.log10(np.abs(signal_fft) + 1e-10)

    # 3. Graficar ambos
    
    # Escalamos el espectro LPC para que coincida visualmente
    # (Esto es solo para visualización, ya que no calculamos la ganancia 'G' real)
    scaling_factor = np.max(spectrum_signal_db) - np.max(spectrum_lpc_db)
    spectrum_lpc_db += scaling_factor
    
    plt.figure(figsize=(10, 6))
    
    # Trazamos el espectro de la señal real (azul)
    plt.plot(
        frequencies_fft, 
        spectrum_signal_db, 
        label='Espectro Original (FFT)', 
        alpha=0.7
    )
    
    # Trazamos el sobre LPC (naranja)
    plt.plot(
        w, 
        spectrum_lpc_db, 
        label=f'Sobre Espectral (LPC Orden {LPC_ORDER})', 
        color='darkorange', 
        linewidth=2.5
    )
    
    plt.title(f'Comparación de Espectro: FFT vs. LPC (Frame {frame_index})')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlim(0, sr / 2) # Solo mostrar hasta la frecuencia de Nyquist
    plt.tight_layout()

    # Mostrar todas las gráficas generadas
    plt.show()


if __name__ == "__main__":
    main()