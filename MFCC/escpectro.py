import os
import numpy as np
import scipy
from scipy.io import wavfile
from scipy.signal import get_window
import IPython.display as Audio
import matplotlib.pyplot as plt
from librosa.display import specshow
import seaborn as sbn
sample_rate, signal = wavfile.read('C:/Users/usuario/Desktop/ProcesamientoVOZ/MFCC/cero_1.wav')  # File assumed to be in the same directory

# Convert stereo to mono by averaging channels if necessary
if signal.ndim > 1:
	y = signal.mean(axis=1)
else:
	y = signal.copy()

# Convert to float in range [-1, 1] if input is integer type
if np.issubdtype(y.dtype, np.integer):
	max_val = np.iinfo(signal.dtype).max
	y = y.astype(np.float32) / float(max_val)
else:
	y = y.astype(np.float32)

print(len(signal))
print("Sample rate: {0}Hz".format(sample_rate))
print("Audio duration: {0}s".format(len(y) / sample_rate))

# Frame parameters
ventana = 256
traslape = 128

# If signal shorter than one window, pad with zeros
if y.shape[0] < ventana:
	pad_len = ventana - y.shape[0]
	y = np.pad(y, (0, pad_len), mode='constant')

# Compute frame indices
inicios = np.arange(0, y.shape[0] - ventana + 1, traslape)
idx = np.arange(ventana)
print('inicios:', inicios[:5], '... total frames:', inicios.shape[0])
idxx = inicios[:, np.newaxis] + idx[np.newaxis, :]
print('idxx shape:', idxx.shape)

# Extract frames: shape (n_frames, ventana)
frames = y[idxx]

# Hanning window (1D) and apply per frame
hann = np.hanning(ventana)
frames_win = frames * hann[np.newaxis, :]

# Compute rFFT for each frame and convert to dB
Yfft = np.fft.rfft(frames_win, axis=1)
eps = 1e-10
YfftdB = 20.0 * np.log10(np.abs(Yfft) + eps)
print('YfftdB shape:', YfftdB.shape)

# Frequencies and time axes
frequencies = np.fft.rfftfreq(ventana, d=1.0/sample_rate)
time = inicios / sample_rate

# Plot spectrogram (freq x time)
plt.figure(figsize=(10, 4))
plt.pcolormesh(time, frequencies, YfftdB.T, shading='gouraud')
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Magnitude (dB)")
plt.ylim([0, sample_rate/2.])
plt.show()

# Show with librosa's display (expects freq x time)
specshow(YfftdB.T, sr=sample_rate, x_axis='time', y_axis='hz')
plt.xlabel('tiempo [s]')
plt.ylabel('frecuencia [Hz]')
plt.colorbar()
plt.show()

# Quick seaborn heatmap (may be slow for large inputs)
try:
	sbn.heatmap(YfftdB.T, xticklabels=10, yticklabels=50)
	plt.show()
except Exception:
	# Ignore plotting errors
	pass

if frequencies.size > 100:
	print('f inicial: ', frequencies[100])
else:
	print('f inicial: None (too few frequency bins)')
if frequencies.size > 200:
	print('f final:  ', frequencies[200])
else:
	print('f final: None (too few frequency bins)')