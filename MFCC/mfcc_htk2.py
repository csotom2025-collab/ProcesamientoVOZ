"""mfcc_htk2.py

Converted from mfcc_htk2.ipynb — computes MFCC-like features using the same
steps as the notebook. Run from the folder that contains `cero_1.wav`.

Usage:
    python mfcc_htk2.py

This script tries to follow the original notebook logic with minimal changes
to make it runnable as a standalone script.
"""

import os
import sys
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
import matplotlib.pyplot as plt


def load_audio(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        sample_rate, signal = wavfile.read(path)
    except Exception as e:
        raise RuntimeError(f"Error reading WAV file with scipy.io.wavfile: {e}\n"
                           "If the file is compressed try converting to PCM WAV or use librosa.load.")

    # Convert integer PCM to float in range [-1,1] when needed
    if np.issubdtype(signal.dtype, np.integer):
        max_val = np.iinfo(signal.dtype).max
        signal = signal.astype(np.float32) / max_val
    else:
        signal = signal.astype(np.float32)

    # If stereo, make mono by taking the first channel
    if signal.ndim > 1:
        signal = signal[:, 0]

    return sample_rate, signal


def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=44100):
    # hop_size in ms
    frame_len = int(np.round(sample_rate * hop_size / 1000.0))
    if len(audio) < FFT_size:
        # pad with zeros
        pad_width = FFT_size - len(audio)
        audio = np.pad(audio, (0, pad_width), mode='constant')

    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num, FFT_size), dtype=np.float32)
    for n in range(frame_num):
        frames[n] = audio[n * frame_len: n * frame_len + FFT_size]
    return frames


def freq_to_mel(freq):
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def met_to_freq(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)


def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=44100):
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num + 2)
    freqs = met_to_freq(mels)
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs, mels


def Melk(k, fres):
    return 1127.0 * np.log(1.0 + (k - 1) * fres)


def do_melk(filter_points, FFT_size, num_chan, mels, sample_rate):
    fres = sample_rate / (FFT_size * 700.0)
    chan = 0
    Nby2 = int(FFT_size / 2)
    maxChan = num_chan
    fblowChan = np.zeros((Nby2,), dtype=int)

    for k in range(0, Nby2 - 1):
        melk = Melk(k, fres)
        if (k < 2 or k > Nby2):
            fblowChan[k] = -1
        else:
            while (chan <= maxChan and mels[chan] < melk):
                chan += 1
            fblowChan[k] = chan - 1

    for k in range(0, Nby2 - 2):
        fblowChan[k] = fblowChan[k + 1]
    fblowChan[Nby2 - 1] = num_chan
    return fblowChan


def create_vector_lowerChanWeights(filter_points, FFT_size, lowChan, mels, sample_rate):
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
                # Guard division by zero
                denom = (mels[chan + 1] - mels[chan]) if (mels[chan + 1] - mels[chan]) != 0 else 1e-8
                fblowWt[k] = ((mels[chan + 1]) - Melk(k + 1, fres)) / denom
            else:
                denom = (mels[1] - 0) if (mels[1] - 0) != 0 else 1e-8
                fblowWt[1] = ((mels[1]) - Melk(k + 1, fres)) / denom
    return fblowWt


def fill_bins(ek, lowchan, lowWt, numChans, FFT_size):
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


def take_logs(fbankrec, numChans):
    valor = fbankrec.shape[0]
    fbankdev = np.zeros((valor, numChans + 1))
    for j in range(0, valor):
        for bin in range(1, numChans):
            t1 = fbankrec[j, bin]
            if t1 < 1.0:
                t1 = 1.0
            fbankdev[j, bin] = np.log(t1)
    return fbankdev


def dct_htk(audio_logfbank, dct_filter_num, filter_len):
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


def main():
    audio_path = 'cero_1.wav'
    print('Loading', audio_path)
    sample_rate, signal = load_audio(audio_path)
    audio = signal
    print(f"Sample rate: {sample_rate}Hz")
    print(f"Audio duration: {len(signal) / sample_rate}s")

    # Basic plotting (optional) - comment out display in headless environments
    try:
        plt.figure(figsize=(12, 3))
        plt.plot(np.linspace(0, len(audio) / sample_rate, num=len(audio)), audio)
        plt.title('Waveform')
        plt.grid(True)
        plt.tight_layout()
        plt.show()  # Uncomment if you want to display the plot
    except Exception:
        pass

    # Framing
    hop_size = 10  # ms
    FFT_size = 256
    audio_framed = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=sample_rate)
    print('Framed audio shape:', audio_framed.shape)

    # Pre-emphasis (applied per-frame)
    pre_emphasis = 0.97
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
        audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]
    audio_fft = np.transpose(audio_fft)
    mag_frames = np.abs(np.fft.fft(audio_window := audio_win, FFT_size))

    # Mel filter bank
    freq_min = 0
    freq_high = sample_rate / 2
    mel_filter_num = 28
    filter_points, mel_freqs, mels = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=sample_rate)

    lowChan = do_melk(filter_points, FFT_size, mel_filter_num, mels, sample_rate)
    lowWt = create_vector_lowerChanWeights(filter_points, FFT_size, lowChan, mels, sample_rate)

    fbank = fill_bins(mag_frames, lowChan, lowWt, mel_filter_num, FFT_size)
    fbanklog = take_logs(fbank, mel_filter_num)

    dct_filter_num = 13
    dct_mfcc = dct_htk(fbanklog, dct_filter_num, mel_filter_num)
    mfcc_htk = np.transpose(dct_mfcc)
    print('MFCC shape:', mfcc_htk.shape)


if __name__ == '__main__':
    main()
