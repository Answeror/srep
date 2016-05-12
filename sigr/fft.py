from __future__ import division
import numpy as np


def fft(data, fs):
    n = data.shape[-1]
    window = np.hanning(n)
    windowed = data * window
    spectrum = np.fft.fft(windowed)
    freq = np.fft.fftfreq(n, 1 / fs)
    half_n = np.ceil(n / 2)
    spectrum_half = (2 / n) * spectrum[..., :half_n]
    freq_half = freq[:half_n]
    return freq_half, np.abs(spectrum_half)
