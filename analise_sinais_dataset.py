# -*- coding: utf-8 -*-
"""
@author: Caio Cezar, Isabely e Thaissa
Adaptado para leitura de dataset com sinais reais.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.signal import butter, lfilter, freqz
import random

plt.close('all')

# LEITURA E ORGANIZAÇÃO DO DATASET

df = pd.read_csv("full_dataset.csv")

idle_samples = df[df['label'] == 'idle'].iloc[:500, 1:]
motion_samples = df[df['label'] == 'motion'].iloc[:500, 1:]

idle_matrix = idle_samples.T
motion_matrix = motion_samples.T

combined_matrix = pd.concat([idle_matrix, motion_matrix], axis=1)

# PARÂMETROS DO SINAL

fs = 80
tf = combined_matrix.shape[0] / fs
t = np.linspace(0, tf, combined_matrix.shape[0], endpoint=False)

# PLOTAR 3 SINAIS ALEATÓRIOS 

random_indices = random.sample(range(combined_matrix.shape[1]), 3)

for idx in random_indices:
    y = combined_matrix.iloc[:, idx].values
    classe = 'motion' if idx >= 500 else 'idle'
    plt.figure()
    plt.plot(t, y)
    plt.title(f"Sinal aleatório (coluna {idx}) - Classe: {classe}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.show()

    # FFT 
    N = len(y)
    yf = rfft(y)
    ff = rfftfreq(N, 1/fs)
    yf_abs_norm = np.abs(yf) / max(np.abs(yf))

    plt.figure()
    plt.plot(ff, yf_abs_norm)
    plt.title(f"Espectro de Frequências - Coluna {idx}")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Amplitude Normalizada")
    plt.show()

    # FILTRAGEM 
    def butter_lowpass(cutoff, sample_rate, order):
        return butter(order, cutoff, fs=sample_rate, btype='low', analog=False)

    fc = 10     # Frequência de corte (Hz)
    nfilt = 5   # Ordem do filtro
    b, a = butter_lowpass(fc, fs, nfilt)

    y_filt = lfilter(b, a, y)

    plt.figure()
    plt.plot(t, y, label="Original")
    plt.plot(t, y_filt, label="Filtrado", color='r')
    plt.title(f"Sinal Antes e Depois da Filtragem - Coluna {idx}")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.show()

    # FFT do sinal filtrado
    yf_filt = rfft(y_filt)
    yfilt_abs_norm = np.abs(yf_filt) / max(np.abs(yf_filt))

    plt.figure()
    plt.plot(ff, yfilt_abs_norm)
    plt.title(f"Espectro do Sinal Filtrado - Coluna {idx}")
    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Amplitude Normalizada")
    plt.show()
    
