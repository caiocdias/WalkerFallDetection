import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq

def plot_time_and_spectrum(df, fs=80.0, max_cols=None):
    """
    Plots time-domain signals and their frequency spectrum using FFT.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with each column representing a sensor reading over time.
    fs : float
        Sampling frequency in Hz (default: 80.0).
    max_cols : int, optional
        Maximum number of columns to plot. If None, all columns are plotted.
    """
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    cols = df_numeric.columns[:max_cols] if max_cols else df_numeric.columns
    n = df_numeric.shape[0]
    t = np.arange(n) / fs

    # Plot time-domain signals
    plt.figure()
    for col in cols:
        plt.plot(t, df_numeric[col], label=col)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Time-Domain Sensor Readings')
    plt.legend()
    plt.show()

    # Compute FFT and plot frequency spectrum
    yf = rfft(df_numeric[cols].values, axis=0)
    xf = rfftfreq(n, 1/fs)
    plt.figure()
    for idx, col in enumerate(cols):
        plt.plot(xf, np.abs(yf[:, idx]), label=col)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Frequency Spectrum via FFT')
    plt.legend()
    plt.show()