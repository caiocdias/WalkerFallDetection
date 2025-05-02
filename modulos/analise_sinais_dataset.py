import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_signals(df: pd.DataFrame, max_cols: int = None, fs: float = 80.0):
    # Garantir apenas dados numéricos
    df_numeric = df.select_dtypes(include=[np.number])

    # Selecionar colunas com base em max_cols
    cols_to_plot = df_numeric.columns[:max_cols] if max_cols is not None else df_numeric.columns

    # Vetor de tempo
    n_samples = df_numeric.shape[0]
    t = np.arange(n_samples) / fs

    # Plotagem
    plt.figure(figsize=(15, 7))
    for col in cols_to_plot:
        plt.plot(t, df_numeric[col], label=f'Sensor: {col}')

    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title('Sinais no domínio do tempo')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()