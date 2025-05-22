import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt

def plot_signals(df: pd.DataFrame, max_cols: int = None, fs: float = 80.0, nome: str = None):
    df_numeric = df.select_dtypes(include=[np.number])

    cols_to_plot = df_numeric.columns[:max_cols] if max_cols is not None else df_numeric.columns
    print(cols_to_plot)
    n_samples = df_numeric.shape[0]
    t = np.arange(n_samples) / fs

    plt.figure(figsize=(15, 7))
    for col in cols_to_plot:
        plt.plot(t, df_numeric[col], label=f'Sensor: {col}')

    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Sinais no domínio do tempo {nome}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_fft(df: pd.DataFrame, max_cols: int = None, fs: float = 80.0, nome: str = None):
    df_numeric = df.select_dtypes(include=[np.number])

    cols_to_plot = df_numeric.columns[:max_cols] if max_cols is not None else df_numeric.columns

    n = df_numeric.shape[0]
    freqs = np.fft.rfftfreq(n, d=1/fs)

    fft_vals = np.fft.rfft(df_numeric.values, axis=0)
    mag = np.abs(fft_vals) / n

    plt.figure(figsize=(15, 7))
    for idx, col in enumerate(cols_to_plot):
        plt.plot(freqs, mag[:, idx], label=f'Sensor: {col}')

    plt.xlabel('Frequência (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Espectro de Frequência {nome or ""}'.strip())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def extrair_atributos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai atributos estatísticos de um DataFrame onde cada coluna é um sinal.
    """
    atributos = {
        'pico_max': df.max(),
        'pico_min': df.min(),
        'media': df.mean(),
        'desvio_padrao': df.std(),
        'skewness': df.apply(skew),
        'kurtosis': df.apply(kurtosis),
        'rms': np.sqrt((df ** 2).mean()),
        'energia': (df ** 2).sum()
    }
    return pd.DataFrame(atributos)


dataset_raw = pd.read_csv(r"./Dataset/full_dataset.csv")

idle_samples  = dataset_raw[dataset_raw['label'] == 'idle'].iloc[:500, 1:].reset_index(drop=True)
motion_samples = dataset_raw[dataset_raw['label'] == 'motion'].iloc[:500, 1:].reset_index(drop=True)

idle_matrix   = idle_samples.T
motion_matrix = motion_samples.T

idle_matrix.columns   = [f"idle{i}" for i in idle_matrix.columns]
motion_matrix.columns = [f"motion{j}" for j in motion_matrix.columns]

prefixes = ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']

idle_acc_x_df = idle_matrix.loc[idle_matrix.index.str.startswith('acc_x'), :]
idle_acc_y_df = idle_matrix.loc[idle_matrix.index.str.startswith('acc_y'), :]
idle_acc_z_df = idle_matrix.loc[idle_matrix.index.str.startswith('acc_z'), :]
idle_gy_x_df  = idle_matrix.loc[idle_matrix.index.str.startswith('gy_x'), :]
idle_gy_y_df  = idle_matrix.loc[idle_matrix.index.str.startswith('gy_y'), :]
idle_gy_z_df  = idle_matrix.loc[idle_matrix.index.str.startswith('gy_z'), :]

motion_acc_x_df = motion_matrix.loc[motion_matrix.index.str.startswith('acc_x'), :]
motion_acc_y_df = motion_matrix.loc[motion_matrix.index.str.startswith('acc_y'), :]
motion_acc_z_df = motion_matrix.loc[motion_matrix.index.str.startswith('acc_z'), :]
motion_gy_x_df = motion_matrix.loc[motion_matrix.index.str.startswith('gy_x'), :]
motion_gy_y_df = motion_matrix.loc[motion_matrix.index.str.startswith('gy_y'), :]
motion_gy_z_df = motion_matrix.loc[motion_matrix.index.str.startswith('gy_z'), :]

combined_acc_x_df = pd.concat([motion_acc_x_df, idle_acc_x_df], axis=1)
combined_acc_x_df = combined_acc_x_df[['motion1', 'motion2', 'motion3', 'idle1', 'idle2', 'idle3']]

atributos_acc_x = extrair_atributos(combined_acc_x_df)
print(atributos_acc_x)

combined_acc_y_df = pd.concat([motion_acc_y_df, idle_acc_y_df], axis=1)
combined_acc_y_df = combined_acc_y_df[['motion1', 'motion2', 'motion3', 'idle1', 'idle2', 'idle3']]

atributos_acc_y = extrair_atributos(combined_acc_y_df)
print(atributos_acc_y)

combined_acc_z_df = pd.concat([motion_acc_z_df, idle_acc_z_df], axis=1)
combined_acc_z_df = combined_acc_z_df[['motion1', 'motion2', 'motion3', 'idle1', 'idle2', 'idle3']]

atributos_acc_z = extrair_atributos(combined_acc_z_df)
print(atributos_acc_z)

combined_gy_x_df = pd.concat([motion_gy_x_df, idle_gy_x_df], axis=1)
combined_gy_x_df = combined_gy_x_df[['motion1', 'motion2', 'motion3', 'idle1', 'idle2', 'idle3']]

atributos_gy_x = extrair_atributos(combined_gy_x_df)
print(atributos_gy_x)

combined_gy_y_df = pd.concat([motion_gy_y_df, idle_gy_y_df], axis=1)
combined_gy_y_df = combined_gy_y_df[['motion1', 'motion2', 'motion3', 'idle1', 'idle2', 'idle3']]

atributos_gy_y = extrair_atributos(combined_gy_y_df)
print(atributos_gy_y)

combined_gy_z_df = pd.concat([motion_gy_z_df, idle_gy_z_df], axis=1)
combined_gy_z_df = combined_gy_z_df[['motion1', 'motion2', 'motion3', 'idle1', 'idle2', 'idle3']]

atributos_gy_z = extrair_atributos(combined_gy_z_df)
print(atributos_gy_z)

plot_signals(combined_acc_x_df, fs=80.0, max_cols=6, nome='acc_x')
plot_signals(combined_acc_y_df, fs=80.0, max_cols=6, nome='acc_y')
plot_signals(combined_acc_z_df, fs=80.0, max_cols=6, nome='acc_z')

plot_signals(combined_gy_x_df, fs=80.0, max_cols=6, nome='gy_x')
plot_signals(combined_gy_y_df, fs=80.0, max_cols=6, nome='gy_y')
plot_signals(combined_gy_z_df, fs=80.0, max_cols=6, nome='gy_z')

plot_fft(combined_acc_x_df, fs=80.0, max_cols=6, nome='acc_x')
plot_fft(combined_acc_y_df, fs=80.0, max_cols=6, nome='acc_y')
plot_fft(combined_acc_z_df, fs=80.0, max_cols=6, nome='acc_z')

plot_fft(combined_gy_x_df, fs=80.0, max_cols=6, nome='gy_x')
plot_fft(combined_gy_y_df, fs=80.0, max_cols=6, nome='gy_y')
plot_fft(combined_gy_z_df, fs=80.0, max_cols=6, nome='gy_z')