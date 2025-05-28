import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, pearsonr
import matplotlib.pyplot as plt
import os

def plot_signals(df: pd.DataFrame, fs: float = 80.0, nome: str = None):
    df_numeric = df.select_dtypes(include=[np.number])

    cols_to_plot = df_numeric.columns
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

def plot_fft(df: pd.DataFrame, fs: float = 80.0, nome: str = None):
    df_numeric = df.select_dtypes(include=[np.number])

    cols_to_plot = df_numeric.columns
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

def plot_attributes(df, nome: str = None, figsize=(10, 6), exclude: list[str] = None):
    exclude = exclude or []

    sensors = df.index.astype(str)
    x = np.arange(len(sensors))

    plt.figure(figsize=figsize)
    for atributo in df.columns:
        if atributo in exclude:
            continue
        plt.scatter(x, df[atributo], s=50, label=atributo)

    # Configurações dos eixos e legendas
    plt.xticks(x, sensors, rotation=45, ha='right')
    plt.xlabel('Sensor')
    plt.ylabel('Valor do Atributo')
    titulo = 'Atributos por Sensor'
    if nome:
        titulo += f' – {nome}'
    plt.title(titulo)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
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

def carregar_dados(caminho_arquivo, amostra_tamanho):
    """Carrega e prepara os dados iniciais do dataset."""
    dataset_raw = pd.read_csv(caminho_arquivo)
    idle = dataset_raw[dataset_raw['label'] == 'idle'].iloc[:amostra_tamanho, 1:].reset_index(drop=True)
    motion = dataset_raw[dataset_raw['label'] == 'motion'].iloc[:amostra_tamanho, 1:].reset_index(drop=True)

    return preparar_matrizes(idle, motion)

def preparar_matrizes(idle_dados, motion_dados):
    """Prepara as matrizes de dados idle e motion."""
    idle_matrix = idle_dados.T
    motion_matrix = motion_dados.T

    idle_matrix.columns = [f"idle{i}" for i in idle_matrix.columns]
    motion_matrix.columns = [f"motion{j}" for j in motion_matrix.columns]

    return idle_matrix, motion_matrix

def processar_sinal(idle_matrix, motion_matrix, prefixo, colunas_selecionadas, freq_amostragem, flag_plotar_sinais, flag_plotar_espectro):
    """Processa um tipo específico de sinal (acc ou gy)."""
    motion_df = motion_matrix.loc[motion_matrix.index.str.startswith(prefixo), :]
    idle_df = idle_matrix.loc[idle_matrix.index.str.startswith(prefixo), :]

    combined_df = pd.concat([motion_df, idle_df], axis=1)
    combined_df = combined_df[colunas_selecionadas]

    atributos = extrair_atributos(combined_df)


    if flag_plotar_sinais == 1:
        plot_signals(combined_df, fs=freq_amostragem, nome=prefixo)
    if flag_plotar_espectro == 1:
        plot_fft(combined_df, fs=freq_amostragem, nome=prefixo)

    return atributos

if __name__ == "__main__":
    try:
        amostra_tamanho = int(input("Digite o tamanho da amostra: "))
        flag_plotar_sinais = int(input("Deseja plotar os sinais? (1 - Sim, 0 - Não): "))
        flag_plotar_espectro = int(input("Deseja plotar os espectros de frequência? (1 - Sim, 0 - Não): "))
        flag_plotar_atributos = int(input("Deseja plotar os dados dos atributos? (1 - Sim, 0 - Não): "))

        colunas_selecionadas = [f'motion{i}' for i in range(0, amostra_tamanho)] + [f'idle{i}' for i in range(0, amostra_tamanho)]
        freq_amostragem = 80.0
        prefixos_sensores = {
            'acc': ['acc_x', 'acc_y', 'acc_z'],
            'gy': ['gy_x', 'gy_y', 'gy_z']
        }

        # Carregamento dos dados
        idle_matrix, motion_matrix = carregar_dados("./Dataset/full_dataset.csv", amostra_tamanho)

        # Processamento para cada tipo de sensor
        for tipo_sensor, prefixos in prefixos_sensores.items():
            for prefixo in prefixos:
                # Processa sinais e monta DataFrame
                atributos = processar_sinal(
                    idle_matrix, motion_matrix,
                    prefixo, colunas_selecionadas,
                    freq_amostragem, flag_plotar_sinais, flag_plotar_espectro
                )
                result = pd.DataFrame(atributos)

                if flag_plotar_atributos == 1:
                    plot_attributes(result, nome=f"{prefixo}", exclude=["energia"])
                result_corr = result.corr()

                # Exporta para Excel com duas abas
                os.makedirs("./export", exist_ok=True)
                arquivo = f"./export/result_{prefixo}.xlsx"
                with pd.ExcelWriter(arquivo, engine="openpyxl") as writer:
                    result.to_excel(writer, sheet_name="result")
                    result_corr.to_excel(writer, sheet_name="result_corr")
        print("Processamento concluído. Arquivos exportados com sucesso para ./export/")

    except Exception as e:
        print(f"Erro ao processar os dados: {e}")