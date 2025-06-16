import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import os
import seaborn as sns


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

def plot_attributes(df, nome: str = None, figsize=(10, 6), include: list[str] = None):
    include = include or []

    sensors = df.index.astype(str)
    x = np.arange(len(sensors))

    plt.figure(figsize=figsize)
    for atributo in df.columns:
        if atributo not in include:
            continue
        plt.scatter(x, df[atributo], s=50, label=atributo)

    plt.xticks([])

    plt.ylabel('Valor do Atributo')
    titulo = 'Atributos'
    if nome:
        titulo += f' – {nome}'
    plt.title(titulo)
    plt.legend(loc='best')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

def extrair_atributos(df: pd.DataFrame) -> pd.DataFrame:
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
    dataset_raw = pd.read_csv(caminho_arquivo)
    idle = dataset_raw[dataset_raw['label'] == 'idle'].iloc[:amostra_tamanho, 1:].reset_index(drop=True)
    motion = dataset_raw[dataset_raw['label'] == 'motion'].iloc[:amostra_tamanho, 1:].reset_index(drop=True)

    return preparar_matrizes(idle, motion)

def preparar_matrizes(idle_dados, motion_dados):
    idle_matrix = idle_dados.T
    motion_matrix = motion_dados.T

    idle_matrix.columns = [f"i{i}" for i in idle_matrix.columns]
    motion_matrix.columns = [f"m{j}" for j in motion_matrix.columns]

    return idle_matrix, motion_matrix

def processar_sinal(idle_matrix, motion_matrix, prefixo, colunas_selecionadas, freq_amostragem, flag_plotar_sinais, flag_plotar_espectro):
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

def correlacao_geral_fdr(flag_plotar_correlacao, amostra_tamanho):
    result_acc_x = pd.read_excel("./export/result_acc_x.xlsx", sheet_name="result", index_col=0)
    result_acc_y = pd.read_excel("./export/result_acc_y.xlsx", sheet_name="result", index_col=0)
    result_acc_z = pd.read_excel("./export/result_acc_z.xlsx", sheet_name="result", index_col=0)
    result_gy_x = pd.read_excel("./export/result_gy_x.xlsx", sheet_name="result", index_col=0)
    result_gy_y = pd.read_excel("./export/result_gy_y.xlsx", sheet_name="result", index_col=0)
    result_gy_z = pd.read_excel("./export/result_gy_z.xlsx", sheet_name="result", index_col=0)

    result_acc_x.columns = [f"{column}_accx" for column in result_acc_x.columns]
    result_acc_y.columns = [f"{column}_accy" for column in result_acc_y.columns]
    result_acc_z.columns = [f"{column}_accz" for column in result_acc_z.columns]
    result_gy_x.columns = [f"{column}_gyx" for column in result_gy_x.columns]
    result_gy_y.columns = [f"{column}_gyy" for column in result_gy_y.columns]
    result_gy_z.columns = [f"{column}_gyz" for column in result_gy_z.columns]

    result_general = pd.concat([result_acc_x, result_acc_y, result_acc_z, result_gy_x, result_gy_y, result_gy_z], axis=1)

    valores_target = [-1] * amostra_tamanho + [1] * amostra_tamanho
    result_general['target'] = valores_target

    result_general_corr = result_general.corr()

    result_corr_tratado = remover_atributos_correlacionados(result_general_corr, threshold=0.5)

    with pd.ExcelWriter(r"./export/result_general.xlsx", engine="openpyxl") as writer:
        result_general.to_excel(writer, sheet_name="result")
        result_general_corr.to_excel(writer, sheet_name="result_corr")
        result_corr_tratado.to_excel(writer, sheet_name="result_corr_tratado")

    fdr_df = pd.DataFrame(columns=['Atributo', 'Valor'])

    for column in result_general_corr.columns:
        motion_results = result_general[result_general.index.str.startswith('m')]
        idle_results = result_general[result_general.index.str.startswith('i')]

        avg_motion = motion_results[column].mean()
        avg_idle = idle_results[column].mean()

        std_deviation_motion = motion_results[column].std()
        std_deviation_idle = idle_results[column].std()

        fdr = pow(avg_motion - avg_idle, 2) / (pow(std_deviation_motion, 2) + pow(std_deviation_idle, 2))
        fdr_df.loc[len(fdr_df)] = {'Atributo': str(column).strip("'"), 'Valor': float(fdr)}

    fdr_df.to_excel(r"./export/fdr.xlsx", index=False)

    if flag_plotar_correlacao == 1:
        plot_correlation_heatmap(result_general_corr, title='Matriz de Correlação Geral dos Atributos de Sensores')
        plot_correlation_heatmap(result_corr_tratado, title='Matriz de Correlação Tratada dos Atributos de Sensores')

    return result_general, fdr_df

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, title: str = 'Matriz de Correlação'):

    plt.figure(figsize=(24, 20))

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    cmap = 'coolwarm'

    # Plota o heatmap com seaborn
    sns.heatmap(corr_matrix,
                mask=mask,
                cmap=cmap,
                vmax=1.0,
                vmin=-1.0,
                center=0,
                linewidths=.5, # Adiciona linhas finas entre as células
                cbar_kws={"shrink": .75}) # Ajusta o tamanho da barra de cores

    plt.title(title, fontsize=20)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout(pad=2.0) # Ajusta o layout para evitar sobreposição de texto
    plt.show()

def remover_atributos_correlacionados(corr_matrix, threshold):
    corr_matrix_copy = corr_matrix.copy()

    colunas_para_remover = set()

    for i in range(len(corr_matrix_copy.columns)):
        for j in range(i):
            if abs(corr_matrix_copy.iloc[i, j]) > threshold:
                nome_coluna = corr_matrix_copy.columns[i]
                colunas_para_remover.add(nome_coluna)

    nova_corr_matrix = corr_matrix_copy.drop(list(colunas_para_remover), axis=1)
    nova_corr_matrix = nova_corr_matrix.drop(list(colunas_para_remover), axis=0)

    return nova_corr_matrix

if __name__ == "__main__":
    amostra_tamanho = int(input("Digite o tamanho da amostra: "))
    flag_plotar_sinais = int(input("Deseja plotar os sinais? (1 - Sim, 0 - Não): "))
    flag_plotar_espectro = int(input("Deseja plotar os espectros de frequência? (1 - Sim, 0 - Não): "))
    flag_plotar_atributos = int(input("Deseja plotar os dados dos atributos? (1 - Sim, 0 - Não): "))
    flag_plotar_correlacao = int(input("Deseja plotar a matriz de correlação? (1 - Sim, 0 - Não): "))

    colunas_selecionadas = [f'm{i}' for i in range(0, amostra_tamanho)] + [f'i{i}' for i in range(0, amostra_tamanho)]
    freq_amostragem = 80.0
    prefixos_sensores = {
        'acc': ['acc_x', 'acc_y', 'acc_z'],
        'gy': ['gy_x', 'gy_y', 'gy_z']
    }

    idle_matrix, motion_matrix = carregar_dados("./Dataset/full_dataset.csv", amostra_tamanho)

    for tipo_sensor, prefixos in prefixos_sensores.items():
        for prefixo in prefixos:
            atributos = processar_sinal(
                idle_matrix, motion_matrix,
                prefixo, colunas_selecionadas,
                freq_amostragem, flag_plotar_sinais, flag_plotar_espectro
            )
            result = pd.DataFrame(atributos)

            if flag_plotar_atributos == 1:
                plot_attributes(result, nome=f"{prefixo}", include=['desvio_padrao'])
                plot_attributes(result, nome=f"{prefixo}", include=['rms'])
                plot_attributes(result, nome=f"{prefixo}", include=['energia'])
                plot_attributes(result, nome=f"{prefixo}", include=['pico_min'])
                plot_attributes(result, nome=f"{prefixo}", include=['pico_max'])
                plot_attributes(result, nome=f"{prefixo}", include=['skewness'])
                plot_attributes(result, nome=f"{prefixo}", include=['kurtosis'])
                plot_attributes(result, nome=f"{prefixo}", include=['media'])
            result_corr = result.corr()

            if not os.path.exists("./export"):
                os.makedirs("./export", exist_ok=True)

            arquivo = f"./export/result_{prefixo}.xlsx"
            with pd.ExcelWriter(arquivo, engine="openpyxl") as writer:
                result.to_excel(writer, sheet_name="result")
                result_corr.to_excel(writer, sheet_name="result_corr")

    result_geral, fdr_df = correlacao_geral_fdr(flag_plotar_correlacao, amostra_tamanho)
    print("Processamento concluído. Arquivos exportados com sucesso para ./export/")
