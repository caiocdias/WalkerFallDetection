from modulos import *
import pandas as pd

dataset_raw = pd.read_csv(r".\Dataset\full_dataset.csv")

idle_samples  = dataset_raw[dataset_raw['label'] == 'idle'].iloc[:500, 1:]
motion_samples = dataset_raw[dataset_raw['label'] == 'motion'].iloc[:500, 1:]

idle_matrix   = idle_samples.T
motion_matrix = motion_samples.T

idle_matrix.columns   = [f"idle" for i in idle_matrix.columns]
motion_matrix.columns = [f"motion" for i in motion_matrix.columns]

prefixes = ['acc_x', 'acc_y', 'acc_z', 'gy_x', 'gy_y', 'gy_z']

idle_acc_x_df = idle_matrix.loc[idle_matrix.index.str.startswith('acc_x'), :]
idle_acc_y_df = idle_matrix.loc[idle_matrix.index.str.startswith('acc_y'), :]
idle_acc_z_df = idle_matrix.loc[idle_matrix.index.str.startswith('acc_z'), :]
idle_gy_x_df  = idle_matrix.loc[idle_matrix.index.str.startswith('gy_x'), :]
idle_gy_y_df  = idle_matrix.loc[idle_matrix.index.str.startswith('gy_y'), :]
idle_gy_z_df  = idle_matrix.loc[idle_matrix.index.str.startswith('gy_z'), :]

#acc_x_df.to_excel(r".\Dataset\acc_x_samples.xlsx")
#acc_y_df.to_excel(r".\Dataset\acc_y_samples.xlsx")
#acc_z_df.to_excel(r".\Dataset\acc_z_samples.xlsx")
#gy_x_df.to_excel(r".\Dataset\gy_x_samples.xlsx")
#gy_y_df.to_excel(r".\Dataset\gy_y_samples.xlsx")
#gy_z_df.to_excel(r".\Dataset\gy_z_samples.xlsx")
plot_signals(idle_acc_x_df, fs=80.0, max_cols=5)