from modulos import *
import pandas as pd

dataset_raw = pd.read_csv(r"./Dataset/full_dataset.csv")

idle_samples  = dataset_raw[dataset_raw['label'] == 'idle'].iloc[:500, 1:]
motion_samples = dataset_raw[dataset_raw['label'] == 'motion'].iloc[:500, 1:]

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
combined_acc_x_df = combined_acc_x_df[['motion3', 'motion5', 'idle0', 'idle7']]

combined_acc_y_df = pd.concat([motion_acc_y_df, idle_acc_y_df], axis=1)
combined_acc_y_df = combined_acc_y_df[['motion3', 'motion5', 'idle0', 'idle7']]

combined_acc_z_df = pd.concat([motion_acc_z_df, idle_acc_z_df], axis=1)
combined_acc_z_df = combined_acc_z_df[['motion3', 'motion5', 'idle0', 'idle7']]


plot_signals(combined_acc_x_df, fs=80.0, max_cols=6, nome='acc_x')
plot_signals(combined_acc_y_df, fs=80.0, max_cols=6, nome='acc_y')
plot_signals(combined_acc_z_df, fs=80.0, max_cols=6, nome='acc_z')