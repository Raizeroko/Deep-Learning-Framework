import scipy.io as scio
import os
import numpy as np
import time
import mne
from scipy.signal import cheby1, filtfilt


def preprocess_SEED_Time_cheby(session):
    path = f'E:/datasets/UnPreprocessed/SEED/session{session}'

    for info in os.listdir(path):
        info_ = os.path.join(path, info)
        data = scio.loadmat(info_)
        subject_number = info.split('_')[0]

        label_info = (2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0)
        window_size = 600
        feature = {}
        label = {}
        # 15个trail处理

        for trial in range(1, 16):
            key = next((k for k in data.keys() if k.endswith(f'eeg{trial}')), None)

            if key is None:
                print(f"Warning: No key found for trial {trial} in subject {subject_number}")
                continue

            feature_trial = data[key]
            sfreq = 200
            n_channels = 62

            # Chebyshev filter [4, 47Hz]
            fs = 128
            lowcut = 4  # 带通滤波的低频限
            highcut = 45  # 带通滤波的高频限
            order = 6  # Chebyshev滤波器的阶数
            b, a = cheby1(order, rp=0.5, Wn=[lowcut, highcut], btype='band', fs=sfreq)
            filtered_data = filtfilt(b, a, feature_trial, axis=1)

            # z-score standardization




            # mne band-pass filtering + downsample
            # ch_names = [f'EEG {i + 1:03d}' for i in range(n_channels)]
            # ch_types = ['eeg'] * n_channels
            # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

            # 创建 MNE 的 RawArray 对象
            # raw = mne.io.RawArray(value, info)

            # # 对数据进行降采样到 128 Hz
            # target_sfreq = 128
            # raw.resample(sfreq=target_sfreq)
            #
            # # 对数据进行 4-47 Hz 带通滤波
            # raw.filter(l_freq=4., h_freq=47.)
            #
            # # 获取数据并转换为 NumPy ndarray
            # feature_trial = raw.get_data()

            # 去除基线信号
            # baseline = feature_trial[:, :sfreq]
            # feature_trial = feature_trial[:, sfreq:]
            # for i in range(int(feature_trial.shape[1] / sfreq)):
            #     feature_trial[:, i * sfreq:(i + 1) * sfreq] = feature_trial[:, i * sfreq:(i + 1) * sfreq] - baseline

            # 将数据 reshape 为 (n_channels, window_size, n_windows)
            n_samples = feature_trial.shape[1]
            n_windows = n_samples // window_size  # 计算窗口数量

            # Reshape 成 (n_channels, n_windows, window_size)
            feature_trial = feature_trial[:, :n_windows * window_size]  # 去除多余数据
            feature_trial = feature_trial.reshape(n_channels, n_windows, window_size)

            # Permute (n_windows, n_channels, window_size)
            feature_trial = feature_trial.transpose(1, 2, 0)
            # test1 = feature_trial[1, :, 1,].reshape(-1)

            # 进行归一化处理
            # mean = np.mean(feature_trial, axis=0, keepdims=True)  # 计算每个通道的均值
            # std = np.std(feature_trial, axis=0, keepdims=True)  # 计算每个通道的标准差
            # # 避免除以 0 的情况
            # std[std == 0] = 1
            # feature_trial_normalized = (feature_trial - mean) / std

            # test2 = feature_trial_normalized[1, :, 1].reshape(-1)
            # 存储处理后的特征
            feature[f"trial{trial}"] = feature_trial

            # 创建与窗口匹配的标签

            label[f"trial{trial}"] = np.repeat(label_info[trial - 1], feature_trial.shape[0])

            print(f"Trial {trial} feature shape: {feature_trial.shape}")
            print(f"Trial {trial} label shape: {label[f'trial{trial}'].shape}")

        save_data = {'feature': feature, 'label': label}
        file_name = f"subject{subject_number}.mat"
        file_path = os.path.join(f'E:/datasets/SEED_Time_Preprocessed_{window_size}/Session{session}', file_name)
        # scio.savemat(file_path, save_data)


def preprocess_SEED_Time(session):
    path = f'E:/datasets/UnPreprocessed/SEED/session{session}'

    for info in os.listdir(path):
        info_ = os.path.join(path, info)
        data = scio.loadmat(info_)
        subject_number = info.split('_')[0]

        label_info = (2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0)
        window_size = 600
        feature = {}
        label = {}
        # 15个trail处理

        for trial in range(1, 16):
            key = next((k for k in data.keys() if k.endswith(f'eeg{trial}')), None)

            if key is None:
                print(f"Warning: No key found for trial {trial} in subject {subject_number}")
                continue

            feature_trial = data[key]
            sfreq = 200
            n_channels = 62
            baseline = feature_trial[:, :sfreq]
            feature_trial = feature_trial[:, sfreq:]
            # ch_names = [f'EEG {i + 1:03d}' for i in range(n_channels)]
            # ch_types = ['eeg'] * n_channels
            # info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

            # 创建 MNE 的 RawArray 对象
            # raw = mne.io.RawArray(value, info)

            # # 对数据进行降采样到 128 Hz
            # target_sfreq = 128
            # raw.resample(sfreq=target_sfreq)
            #
            # # 对数据进行 4-47 Hz 带通滤波
            # raw.filter(l_freq=4., h_freq=47.)
            #
            # # 获取数据并转换为 NumPy ndarray
            # feature_trial = raw.get_data()

            # 去除基线信号
            for i in range(int(feature_trial.shape[1] / sfreq)):
                feature_trial[:, i * sfreq:(i + 1) * sfreq] = feature_trial[:, i * sfreq:(i + 1) * sfreq] - baseline

            # 将数据 reshape 为 (n_channels, window_size, n_windows)
            n_samples = feature_trial.shape[1]
            n_windows = n_samples // window_size  # 计算窗口数量

            # Reshape 成 (n_channels, n_windows, window_size)
            feature_trial = feature_trial[:, :n_windows * window_size]  # 去除多余数据
            feature_trial = feature_trial.reshape(n_channels, n_windows, window_size)

            # Permute (n_windows, n_channels, window_size)
            feature_trial = feature_trial.transpose(1, 2, 0)
            # test1 = feature_trial[1, :, 1,].reshape(-1)

            # 进行归一化处理
            # mean = np.mean(feature_trial, axis=0, keepdims=True)  # 计算每个通道的均值
            # std = np.std(feature_trial, axis=0, keepdims=True)  # 计算每个通道的标准差
            # # 避免除以 0 的情况
            # std[std == 0] = 1
            # feature_trial_normalized = (feature_trial - mean) / std


            # test2 = feature_trial_normalized[1, :, 1].reshape(-1)
            # 存储处理后的特征
            feature[f"trial{trial}"] = feature_trial

            # 创建与窗口匹配的标签

            label[f"trial{trial}"] = np.repeat(label_info[trial - 1], feature_trial.shape[0])

            print(f"Trial {trial} feature shape: {feature_trial.shape}")
            print(f"Trial {trial} label shape: {label[f'trial{trial}'].shape}")

        save_data = {'feature': feature, 'label': label}
        file_name = f"subject{subject_number}.mat"
        file_path = os.path.join(f'E:/datasets/SEED_Time_Preprocessed_{window_size}/Session{session}', file_name)
        scio.savemat(file_path, save_data)


if __name__ == '__main__':
    for session in range(1, 4):
        # preprocess_SEED_Time(session)
        preprocess_SEED_Time_cheby(session)
