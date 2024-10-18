import scipy.io as scio
import os
import numpy as np
import time
import mne

def preprocess_SEED_Time(session):
    path = f'E:/datasets/SEED/session{session}'


    for info in os.listdir(path):
        info_ = os.path.join(path, info)
        data = scio.loadmat(info_)
        subject_number = info.split('_')[0]

        label_info = (2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0)
        window_size = 128
        feature = {}
        label = {}
        # 15个trail处理

        for i in range(1, 16):
            key = next((k for k in data.keys() if k.endswith(f'eeg{i}')), None)

            if key is None:
                print(f"Warning: No key found for trial {i} in subject {subject_number}")
                continue

            value = data[key]
            sfreq = 200
            n_channels = 62
            ch_names = [f'EEG {i + 1:03d}' for i in range(n_channels)]
            ch_types = ['eeg'] * n_channels
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

            # 创建 MNE 的 RawArray 对象
            raw = mne.io.RawArray(value, info)

            # 对数据进行降采样到 128 Hz
            target_sfreq = 128
            raw.resample(sfreq=target_sfreq)

            # 对数据进行 4-47 Hz 带通滤波
            raw.filter(l_freq=4., h_freq=47.)

            # 获取数据并转换为 NumPy ndarray
            feature_trial = raw.get_data()

            # 将数据 reshape 为 (n_channels, window_size, n_windows)
            n_samples = feature_trial.shape[1]
            n_windows = n_samples // window_size  # 计算窗口数量

            # Reshape 成 (n_channels, n_windows, window_size)
            feature_trial = feature_trial[:, :n_windows * window_size]  # 去除多余数据
            feature_trial = feature_trial.reshape(n_channels, n_windows, window_size)

            # Permute (n_windows, n_channels, window_size)
            feature_trial = feature_trial.transpose(1, 2, 0)

            # 存储处理后的特征
            feature[f"trial{i}"] = feature_trial

            # 创建与窗口匹配的标签

            label[f"trial{i}"] = np.repeat(label_info[i - 1], feature_trial.shape[0])

            print(f"Trial {i} feature shape: {feature_trial.shape}")
            print(f"Trial {i} label shape: {label[f'trial{i}'].shape}")

        save_data = {'feature': feature, 'label': label}
        file_name = f"subject{subject_number}.mat"
        file_path = os.path.join(f'E:/datasets/SEED_Time_Preprocessed_128/Session{session}', file_name)
        scio.savemat(file_path, save_data)


if __name__ == '__main__':
    for session in range(1, 4):
        preprocess_SEED_Time(session)
