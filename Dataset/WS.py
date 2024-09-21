import os
import torch
import torch.utils.data as Data
import scipy.io as scio

def SEED_Dataset_WSSSCV(input_dir, session):
    data_dir = os.path.join(input_dir, f'Session{session}')

    # 初始化空列表，用于存储特征和标签
    source_feature_list = []
    source_label_list = []
    target_feature_list = []
    target_label_list = []

    for info in os.listdir(data_dir):
        file_path = os.path.join(data_dir, info)
        data = scio.loadmat(file_path)
        feature_trial = data['feature']
        label_trial = data['label']

        source_feature = None
        source_label = None
        target_feature = None
        target_label = None

        # 提取特征和标签
        for i in range(1, 16):
            if i < 10:
                if source_feature is None:
                    source_feature = torch.tensor(feature_trial[f'trial{i}'][0][0])
                    source_label = torch.tensor(label_trial[f'trial{i}'][0][0])
                else:
                    source_feature = torch.cat((source_feature, torch.tensor(feature_trial[f'trial{i}'][0][0])), dim=1)
                    source_label = torch.cat((source_label, torch.tensor(label_trial[f'trial{i}'][0][0])), dim=1)
            else:
                if target_feature is None:
                    target_feature = torch.tensor(feature_trial[f'trial{i}'][0][0])
                    target_label = torch.tensor(label_trial[f'trial{i}'][0][0])
                else:
                    target_feature = torch.cat((target_feature, torch.tensor(feature_trial[f'trial{i}'][0][0])), dim=1)
                    target_label = torch.cat((target_label, torch.tensor(label_trial[f'trial{i}'][0][0])), dim=1)

        # 添加到相应的列表中
        source_feature_list.append(source_feature.permute(1, 0, 2))
        source_label_list.append(source_label.reshape(-1) + 1)
        target_feature_list.append(target_feature.permute(1, 0, 2))
        target_label_list.append(target_label.reshape(-1) + 1)

    # 合并为Tensor
    source_feature = torch.stack(source_feature_list).reshape(-1, 62, 5).float()
    source_label = torch.stack(source_label_list).reshape(-1).long()
    target_feature = torch.stack(target_feature_list).reshape(-1, 62, 5).float()
    target_label = torch.stack(target_label_list).reshape(-1).long()

    # 构建源域和目标域数据集
    source_set = {'feature': source_feature, 'label': source_label}
    target_set = {'feature': target_feature, 'label': target_label}

    #转为tensor数据集
    train_dataset = Data.TensorDataset(source_set['feature'],
                                       source_set['label'])
    test_dataset = Data.TensorDataset(target_set['feature'],
                                      target_set['label'])

    return train_dataset, test_dataset

def SEED_Dataset_WS(input_dir, session, target_id):
    data_dir = os.path.join(input_dir, f'Session{session}')

    file_path = os.path.join(data_dir, f'subject{target_id}.mat')
    data = scio.loadmat(file_path)
    feature_trial = data['feature']
    label_trial = data['label']

    source_feature = None
    source_label = None
    target_feature = None
    target_label = None

    # 提取特征和标签
    for i in range(1, 16):
        if i < 10:
            if source_feature is None:
                source_feature = torch.tensor(feature_trial[f'trial{i}'][0][0])
                source_label = torch.tensor(label_trial[f'trial{i}'][0][0])
            else:
                source_feature = torch.cat((source_feature, torch.tensor(feature_trial[f'trial{i}'][0][0])), dim=1)
                source_label = torch.cat((source_label, torch.tensor(label_trial[f'trial{i}'][0][0])), dim=1)
        else:
            if target_feature is None:
                target_feature = torch.tensor(feature_trial[f'trial{i}'][0][0])
                target_label = torch.tensor(label_trial[f'trial{i}'][0][0])
            else:
                target_feature = torch.cat((target_feature, torch.tensor(feature_trial[f'trial{i}'][0][0])), dim=1)
                target_label = torch.cat((target_label, torch.tensor(label_trial[f'trial{i}'][0][0])), dim=1)

    source_feature = source_feature.permute(1, 0, 2).float()
    target_feature = target_feature.permute(1, 0, 2).float()
    source_label = (source_label.reshape(-1) + 1).long()
    target_label = (target_label.reshape(-1) + 1).long()

    # 构建源域和目标域数据集
    source_set = {'feature': source_feature, 'label': source_label}
    target_set = {'feature': target_feature, 'label': target_label}

    # 转为tensor数据集
    train_dataset = Data.TensorDataset(source_set['feature'],
                                       source_set['label'])
    test_dataset = Data.TensorDataset(target_set['feature'],
                                      target_set['label'])

    return train_dataset, test_dataset

def DEAP_Dataset_WS(input_dir, session, target_id):
    if session == 1:
        data_dir = os.path.join(input_dir, 'Arousal')
    elif session == 2:
        data_dir = os.path.join(input_dir, 'Valence')

    file_path = os.path.join(data_dir, f'subject{target_id}.mat')
    data = scio.loadmat(file_path)
    feature_trial = data['feature']
    label_trial = data['label']

    source_feature = None
    source_label = None
    target_feature = None
    target_label = None

    # 提取特征和标签
    for i in range(1, 41):
        if i < 37:
            if source_feature is None:
                source_feature = torch.tensor(feature_trial[f'trial{i}'][0][0])
                source_label = torch.tensor(label_trial[f'trial{i}'][0][0])
            else:
                source_feature = torch.cat((source_feature, torch.tensor(feature_trial[f'trial{i}'][0][0])), dim=0)
                source_label = torch.cat((source_label, torch.tensor(label_trial[f'trial{i}'][0][0])), dim=0)
        else:
            if target_feature is None:
                target_feature = torch.tensor(feature_trial[f'trial{i}'][0][0])
                target_label = torch.tensor(label_trial[f'trial{i}'][0][0])
            else:
                target_feature = torch.cat((target_feature, torch.tensor(feature_trial[f'trial{i}'][0][0])), dim=0)
                target_label = torch.cat((target_label, torch.tensor(label_trial[f'trial{i}'][0][0])), dim=0)

    source_feature = source_feature.permute(0, 2, 1).float()
    target_feature = target_feature.permute(0, 2, 1).float()
    source_label = source_label.reshape(-1).long()
    target_label = target_label.reshape(-1).long()

    # 构建源域和目标域数据集
    source_set = {'feature': source_feature, 'label': source_label}
    target_set = {'feature': target_feature, 'label': target_label}

    # 转为tensor数据集
    train_dataset = Data.TensorDataset(source_set['feature'],
                                       source_set['label'])
    test_dataset = Data.TensorDataset(target_set['feature'],
                                      target_set['label'])

    return train_dataset, test_dataset


def SEED_Dataset_Window_WS(window_size, input_dir, session, target_id):
    data_dir = os.path.join(input_dir, f'Session{session}')

    file_path = os.path.join(data_dir, f'subject{target_id}.mat')
    data = scio.loadmat(file_path)
    feature_trial = data['feature']
    label_trial = data['label']

    source_feature = None
    source_label = None
    target_feature = None
    target_label = None

    for i in range(1, 16):
        trial_feature = torch.tensor(feature_trial[f'trial{i}'][0][0])
        trial_label = torch.tensor(label_trial[f'trial{i}'][0][0]).reshape(-1)

        # 提取特征和标签
        if i < 10:
            if source_feature is None:
                source_feature, source_label = apply_sliding_window(trial_feature, trial_label, window_size)
            else:
                source_feature_windowed, source_label_windowed = apply_sliding_window(trial_feature, trial_label, window_size)
                source_feature = torch.cat((source_feature, source_feature_windowed), dim=1)
                source_label = torch.cat((source_label, source_label_windowed), dim=0)
        else:
            if target_feature is None:
                target_feature, target_label = apply_sliding_window(trial_feature, trial_label, window_size)
            else:
                target_feature_windowed, target_label_windowed = apply_sliding_window(trial_feature, trial_label, window_size)
                target_feature = torch.cat((target_feature, target_feature_windowed), dim=1)
                target_label = torch.cat((target_label, target_label_windowed), dim=0)

    source_feature = source_feature.permute(1, 2, 0, 3).float()
    target_feature = target_feature.permute(1, 2, 0, 3).float()
    source_label = (source_label.reshape(-1) + 1).long()
    target_label = (target_label.reshape(-1) + 1).long()

    # 构建源域和目标域数据集
    source_set = {'feature': source_feature, 'label': source_label}
    target_set = {'feature': target_feature, 'label': target_label}

    # 转为tensor数据集
    train_dataset = Data.TensorDataset(source_set['feature'], source_set['label'])
    test_dataset = Data.TensorDataset(target_set['feature'], target_set['label'])

    return train_dataset, test_dataset

def apply_sliding_window(features, label, window_size):
    num_electrodes, num_samples, feature_dim = features.shape
    windows = []
    for i in range(num_samples - window_size + 1):
        window = features[:, i:i + window_size, :]
        windows.append(window)

    feature = torch.stack(windows, dim=1)
    label = label[:feature.shape[1]]
    return feature, label


if __name__ == '__main__':
    input_dir = "E:/datasets/SEED_Preprocessed"
    session = 1
    k_fold = 10
    for i in range(1, 2):
        for fold in range(10):
            SEED_Dataset_Window_WS(5, input_dir, session, i)