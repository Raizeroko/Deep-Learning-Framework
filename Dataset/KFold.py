import os
import torch
import torch.utils.data as Data
import scipy.io as scio

def SEED_Dataset_KFold_Sample(input_dir, session, target_id, k, fold,  shuffle=True):
    return

def DEAP_Dataset_KFold_Sample(input_dir, session, target_id, k, fold,  shuffle=True):
    global shuffle_indices
    if session == 1:
        data_dir = os.path.join(input_dir, 'Arousal')
    elif session == 2:
        data_dir = os.path.join(input_dir, 'Valence')

    file_path = os.path.join(data_dir, f'subject{target_id}.mat')
    data = scio.loadmat(file_path)
    feature_trial = data['feature']
    label_trial = data['label']

    feature = None
    label = None

    for i in range(0, 40):
        if feature is None:
            feature = torch.tensor(feature_trial[f'trial{i + 1}'][0][0])
            label = torch.tensor(label_trial[f'trial{i + 1}'][0][0])
        else:
            feature = torch.cat((feature, torch.tensor(feature_trial[f'trial{i + 1}'][0][0])), dim=0)
            label = torch.cat((label, torch.tensor(label_trial[f'trial{i + 1}'][0][0])), dim=0)

    feature = feature.permute(0, 2, 1).float()
    label = label.reshape(-1).long()

    if shuffle and shuffle_indices is None:
        shuffle_indices = torch.randperm(feature.size(0))

    if shuffle:
        feature = feature[shuffle_indices]
        label = label[shuffle_indices]

    print(shuffle_indices)
    # 计算每个折的大小
    fold_size = feature.size(0) // k

    # 获取验证集的索引
    val_start = fold * fold_size
    val_end = val_start + fold_size

    # 将数据分成训练集和验证集
    target_feature = feature[val_start:val_end]
    target_label = label[val_start:val_end]

    source_feature = torch.cat((feature[:val_start], feature[val_end:]), dim=0)
    source_label = torch.cat((label[:val_start], label[val_end:]), dim=0)

    # 构建源域和目标域数据集
    source_set = {'feature': source_feature, 'label': source_label}
    target_set = {'feature': target_feature, 'label': target_label}

    # 转为tensor数据集
    train_dataset = Data.TensorDataset(source_set['feature'],
                                       source_set['label'])
    test_dataset = Data.TensorDataset(target_set['feature'],
                                      target_set['label'])

    return train_dataset, test_dataset
# DEAP_Dataset_WithinTrialKFold的全局变量声明，保证每次打乱的顺序相同
shuffle_indices = None


def DEAP_Dataset_KFold_Trial(input_dir, session, target_id, k, fold,  shuffle=True):
    global shuffle_indices
    if session == 1:
        data_dir = os.path.join(input_dir, 'Arousal')
    elif session == 2:
        data_dir = os.path.join(input_dir, 'Valence')

    file_path = os.path.join(data_dir, f'subject{target_id}.mat')
    data = scio.loadmat(file_path)
    feature_trial = data['feature']
    label_trial = data['label']

    feature = None
    label = None

    for i in range(0, 40):
        if feature is None:
            feature = torch.tensor(feature_trial[f'trial{i + 1}'][0][0]).unsqueeze(0)
            label = torch.tensor(label_trial[f'trial{i + 1}'][0][0]).unsqueeze(0)
        else:
            feature = torch.cat((feature, torch.tensor(feature_trial[f'trial{i + 1}'][0][0]).unsqueeze(0)), dim=0)
            label = torch.cat((label, torch.tensor(label_trial[f'trial{i + 1}'][0][0]).unsqueeze(0)), dim=0)

    feature = feature.permute(0, 1, 3, 2).float()
    label = label.squeeze(1).long()

    if shuffle and shuffle_indices is None:
        shuffle_indices = torch.randperm(feature.size(0))

    if shuffle:
        feature = feature[shuffle_indices]
        label = label[shuffle_indices]

    print(shuffle_indices)
    # 计算每个折的大小
    fold_size = feature.size(0) // k

    # 获取验证集的索引
    val_start = fold * fold_size
    val_end = val_start + fold_size

    # 将数据分成训练集和验证集
    target_feature = feature[val_start:val_end]
    target_label = label[val_start:val_end]

    source_feature = torch.cat((feature[:val_start], feature[val_end:]), dim=0)
    source_label = torch.cat((label[:val_start], label[val_end:]), dim=0)

    source_feature = source_feature.reshape(-1, source_feature.shape[2], source_feature.shape[3])
    source_label = source_label.reshape(-1)

    target_feature = target_feature.reshape(-1, target_feature.shape[2], target_feature.shape[3])
    target_label = target_label.reshape(-1)
    # 构建源域和目标域数据集
    source_set = {'feature': source_feature, 'label': source_label}
    target_set = {'feature': target_feature, 'label': target_label}

    # 转为tensor数据集
    train_dataset = Data.TensorDataset(source_set['feature'],
                                       source_set['label'])
    test_dataset = Data.TensorDataset(target_set['feature'],
                                      target_set['label'])

    return train_dataset, test_dataset

def DEAP_Dataset_KFold_USTrial(input_dir, session, target_id, k, fold):
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

    # 计算每个fold包含的trial数量
    trials_per_fold = 40 // k

    # 提取特征和标签
    for i in range(0, 40):
        fold_index = i // trials_per_fold
        if fold_index == fold:
            if target_feature is None:
                target_feature = torch.tensor(feature_trial[f'trial{i + 1}'][0][0])
                target_label = torch.tensor(label_trial[f'trial{i + 1}'][0][0])
            else:
                target_feature = torch.cat((target_feature, torch.tensor(feature_trial[f'trial{i + 1}'][0][0])), dim=0)
                target_label = torch.cat((target_label, torch.tensor(label_trial[f'trial{i + 1}'][0][0])), dim=0)
        else:
            if source_feature is None:
                source_feature = torch.tensor(feature_trial[f'trial{i + 1}'][0][0])
                source_label = torch.tensor(label_trial[f'trial{i + 1}'][0][0])
            else:
                source_feature = torch.cat((source_feature, torch.tensor(feature_trial[f'trial{i + 1}'][0][0])), dim=0)
                source_label = torch.cat((source_label, torch.tensor(label_trial[f'trial{i + 1}'][0][0])), dim=0)

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


if __name__ == '__main__':
    # input_dir = "E:/datasets/DEAP_Preprocessed"
    # input_dir = "E:/datasets/SEED_Preprocessed"
    input_dir = "E:/datasets/DEAP_DE_Preprocessed"
    subjects = 32
    k_fold = 10

    for session in range(1, 3):
        for i in range(1, subjects+1):
            for fold in range(k_fold):
                # DEAP_Dataset_WithinTrialKFold(input_dir, session, i, k_fold, fold)
                DEAP_Dataset_KFold_Trial(input_dir, session, i, k_fold, fold)
                # DEAP_Dataset_KFold(input_dir, session, i, k_fold, fold)
                # DEAP_DE_Dataset_KFold_Shuffle(input_dir, session, i, k_fold, fold)
    print("success")