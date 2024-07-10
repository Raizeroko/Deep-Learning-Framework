import os
import torch
import torch.utils.data as Data
import scipy.io as scio

# SEED未完成
def SEED_Dataset_KFold(input_dir, session, target_id, k, fold):
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

    # 转为tensor数据集
    train_dataset = Data.TensorDataset(source_set['feature'],
                                       source_set['label'])
    test_dataset = Data.TensorDataset(target_set['feature'],
                                      target_set['label'])

    return train_dataset, test_dataset



def DEAP_Dataset_WithinTrialKFold(input_dir, session, target_id, k, fold,  shuffle=True):
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


def DEAP_Dataset_KFold_Shuffle(input_dir, session, target_id, k, fold,  shuffle=True):
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

def DEAP_Dataset_KFold(input_dir, session, target_id, k, fold):
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
    input_dir = "E:/datasets/DEAP_Preprocessed"
    session = 1
    k_fold = 10
    for i in range(1,2):
        for fold in range(10):
            # DEAP_Dataset_WithinTrialKFold(input_dir, session, i, k_fold, fold)
            DEAP_Dataset_KFold_Shuffle(input_dir, session, i, k_fold, fold)
            # DEAP_Dataset_KFold(input_dir, session, i, k_fold, fold)