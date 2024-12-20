import os
import torch
import torch.utils.data as Data
import scipy.io as scio

shuffle_indices = None

def Dataset_KFold_Sample(dataset, input_dir, session, target_id, trial, k, fold, shuffle=True):
    global shuffle_indices
    data_dir = None
    if dataset == 'SEED' or dataset == 'SEEDIV':
        # SEED/SEEDIV
        data_dir = os.path.join(input_dir, f'Session{session}')
    elif dataset == 'DREAMER' or dataset == 'DEAP':
        # DEAP/DREAMER
        if session == 1:
            data_dir = os.path.join(input_dir, 'Arousal')
        elif session == 2:
            data_dir = os.path.join(input_dir, 'Valence')
        elif session == 3:
            data_dir = os.path.join(input_dir, 'Dominance')
    elif dataset == 'CEED':
        data_dir = input_dir
    if data_dir == None:
        print('Dataset Error!')

    file_path = os.path.join(data_dir, f'subject{target_id}.mat')
    data = scio.loadmat(file_path)
    feature_trial = data['feature']
    label_trial = data['label']

    feature = None
    label = None

    for i in range(trial):
        if feature is None:
            feature = torch.tensor(feature_trial[f'trial{i + 1}'][0][0])
            label = torch.tensor(label_trial[f'trial{i + 1}'][0][0]).reshape(-1)
        else:
            feature = torch.cat((feature, torch.tensor(feature_trial[f'trial{i + 1}'][0][0])), dim=0)
            label = torch.cat((label, torch.tensor(label_trial[f'trial{i + 1}'][0][0]).reshape(-1)), dim=0)

    feature = feature.permute(0, 2, 1).float()
    label = label.long()

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


def Dataset_KFold_Trial(dataset, input_dir, session, target_id, trial, k, fold,  shuffle=True):
    global shuffle_indices
    data_dir = None
    if dataset == 'SEED' or dataset == 'SEEDIV':
        # SEED
        data_dir = os.path.join(input_dir, f'Session{session}')
    elif dataset == 'DREAMER' or dataset == 'DEAP':
        # DEAP/DREAMER
        if session == 1:
            data_dir = os.path.join(input_dir, 'Arousal')
        elif session == 2:
            data_dir = os.path.join(input_dir, 'Valence')
        elif session == 3:
            data_dir = os.path.join(input_dir, 'Dominance')
    elif dataset == 'CEED':
        data_dir = input_dir
    if data_dir == None:
        print('Dataset Error!')

    file_path = os.path.join(data_dir, f'subject{target_id}.mat')
    data = scio.loadmat(file_path)
    feature_trial = data['feature']
    label_trial = data['label']

    # 使用字典存储 trial 数据
    trials = {'feature': [], 'label': []}

    # 如果需要打乱 trial 的顺序，生成打乱后的索引
    if shuffle_indices is None:
        if shuffle:
            shuffle_indices = torch.randperm(trial)
        else:
            shuffle_indices = torch.arange(trial)

    print(shuffle_indices)

    for i in shuffle_indices:
        trial_feature = torch.tensor(feature_trial[f'trial{i + 1}'][0][0])
        trial_label = torch.tensor(label_trial[f'trial{i + 1}'][0][0]).long().reshape(-1)

        # 将数据存入字典
        trials['feature'].append(trial_feature)
        trials['label'].append(trial_label)

    if trial % k != 0:
        raise ValueError(f"Trial {trial} is not divisible by {k}")

    # 计算每个 fold 的大小
    fold_size = len(trials['feature']) // k
    val_start = fold * fold_size
    val_end = val_start + fold_size

    # 构建 source 和 target 字典
    source = {'feature': trials['feature'][:val_start] + trials['feature'][val_end:],
              'label': trials['label'][:val_start] + trials['label'][val_end:]}
    target = {'feature': trials['feature'][val_start:val_end],
              'label': trials['label'][val_start:val_end]}

    # 将 source 和 target 中的列表堆叠到一起, (B, C, F)
    source_feature = torch.cat(source['feature'], dim=0).permute(0, 2, 1).float()
    source_label = torch.cat(source['label'], dim=0)


    target_feature = torch.cat(target['feature'], dim=0).permute(0, 2, 1).float()
    target_label = torch.cat(target['label'], dim=0)

    # 构建源域和目标域数据集
    train_dataset = Data.TensorDataset(source_feature, source_label)
    test_dataset = Data.TensorDataset(target_feature, target_label)

    return train_dataset, test_dataset

if __name__ == '__main__':
    input_dir = "E:/datasets/SEED_DE_Preprocessed_128"
    # input_dir = "E:/datasets/DEAP_DE_Preprocessed_384"
    # input_dir = "E:/datasets/DEAP_Time_Preprocessed_128"
    # input_dir = "E:/datasets/DEAP_Preprocessed"
    # input_dir = "E:/datasets/DREAMER_Preprocessed"
    # input_dir = "E:/datasets/SEED_DE_Preprocessed_128"
    # input_dir = "E:/datasets/SEED_Time_Preprocessed_128"

    dataset = input_dir.split('/')[-1].split('_')[0]
    if dataset == 'SEED' or dataset == 'SEEDIV':
        subjects = 15
        trial = 15
        kfold = 5
    elif dataset == 'DEAP':
        subjects = 32
        trial = 40
        kfold = 10
    elif dataset == 'DREAMER':
        subjects = 23
        trial = 18
        kfold = 6

    for session in range(1, 4):
        for i in range(1, subjects+1):
            for fold in range(kfold):
                train_dataset, test_dataset = Dataset_KFold_Trial(dataset, input_dir, session, i, trial, kfold, fold)
                # train_dataset, test_dataset = Dataset_KFold_Sample(dataset, input_dir, session, i, trial, kfold, fold)

                print(f"session: {session}, subject: {i}, fold: {fold}")
    print("success")