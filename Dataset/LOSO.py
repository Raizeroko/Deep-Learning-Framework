import os
import torch
import torch.utils.data as Data
import scipy.io as scio

def SEED_Dataset_LOSOCV(input_dir, session, target_id):
    data_dir = os.path.join(input_dir, f'Session{session}')

    # 初始话空列表，用于存储从数据文件中读取的特征和标签
    feature_list = []
    label_list = []
    for i in range(1, 16):
        file_path = os.path.join(data_dir, f'subject{i}.mat')
        data = scio.loadmat(file_path)
        feature_trial = data['feature']
        label_trial = data['label']
        feature = None
        label = None

        # 提取subject i 的feature和label
        for i in range(1, 16):
            if feature == None:
                feature = torch.tensor(feature_trial[f'trial{i}'][0][0])
                label = torch.tensor(label_trial[f'trial{i}'][0][0])
            else:
                feature = torch.cat((feature, torch.tensor(feature_trial[f'trial{i}'][0][0])), dim=1)
                label = torch.cat((label, torch.tensor(label_trial[f'trial{i}'][0][0])), dim=1)

        feature = feature.permute(1, 0, 2)
        # 将label转为one_hot
        label = label.reshape(-1)
        label = label.long() + 1
        # one_hot_label = torch.nn.functional.one_hot(label)

        feature_list.append(feature)
        label_list.append(label)
        # label_list.append(one_hot_label)

    target_feature, target_label = feature_list[target_id], label_list[target_id]
    del feature_list[target_id]
    del label_list[target_id]
    source_feature, source_label = torch.stack(feature_list), torch.stack(label_list)
    # 构建目标域数据集和源域数据集并返回
    source_feature = source_feature.reshape(-1, 62, 5)
    source_label = source_label.reshape(-1)

    source_set = {'feature': source_feature.float(), 'label': source_label}
    target_set = {'feature': target_feature.float(), 'label': target_label}

    # 转为tensor数据集
    train_dataset = Data.TensorDataset(source_set['feature'],
                                       source_set['label'])
    test_dataset = Data.TensorDataset(target_set['feature'],
                                      target_set['label'])

    return train_dataset, test_dataset