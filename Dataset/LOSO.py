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

    target_feature, target_label = feature_list[target_id-1], label_list[target_id-1]
    del feature_list[target_id-1]
    del label_list[target_id-1]
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





def DEAP_Dataset_LOSOCV(input_dir, session, target_id):
    if session == 1:
        data_dir = os.path.join(input_dir, 'Arousal')
    elif session == 2:
        data_dir = os.path.join(input_dir, 'Valence')

    # 初始话空列表，用于存储从数据文件中读取的特征和标签
    feature_list = []
    label_list = []
    for i in range(32):
        file_path = os.path.join(data_dir, f'subject{i+1}.mat')
        data = scio.loadmat(file_path)
        feature_trial = data['feature']
        label_trial = data['label']
        feature = None
        label = None

        # 提取subject i 的feature和label
        for i in range(40):
            if feature == None:
                feature = torch.tensor(feature_trial[f'trial{i+1}'][0][0])
                label = torch.tensor(label_trial[f'trial{i+1}'][0][0])
            else:
                feature = torch.cat((feature, torch.tensor(feature_trial[f'trial{i+1}'][0][0])), dim=0)
                label = torch.cat((label, torch.tensor(label_trial[f'trial{i+1}'][0][0])), dim=0)

        feature = feature.permute(0, 2, 1).float()
        label = label.reshape(-1).long()

        feature_list.append(feature)
        label_list.append(label)
        # label_list.append(one_hot_label)

    target_feature, target_label = feature_list[target_id-1], label_list[target_id-1]
    del feature_list[target_id-1]
    del label_list[target_id-1]
    source_feature, source_label = torch.stack(feature_list), torch.stack(label_list)
    # 构建目标域数据集和源域数据集并返回
    source_feature = source_feature.reshape(-1, source_feature.shape[2], source_feature.shape[3])
    source_label = source_label.reshape(-1)

    source_set = {'feature': source_feature.float(), 'label': source_label}
    target_set = {'feature': target_feature.float(), 'label': target_label}

    # 转为tensor数据集
    train_dataset = Data.TensorDataset(source_set['feature'],
                                       source_set['label'])
    test_dataset = Data.TensorDataset(target_set['feature'],
                                      target_set['label'])

    return train_dataset, test_dataset


if __name__ == '__main__':
    # input_dir = "/home/zwr/dataset/DEAP_Preprocessed"         #linux
    input_dir = "E:/datasets/DEAP_Preprocessed"                 #windows
    # input_dir = "E:/datasets/DEAP_DE_Preprocessed"

    subjects = 32
    for session in range(1, 3):
        for i in range(1, subjects+1):
            print(f"session: {session}, subject: {i}")
            DEAP_Dataset_LOSOCV(input_dir, session, i)
    print("success")
