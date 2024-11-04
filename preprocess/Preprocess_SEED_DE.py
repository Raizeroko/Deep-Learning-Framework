import numpy as np
import os
from sklearn import preprocessing
import scipy.io as scio


def preprocess_SEED_DE(session):
    # 调试
    path = f'E:/datasets/UnPreprocessed/SEED_Mamba/feature_for_net_session{session}_LDS_de'
    data_label = (2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0)

    # os.chdir: 改变工作目录
    os.chdir(path)

    # os.listdir:获取路径下的文件
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        info_ = os.path.join(domain, info)

        subject_number = info.split('_')[0]

        data = scio.loadmat(info_)


        feature = {}
        label = {}
        # 15个trail处理
        for key, value in data.items():
            for i in range(1, 16):
                if key == f"de_LDS{i}":
                    feature[f"trial{i}"] = value.transpose(1, 2, 0)
                    label[f"trial{i}"] = np.repeat(data_label[i - 1], value.shape[1])

        # 构建目标域数据集和源域数据集并返回
        save_data = {'feature': feature, 'label': label}
        file_name = f"subject{subject_number}.mat"
        # 保存数据为.mat文件
        file_path = os.path.join(f'E:\\datasets\\SEED_DE_Preprocessed_128\\Session{session}\\', file_name)
        # scio.savemat(file_path, save_data)


def preprocess_SEEDIV_DE(session):

    # 调试
    path = f'E:\datasets\SEED_IV\eeg_feature_smooth\{session}'
    data_label = {'session1': (1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3),
                  'session2': (2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1),
                  'session3': (1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0)}

    # os.chdir: 改变工作目录
    os.chdir(path)

    # os.listdir:获取路径下的文件
    for info in os.listdir(path):
        domain = os.path.abspath(path)
        info_ = os.path.join(domain, info)

        subject_number = info.split('_')[0]

        data = scio.loadmat(info_)

        feature = {}
        label = {}
        # 15个trial处理
        for key, value in data.items():
            for i in range(1, 25):
                if key == f"de_LDS{i}":
                    feature[f"trial{i}"] = value
                    label[f"trial{i}"] = np.repeat(data_label[f'session{session}'][i - 1], value.shape[1])

        # 构建目标域数据集和源域数据集并返回
        save_data = {'feature': feature, 'label': label}
        file_name = f"subject{subject_number}.mat"
        # 保存数据为.mat文件
        file_path = os.path.join(f'E:\\datasets\\SEEDIV_Preprocessed\\Session{session}\\', file_name)
        # scio.savemat(file_path, save_data)

if __name__ == '__main__':
    for session in range(1, 4):
        preprocess_SEED_DE(session)
        # preprocess_SEEDIV_DE(session)