import os
import scipy.io as scio
import torch
import numpy as np
from torch import nn
import time

params = {'in_channels': 384,
          'num_electrodes': 32,
          'num_layers': 4,
          'hid_channels': 32,
          'num_heads': 6,
          'num_classes': 2,
          'lr': 1e-4,
          'weight_decay': 1e-4,
          'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
          'epoch': 100,
          'batch_size': 20,
          'session': 1,
          'data_dir': "/home/zwr/dataset/DEAP_Preprocessed",      #Linux
          # 'data_dir': "E:/datasets/DEAP_Preprocessed",    #Windows
          'val': "WS",   # 选择验证方式：WS/WSSS/LOSO
          'net': "CMamba"    # 选择网络：DGCNN/py_attention/linear/EmoGT/attention/ACRNN/CGT
          }

def save_results(results):
    # 设置文件名的初始后缀数字
    suffix_number = 1
    # 构建文件名
    file_name = f"./results/{params['val']}/{params['net']}-{params['session']}-{suffix_number}.mat"
    # 检查文件是否存在，如果存在，则增加后缀数字
    while os.path.exists(file_name):
        suffix_number += 1
        file_name = f"./results/{params['val']}/{params['net']}-{params['session']}-{suffix_number}.mat"
    # 执行保存操作
    scio.savemat(file_name, results)


def test_gpu():
    # 打印系统中可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")

    # 如果有多个 GPU，可以选择特定的 GPU
    if num_gpus > 1:
        # 选择第二个 GPU（索引从0开始，所以第二个 GPU 的索引是 1）
        device = torch.device("cuda:1")
    else:
        # 如果只有一个 GPU 或没有 GPU，选择 CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印选择的设备
    print(f"Using device: {device}")

    # 创建张量并将其移动到选择的设备上
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    print(x)


# 测试函数
def test_speed(network, input_data, iterations=100):
    # 热身运行
    for _ in range(10):
        _ = network(input_data)

    # 实际测量
    start_time = time.time()
    for _ in range(iterations):
        _ = network(input_data)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    return avg_time



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_kfold():
    sub_train_loss, sub_val_loss, sub_val_acc = [], [], []
    for i in range(1, 33):
        fold_train_losses = []  # 用于存储每个 epoch 的训练损失
        fold_val_losses = []  # 用于存储每个 epoch 的验证损失
        fold_val_accs = []  # 用于存储每个 epoch 的验证准确率
        for fold in range(10):
            train_losses = []  # 用于存储每个 epoch 的训练损失
            val_losses = []  # 用于存储每个 epoch 的验证损失
            val_accs = []  # 用于存储每个 epoch 的验证准确率
            for epoch in range(100):
                epoch_loss = epoch
                val_loss, val_acc = epoch, epoch
                train_losses.append(epoch_loss)
                val_losses.append(val_loss)
                val_accs.append(val_acc)
            fold_train_losses.append(train_losses)
            fold_val_losses.append(val_losses)
            fold_val_accs.append(val_accs)
        sub_train_loss.append(fold_train_losses)
        sub_val_loss.append(fold_val_losses)
        sub_val_acc.append(fold_val_accs)

    results = {'train_loss': sub_train_loss, 'val_loss': sub_val_loss, 'val_acc': sub_val_acc}
    print(results)
    scio.savemat('./test.mat', results)

def test_plot():
    path = './test.mat'
    results = scio.loadmat(path)
    print(results)

def load_deap():
    return

def test_cat():
    a = torch.randn(131, 128, 14)
    b = torch.randn(199, 128, 14)
    c = torch.stack([a, b], dim=0)
    print(c.shape)

def test_dataset():
    data = scio.loadmat('E:/datasets/SEED_Preprocessed/Session1/subject1.mat')
    data = data['label'][0][0]
    print(data)

if __name__ == '__main__':
    # test_cat()
    test_dataset()
    # load_deap()
    # test_kfold()
    # test_plot()
    # test_cwc()
    # test_mamba()
    # test_gpu()
    # save_results({"test": 1})