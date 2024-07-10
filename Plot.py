import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sn
from sklearn.metrics import confusion_matrix

def plot_acc():
    path = './results/Time/WS/CMamba-2-4.mat'
    results = scio.loadmat(path)

    # 取出文件名
    parts = path.split('/')
    filename_with_extension = parts[-1]
    filename = filename_with_extension.split('.')[0]

    train_loss = results['train_loss']
    val_loss = results['val_loss']
    val_acc = results['val_acc']

    # 计算所有 subject 在每个 epoch 上的平均准确率
    mean_acc_per_epoch = np.mean(val_acc, axis=0)

    # 画出每个 subject 在每个 epoch 上的准确度曲线
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, val_acc.shape[1] + 1)
    for i in range(val_acc.shape[0]):
        plt.plot(epochs, val_acc[i], label=f'Subject {i + 1}', alpha=0.3)  # 透明度设置为0.3

    # 绘制所有 subject 在每个 epoch 上的平均准确率曲线，并加粗
    plt.plot(epochs, mean_acc_per_epoch, label='Mean', color='black', linewidth=2.5)

    plt.title(f'Validation Accuracy for Each Subject and Mean({filename})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 找到每个 subject 的最高准确率
    max_acc_per_subject = np.max(val_acc, axis=1)

    # 计算最高准确率的均值和标准差
    mean_max_acc = np.mean(max_acc_per_subject)
    std_max_acc = np.std(max_acc_per_subject)

    # 绘制每个 subject 的最高准确率柱状图
    plt.figure(figsize=(10, 6))
    subjects = np.arange(1, val_acc.shape[0] + 1)
    plt.bar(subjects, max_acc_per_subject, label='Max Accuracy per Subject')

    # 绘制均值和标准差线
    plt.axhline(mean_max_acc, color='red', linestyle='--', label=f'Mean: {mean_max_acc:.4f}')
    plt.axhline(mean_max_acc + std_max_acc, color='green', linestyle='--',
                label=f'Mean + Std: {mean_max_acc + std_max_acc:.4f}')
    plt.axhline(mean_max_acc - std_max_acc, color='blue', linestyle='--',
                label=f'Mean - Std: {mean_max_acc - std_max_acc:.4f}')

    plt.title(f'Max Validation Accuracy for Each Subject({filename})')
    plt.xlabel('Subject')
    plt.ylabel('Max Accuracy')
    plt.xticks(subjects)
    plt.legend()
    plt.show()

    mean_train_loss = np.mean(train_loss, axis=0)
    mean_val_loss = np.mean(val_loss, axis=0)

    # 绘制平均训练损失和验证损失曲线
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, val_acc.shape[1] + 1)

    # 绘制训练损失曲线
    plt.plot(epochs, mean_train_loss, label='Mean Train Loss', color='blue')

    # 绘制验证损失曲线
    plt.plot(epochs, mean_val_loss, label='Mean Validation Loss', color='red')

    plt.title(f'Mean Train and Validation Loss({filename})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_kfold_acc():
    path = './results/Time/KFold/ACRNN-1-3.mat'
    results = scio.loadmat(path)

    # 取出文件名
    parts = path.split('/')
    filename_with_extension = parts[-1]
    filename = filename_with_extension.split('.')[0]

    train_loss = results['train_loss']
    val_loss = results['val_loss']
    val_acc = results['val_acc']
    val_acc = val_acc[:, :, :]

    # 找到每个 subject 在每个 fold 的最大准确率的 fold 索引
    max_acc_per_subject = np.max(val_acc, axis=2)
    best_fold_indices = np.argmax(max_acc_per_subject, axis=1)

    # 初始化存储最佳 fold 上的准确率数据的数组
    best_fold_acc = np.zeros((val_acc.shape[0], val_acc.shape[2]))

    # 提取每个 subject 在其最佳 fold 上的准确率数据
    for subject_id in range(val_acc.shape[0]):
        best_fold_index = best_fold_indices[subject_id]
        best_fold_acc[subject_id] = val_acc[subject_id, best_fold_index]

    # 计算所有 subject 在每个 epoch 上的平均准确率
    mean_acc_per_epoch = np.mean(best_fold_acc, axis=0)

    # 画出每个 subject 在每个 epoch 上的准确度曲线
    plt.figure(figsize=(15, 10))
    epochs = np.arange(1, val_acc.shape[2] + 1)
    for i in range(best_fold_acc.shape[0]):
        plt.plot(epochs, best_fold_acc[i], label=f'Subject {i + 1}', alpha=0.3)  # 透明度设置为0.3

    # 绘制所有 subject 在每个 epoch 上的平均准确率曲线，并加粗
    plt.plot(epochs, mean_acc_per_epoch, label='Mean', color='black', linewidth=2.5)

    plt.title(f'Validation Accuracy for Each Subject and Mean({filename})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

    max_acc_per_subject = np.max(val_acc, axis=2)
    kfold_mean_acc = np.mean(max_acc_per_subject, axis=1)

    mean_max_acc = np.mean(kfold_mean_acc)
    std_max_acc = np.std(kfold_mean_acc)

    # 绘制每个 subject 的最高准确率柱状图
    plt.figure(figsize=(10, 6))
    subjects = np.arange(1, val_acc.shape[0] + 1)
    plt.bar(subjects, kfold_mean_acc, label='Max Accuracy per Subject')

    # 绘制均值和标准差线
    plt.axhline(mean_max_acc, color='red', linestyle='--', label=f'Mean: {mean_max_acc:.4f}')
    plt.axhline(mean_max_acc + std_max_acc, color='green', linestyle='--',
                label=f'Mean + Std: {mean_max_acc + std_max_acc:.4f}')
    plt.axhline(mean_max_acc - std_max_acc, color='blue', linestyle='--',
                label=f'Mean - Std: {mean_max_acc - std_max_acc:.4f}')

    plt.title(f'Max Validation Accuracy for Each Subject({filename})')
    plt.xlabel('Subject')
    plt.ylabel('Max Accuracy')
    plt.xticks(subjects)
    plt.legend()
    plt.show()

    max_acc = np.max(max_acc_per_subject, axis=1)
    mean_max_acc = np.mean(max_acc)
    std_max_acc = np.std(max_acc)

    # 绘制每个 subject 的最高准确率柱状图
    plt.figure(figsize=(10, 6))
    subjects = np.arange(1, val_acc.shape[0] + 1)
    plt.bar(subjects, max_acc, label='Max Accuracy per Subject')

    # 绘制均值和标准差线
    plt.axhline(mean_max_acc, color='red', linestyle='--', label=f'Mean: {mean_max_acc:.4f}')
    plt.axhline(mean_max_acc + std_max_acc, color='green', linestyle='--',
                label=f'Mean + Std: {mean_max_acc + std_max_acc:.4f}')
    plt.axhline(mean_max_acc - std_max_acc, color='blue', linestyle='--',
                label=f'Mean - Std: {mean_max_acc - std_max_acc:.4f}')

    plt.title(f'Max Validation Accuracy for Each Subject({filename})')
    plt.xlabel('Subject')
    plt.ylabel('Max Accuracy')
    plt.xticks(subjects)
    plt.legend()
    plt.show()


def plot_mean_loss():
    path = './results/Time/KFold/Mamba2-1-7.mat'
    results = scio.loadmat(path)

    # 取出文件名
    parts = path.split('/')
    filename_with_extension = parts[-1]
    filename = filename_with_extension.split('.')[0]

    train_loss = results['train_loss']
    val_loss = results['val_loss']

    # 计算平均损失
    mean_train_loss = np.mean(train_loss, axis=(0, 1))
    mean_val_loss = np.mean(val_loss, axis=(0, 1))

    # 画图
    epochs = np.arange(1, 101)  # 假设有100个epoch

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_train_loss, label='Mean Train Loss')
    plt.plot(epochs, mean_val_loss, label='Mean Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Mean Train and Validation Loss across Subjects and K-Folds')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_acc()


