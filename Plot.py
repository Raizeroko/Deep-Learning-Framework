import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

# 提取文件名
def extract_filename(path):
    parts = path.split('/')
    filename_with_extension = parts[-1]
    filename = filename_with_extension.split('.')[0]
    return filename

# 画出验证准确率
def plot_val_accuracy(val_acc, filename):
    mean_acc_per_epoch = np.mean(val_acc, axis=0)
    epochs = np.arange(1, val_acc.shape[1] + 1)

    plt.figure(figsize=(10, 6))
    for i in range(val_acc.shape[0]):
        plt.plot(epochs, val_acc[i], label=f'Subject {i + 1}', alpha=0.3)

    plt.plot(epochs, mean_acc_per_epoch, label='Mean', color='black', linewidth=2.5)
    plt.title(f'Validation Accuracy for Each Subject and Mean ({filename})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(False)
    plt.show()

# 绘制柱状图
def plot_accuracy_bar(subjects, acc_per_subject, filename, params):

    mean_acc = np.mean(acc_per_subject)
    std_acc = np.std(acc_per_subject)

    plt.figure(figsize=(10, 6))
    plt.bar(subjects, acc_per_subject, label='Max Accuracy per Subject')

    # 绘制均值和标准差线
    plt.axhline(mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.4f}')
    plt.axhline(mean_acc + std_acc, color='green', linestyle='--',
                label=f'Mean + Std: {mean_acc + std_acc:.4f}')
    plt.axhline(mean_acc - std_acc, color='blue', linestyle='--',
                label=f'Mean - Std: {mean_acc - std_acc:.4f}')

    plt.title(f'{filename}(batch_size:{params["batch_size"][0][0]}, epoch:{params["epoch"][0][0]}, val:{params["val"][0]})')
    plt.xlabel('Subject')
    plt.ylabel('Max Accuracy')
    plt.xticks(subjects)
    plt.legend()
    plt.grid(False)
    plt.show()

# 绘制损失曲线
def plot_loss_curve(train_loss, val_loss, filename):
    mean_train_loss = np.mean(train_loss, axis=0)
    mean_val_loss = np.mean(val_loss, axis=0)
    epochs = np.arange(1, train_loss.shape[1] + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_train_loss, label='Mean Train Loss', color='blue')
    plt.plot(epochs, mean_val_loss, label='Mean Validation Loss', color='red')
    plt.title(f'Mean Train and Validation Loss ({filename})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(False)
    plt.show()

# 处理并绘制普通结果
def plot_acc(path):
    results = scio.loadmat(path)
    filename = extract_filename(path)

    # 提取并打印参数
    params = results['params'][0, 0]
    keys = params.dtype.names
    for key in keys:
        print(f'{key}: {params[key][0]}')

    val_acc = results['val_acc']
    train_loss = results['train_loss']
    val_loss = results['val_loss']

    # 画验证准确率曲线
    plot_val_accuracy(val_acc, filename)

    # 排列被试
    subjects = np.arange(1, val_acc.shape[0] + 1)

    # 绘制最大准确率
    max_acc_per_subject = np.max(val_acc, axis=1)
    plot_accuracy_bar(subjects, max_acc_per_subject, filename, params)

    # 绘制损失曲线
    plot_loss_curve(train_loss, val_loss, filename)

# 处理并绘制KFold结果
def plot_kfold_acc(path):
    results = scio.loadmat(path)
    filename = extract_filename(path)

    # 提取并打印参数
    params = results['params'][0, 0]
    keys = params.dtype.names
    for key in keys:
        print(f'{key}: {params[key][0]}')

    val_acc = results['val_acc']
    train_loss = results['train_loss']
    val_loss = results['val_loss']
    train_loss = np.mean(train_loss, axis=1)
    val_loss = np.mean(val_loss, axis=1)
    val_acc = val_acc[:, :, :]

    # 找到每个 subject 的最高 fold 索引
    max_acc_per_subject = np.max(val_acc, axis=2)
    best_fold_indices = np.argmax(max_acc_per_subject, axis=1)

    # 提取每个 subject 在其最佳 fold 上的准确率数据
    best_fold_acc = np.array([val_acc[i, best_fold_indices[i]] for i in range(val_acc.shape[0])])

    # 绘制验证准确率曲线
    plot_val_accuracy(best_fold_acc, filename)

    # 排列被试
    subjects = np.arange(1, val_acc.shape[0] + 1)
    # 绘制10折平均准确率
    kfold_mean_acc = np.mean(max_acc_per_subject, axis=1)
    plot_accuracy_bar(subjects, kfold_mean_acc, filename, params)

    # 绘制10折最大准确率
    kfold_max_acc = np.max(max_acc_per_subject, axis=1)
    plot_accuracy_bar(subjects, kfold_max_acc, filename, params)

    # 绘制损失曲线
    plot_loss_curve(train_loss, val_loss, filename)

if __name__ == '__main__':
    path = './results/Time/KFold/Mamba-1-8.mat'
    # path = './results/Time/WS/GMA-1-2.mat'
    if "KFold" in path:
        plot_kfold_acc(path)
    else:
        plot_acc(path)

