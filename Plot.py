import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
import pandas as pd

def save_excel(data):
    # 将数据转换为 DataFrame
    df = pd.DataFrame(data)
    # 保存数据到 Excel 文件
    df.to_excel('out.xlsx')
    print(f"Max accuracy per subject been saved")


# 提取文件名
def extract_filename(path):
    return path.split('/')[-1].split('.')[0]

# 画出验证准确率
def plot_val_accuracy(val_acc, filename, subject_indices):
    mean_acc_per_epoch = np.mean(val_acc, axis=0)
    epochs = np.arange(1, val_acc.shape[1] + 1)

    plt.figure(figsize=(10, 6))
    for i in subject_indices:
        plt.plot(epochs, val_acc[i], label=f'Subject {i + 1}', alpha=0.3)

    plt.plot(epochs, mean_acc_per_epoch, label='Mean', color='black', linewidth=2.5)
    plt.title(f'Validation Accuracy for Selected Subjects and Mean ({filename})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(False)
    plt.show()

# 绘制柱状图
def plot_accuracy_bar(acc_per_subject, filename, params, subject_indices):
    selected_acc = acc_per_subject[subject_indices]
    mean_acc = np.mean(selected_acc)
    std_acc = np.std(selected_acc)

    plt.figure(figsize=(10, 6))
    plt.bar(subject_indices + 1, selected_acc, label='Max Accuracy per Subject')

    # 绘制均值和标准差线
    plt.axhline(mean_acc, color='red', linestyle='--', label=f'Mean: {mean_acc:.4f}')
    plt.axhline(mean_acc + std_acc, color='green', linestyle='--',
                label=f'Mean + Std: {mean_acc + std_acc:.4f}')
    plt.axhline(mean_acc - std_acc, color='blue', linestyle='--',
                label=f'Mean - Std: {mean_acc - std_acc:.4f}')

    # plt.title(f'{filename} (batch_size:{params["batch_size"][0][0]}, epoch:{params["epoch"][0][0]}, '
    #           f'data:{params["dataset"][0]}, val:{params["val"][0]})')
    plt.xlabel('Subject')
    plt.ylabel('Accuracy')
    plt.xticks(subject_indices + 1)
    plt.legend()
    plt.grid(False)
    plt.show()

# 绘制损失曲线
def plot_loss_curve(train_loss, val_loss, filename, subject_indices):
    mean_train_loss = np.mean(train_loss[subject_indices], axis=0)
    mean_val_loss = np.mean(val_loss[subject_indices], axis=0)
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
def plot_acc(path, subjects=None, excel=False):
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

    # 如果没有指定 subjects 则绘制所有被试
    if subjects is None:
        subject_indices = np.arange(val_acc.shape[0])
    else:
        subject_indices = np.array(subjects) - 1  # subjects 是从 1 开始的

    # 画验证准确率曲线
    plot_val_accuracy(val_acc, filename, subject_indices)

    # 绘制最大准确率
    max_acc_per_subject = np.max(val_acc, axis=1)
    plot_accuracy_bar(max_acc_per_subject, filename, params, subject_indices)

    if excel:
        save_excel(max_acc_per_subject)

    # 绘制损失曲线
    plot_loss_curve(train_loss, val_loss, filename, subject_indices)

# 处理并绘制KFold结果
def plot_kfold_acc(path, subjects=None, excel=False):
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

    # 保存为excel
    if excel:
        save_excel(max_acc_per_subject)

    # 提取每个 subject 在其最佳 fold 上的准确率数据
    best_fold_acc = np.array([val_acc[i, best_fold_indices[i]] for i in range(val_acc.shape[0])])

    # 如果没有指定 subjects 则绘制所有被试
    if subjects is None:
        subject_indices = np.arange(val_acc.shape[0])
    else:
        subject_indices = np.array(subjects) - 1  # subjects 是从 1 开始的

    # 绘制验证准确率曲线
    plot_val_accuracy(best_fold_acc, filename, subject_indices)

    # 绘制10折平均准确率
    kfold_mean_acc = np.mean(max_acc_per_subject, axis=1)
    plot_accuracy_bar(kfold_mean_acc, filename, params, subject_indices)

    # 绘制10折最大准确率
    kfold_max_acc = np.max(max_acc_per_subject, axis=1)
    plot_accuracy_bar(kfold_max_acc, filename, params, subject_indices)

    # 绘制损失曲线
    plot_loss_curve(train_loss, val_loss, filename, subject_indices)


if __name__ == '__main__':
    path = './results/DE/KFold/DGCNN-3-1.mat'
    # path = './results/Time/WS/GMA-1-5.mat'
    # 选择指定被试画图
    # subjects = [3, 5, 8, 18, 27]

    # 保存为excel
    excel = True
    subjects = None
    if "KFold" in path:
        plot_kfold_acc(path, subjects, excel)
    else:
        plot_acc(path, subjects, excel)

