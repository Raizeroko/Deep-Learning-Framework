from torch import nn
from Dataset.WS import *
from Dataset.LOSO import *
from Dataset.KFold import *
from ChooseNet import *

def evaluate_model(model, dataloader, lossFunction, device):
    model.eval()  # 将模型设为评估模式
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = lossFunction(outputs, labels)
            # 统计损失
            running_loss += loss.item() * inputs.size(0)
            # 统计正确预测的样本数量
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_preds += labels.size(0)
    # 计算平均损失和准确率
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = correct_preds / total_preds
    return val_loss, val_acc


# def train_and_test(test_id, max_iter, batch_size):
def train_and_validation(net, train_iter, test_iter, num_epochs, lr, weight_decay, device):
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.to(device)
    new_lr = None

    train_losses, val_losses, val_accs = [], [], []  # 用于存储每个 epoch 的训练损失
    for epoch in range(num_epochs):
        net.train()  # 将模型设为训练模式
        running_loss = 0.0
        for inputs, labels in train_iter:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = inputs.reshape(32, 5, 62, 5)
            # labels = labels.reshape(32, 5)
            # 前向传播
            outputs = net(inputs)
            # 计算损失
            loss = lossFunction(outputs, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 统计损失
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_iter.dataset)
        print(f'Train Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        val_loss, val_acc = evaluate_model(net, test_iter, lossFunction, device)
        print(f'Validation Epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if val_acc > 0.92 and new_lr == None:
            new_lr = lr*0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            print(f'Reducing learning rate to {new_lr:.1e}')


    return train_losses, val_losses, val_accs


def train_by_WSSS(params):
    data_dir = params['data_dir']
    net = choose_net(params)

    train_dataset, test_dataset = SEED_Dataset_WSSSCV(data_dir, params['session'])


    loader_train = Data.DataLoader(
        dataset=train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0
    )
    loader_test = Data.DataLoader(
        dataset=test_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=0
    )

    train_losses, val_losses, val_accs = train_and_validation(net,
                                                              loader_train,
                                                              loader_test,
                                                              num_epochs=params['epoch'],
                                                              lr=params['lr'],
                                                              weight_decay=params['weight_decay'],
                                                              device=params['device'])

    results = {'train_loss': train_losses, 'val_loss': val_losses, 'val_acc': val_accs, 'params': params}
    return results


def train_by_WS(params):
    data_dir = params['data_dir']
    torch.autograd.set_detect_anomaly(True)

    sub_train_loss, sub_val_loss, sub_val_acc = [], [], []

    dataset_name = data_dir.split('/')[-1].split('_')[0]
    if dataset_name == 'SEED' or dataset_name == 'SEEDIV':
        subjects = 15
    elif dataset_name == 'DEAP':
        subjects = 32

    for i in range(1, subjects+ 1):

        net = choose_net(params)
        # net = choose_CMamba(params)
        if dataset_name == 'SEED':

            train_dataset, test_dataset = SEED_Dataset_WS(data_dir, params['session'], i)
        elif dataset_name == 'DEAP':
            train_dataset, test_dataset = DEAP_Dataset_WS(data_dir, params['session'], i)

        loader_train = Data.DataLoader(
            dataset=train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0
        )
        loader_test = Data.DataLoader(
            dataset=test_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0
        )

        print(f'subject {i}:')
        train_losses, val_losses, val_accs = train_and_validation(net,
                                                                  loader_train,
                                                                  loader_test,
                                                                  num_epochs=params['epoch'],
                                                                  lr=params['lr'],
                                                                  weight_decay=params['weight_decay'],
                                                                  device=params['device'])
        sub_train_loss.append(train_losses)
        sub_val_loss.append(val_losses)
        sub_val_acc.append(val_accs)

    # 避免params中的device在保存时出错
    params_copy = params.copy()
    if 'device' in params_copy:
        del params_copy['device']

    results = {'train_loss': sub_train_loss, 'val_loss': sub_val_loss, 'val_acc': sub_val_acc, 'params': params_copy}
    return results


def train_by_LOSO(params):
    data_dir = params['data_dir']
    torch.autograd.set_detect_anomaly(True)

    sub_train_loss, sub_val_loss, sub_val_acc = [], [], []

    dataset_name = data_dir.split('/')[-1].split('_')[0]
    if dataset_name == 'SEED' or dataset_name == 'SEEDIV':
        subjects = 15
    elif dataset_name == 'DEAP':
        subjects = 32

    for i in range(1, subjects+1):
        net = choose_net(params)

        if dataset_name == 'SEED':
            train_dataset, test_dataset = SEED_Dataset_LOSOCV(data_dir, params['session'], i)
        elif dataset_name == 'DEAP':
            train_dataset, test_dataset = DEAP_Dataset_LOSOCV(data_dir, params['session'], i)


        loader_train = Data.DataLoader(
            dataset=train_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0
        )
        loader_test = Data.DataLoader(
            dataset=test_dataset,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=0
        )

        train_losses, val_losses, val_accs = train_and_validation(net,
                                                                  loader_train,
                                                                  loader_test,
                                                                  num_epochs=params['epoch'],
                                                                  lr=params['lr'],
                                                                  weight_decay=params['weight_decay'],
                                                                  device=params['device'])
        sub_train_loss.append(train_losses)
        sub_val_loss.append(val_losses)
        sub_val_acc.append(val_accs)

    # 避免params中的device在保存时出错
    params_copy = params.copy()
    if 'device' in params_copy:
        del params_copy['device']

    results = {'train_loss': sub_train_loss, 'val_loss': sub_val_loss, 'val_acc': sub_val_acc, 'params': params_copy}
    return results

def train_by_KFold(params):
    data_dir = params['data_dir']
    torch.autograd.set_detect_anomaly(True)

    sub_train_loss, sub_val_loss, sub_val_acc = [], [], []

    dataset_name = data_dir.split('/')[-1].split('_')[0]
    if dataset_name == 'SEED' or dataset_name == 'SEEDIV':
        subjects = 15
    elif dataset_name == 'DEAP':
        subjects = 32

    for i in range(1, subjects + 1):
        fold_train_loss, fold_val_loss, fold_val_acc = [], [], []
        for fold in range(params['KFold']):
            net = choose_net(params)
            # net = choose_CMamba(params)
            if dataset_name == 'SEED':
                train_dataset, test_dataset = SEED_Dataset_KFold(data_dir, params['session'], i, params['KFold'], fold)
            elif dataset_name == 'DEAP':
                train_dataset, test_dataset = DEAP_Dataset_KFold(data_dir, params['session'], i, params['KFold'], fold)

            loader_train = Data.DataLoader(
                dataset=train_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=0
            )
            loader_test = Data.DataLoader(
                dataset=test_dataset,
                batch_size=params['batch_size'],
                shuffle=True,
                num_workers=0
            )

            print(f'subject {i}:')
            print(f"Fold {fold}:")
            train_losses, val_losses, val_accs = train_and_validation(net,
                                                                      loader_train,
                                                                      loader_test,
                                                                      num_epochs=params['epoch'],
                                                                      lr=params['lr'],
                                                                      weight_decay=params['weight_decay'],
                                                                      device=params['device'])
            fold_train_loss.append(train_losses)
            fold_val_loss.append(val_losses)
            fold_val_acc.append(val_accs)
        sub_train_loss.append(fold_train_loss)
        sub_val_loss.append(fold_val_loss)
        sub_val_acc.append(fold_val_acc)

    # 避免params中的device在保存时出错
    params_copy = params.copy()
    if 'device' in params_copy:
        del params_copy['device']

    results = {'train_loss': sub_train_loss, 'val_loss': sub_val_loss, 'val_acc': sub_val_acc, 'params': params_copy}
    return results



def save_results(results):
    # 设置文件名的初始后缀数字
    suffix_number = 1
    # 构建文件名
    file_name = f"./results/{params['DE/Time']}/{params['val']}/{params['net']}-{params['session']}-{suffix_number}.mat"
    # 检查文件是否存在，如果存在，则增加后缀数字
    while os.path.exists(file_name):
        suffix_number += 1
        file_name = f"./results/{params['DE/Time']}/{params['val']}/{params['net']}-{params['session']}-{suffix_number}.mat"
    # 执行保存操作
    scio.savemat(file_name, results)


# 代码运行开始-设置参数
params = {'emb_dim': 48,  # embedding dimension of Embedding, Self-Attention, Mamba
          'emb_kernel': 16,  # 2D-conv embedding length of Embedding
          'd_state': 16,  # d_state of Mamba2
          'd_conv': 4,  # d_conv of Mamba2
          'expand': 4,  # expand of Mamba2
          'headdim': 8,  # headdim of Mamba2
          'num_layers': 1,  # d_conv of MambaFormer
          'num_classes': 2,  # num classes of emotion
          'dropout': 0.5,  # dropout of Embedding, Self-Attention, Mamba
          'num_electrodes': 32,  # num electordes of dataset
          'num_heads': 8,  # num head of Self-Attention

          'lr': 1e-3,
          'weight_decay': 1e-4,
          'device': torch.device("cuda:0"),
          'epoch': 100,
          'batch_size': 64,
          'session': 1,
          # 'data_dir': "/home/zwr/dataset/DEAP_Preprocessed",  # Linux
          'data_dir': "E:/datasets/DEAP_Preprocessed",                #Windows
          'val': "KFold",  # 选择验证方式：WS/WSSS/LOSO/KFold
          'KFold': 10,
          'net': "ACRNN",  # 选择网络：ACRNN/Mamba
          'DE/Time': "Time"
          }


if __name__ == '__main__':
    # within_subject_single_subject
    if params['val'] == "WS":
        results = train_by_WS(params)
        save_results(results)
    # within_subject_within_subject
    elif params['val'] == "WSSS":
        results = train_by_WSSS(params)
        save_results(results)
    # leave_one_subject_out
    elif params['val'] == "LOSO":
        results = train_by_LOSO(params)
        save_results(results)
    elif params['val'] == "KFold":
        results = train_by_KFold(params)
        save_results(results)


