import torch
import numpy as np
import random
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_params(params):
    if params['dataset'] == 'DEAP':
        params['num_electrodes'] = 32           #DEAP有62给电极通道
        params['num_classes'] = 2               #DEAP是二分类:high, low
        params['subjects'] = 32
        if os.name == 'posix':                  # Linux or macOS
            path = '/home/zwr/dataset/'
        elif os.name == 'nt':                   # Windows
            path = 'E:/datasets/'
        if params['feature'] == 'DE':
            data = "DEAP_DE_Preprocessed_384"
        elif params['feature'] == 'Time':
            data = "DEAP_Preprocessed"
        params['data_dir'] = os.path.join(path, data)

    elif params['dataset'] == 'SEED' or 'SEEDIV':
        params['num_electrodes'] = 62           #SEED有62给电极通道
        params['subjects'] = 32
        if params['dataset'] == 'SEED':
            params['num_classes'] = 3           #SEED是三分类:positive, negative, neutral
            data = 'SEED_Preprocessed'
        elif params['dataset'] == 'SEEDIV':
            params['num_classes'] = 4           #SEEDIV是三分类:sad, fear, happy, neutral
            data = 'SEEDIV_Preprocessed'
        if os.name == 'posix':                  # Linux or macOS
            path = '/home/zwr/dataset/'
        elif os.name == 'nt':                   # Windows
            path = 'E:/datasets/'
        params['data_dir'] = os.path.join(path, data)
    return params

