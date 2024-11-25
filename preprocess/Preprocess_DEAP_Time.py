import scipy.io as scio
import os
import numpy as np
import time
import scipy.io as scio

np.random.seed(0)


def data_1Dto2D(data, Y=9, X=9):
    data_2D = np.zeros([Y, X])
    data_2D[0] = (0, 0, 0, data[0], 0, data[16], 0, 0, 0)
    data_2D[1] = (0, 0, 0, data[1], 0, data[17], 0, 0, 0)
    data_2D[2] = (data[3], 0, data[2], 0, data[18], 0, data[19], 0, data[20])
    data_2D[3] = (0, data[4], 0, data[5], 0, data[22], 0, data[21], 0)
    data_2D[4] = (data[7], 0, data[6], 0, data[23], 0, data[24], 0, data[25])
    data_2D[5] = (0, data[8], 0, data[9], 0, data[27], 0, data[26], 0)
    data_2D[6] = (data[11], 0, data[10], 0, data[15], 0, data[28], 0, data[29])
    data_2D[7] = (0, 0, 0, data[12], 0, data[30], 0, 0, 0)
    data_2D[8] = (0, 0, 0, data[13], data[14], data[31], 0, 0, 0)
    # return shape:9*9
    return data_2D


def norm_dataset(dataset_1D):
    norm_dataset_1D = np.zeros([dataset_1D.shape[0], 32])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_1D[i] = feature_normalize(dataset_1D[i])
    # return shape: m*32
    return norm_dataset_1D


def feature_normalize(data):
    mean = data[data.nonzero()].mean()
    sigma = data[data.nonzero()].std()
    data_normalized = data
    data_normalized[data_normalized.nonzero()] = (data_normalized[data_normalized.nonzero()] - mean) / sigma
    # return shape: 9*9
    return data_normalized


def dataset_1Dto2D(dataset_1D):
    dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        dataset_2D[i] = data_1Dto2D(dataset_1D[i])
    # return shape: m*9*9
    return dataset_2D


def norm_dataset_1Dto2D(dataset_1D):
    norm_dataset_2D = np.zeros([dataset_1D.shape[0], 9, 9])
    for i in range(dataset_1D.shape[0]):
        norm_dataset_2D[i] = feature_normalize(data_1Dto2D(dataset_1D[i]))
    # return shape: m*9*9
    return norm_dataset_2D


def windows(data, size):
    start = 0
    while ((start + size) <= data.shape[0]):
        yield int(start), int(start + size)
        start += size


def segment_signal_without_transition(data, label, label_index, window_size):
    # get data file name and label file name
    for (start, end) in windows(data, window_size):
        # print(data.shape)
        if ((len(data[start:end]) == window_size)):
            if (start == 0):
                segments = data[start:end]
                # segments = np.vstack([segments, data[start:end]])

                labels = np.array(label[label_index])
                # labels = np.append(labels, np.array(label[label_index]))
            else:
                segments = np.vstack([segments, data[start:end]])
                labels = np.append(labels, np.array(
                    label[label_index]))  # labels = np.append(labels, stats.mode(label[start:end])[0][0])
    return segments, labels


def apply_mixup(dataset_file, window_size, label, yes_or_not):  # initial empty label arrays
    print("Processing", dataset_file, "..........")
    data_file_in = scio.loadmat(dataset_file)
    data_in = data_file_in["data"].transpose(0, 2, 1)
    # 0 valence, 1 arousal, 2 dominance, 3 liking
    if label == "Arousal":
        label = 1
    elif label == "Valence":
        label = 0
    elif label == 'Dominance':
        label = 2
    label_in = (data_file_in["labels"][:, label] > 5).astype(int)

    trials = data_in.shape[0]
    feature_out = {}
    label_out = {}

    # Data pre-processing
    for trial in range(trials):
        if yes_or_not == "yes":
            base_signal = (data_in[trial, 0:128, 0:32] + data_in[trial, 128:256, 0:32] + data_in[trial, 256:384,
                                                                                         0:32]) / 3
        else:
            base_signal = 0
        data = data_in[trial, 384:8064, 0:32]
        # compute the deviation between baseline signals and experimental signals
        for i in range(0, 60):
            data[i * 128:(i + 1) * 128, 0:32] = data[i * 128:(i + 1) * 128, 0:32] - base_signal
        label_index = trial
        # read data and label
        data = norm_dataset(data)
        data, label = segment_signal_without_transition(data, label_in, label_index, window_size)
        # cnn data process
        # data_cnn = dataset_1Dto2D(data)
        # data_cnn = data_cnn.reshape(int(data_cnn.shape[0] / window_size), window_size, 9, 9)
        # rnn data process
        data_rnn = data.reshape(int(data.shape[0] / window_size), window_size, 32)
        # append new data and label
        test = data_rnn[1,:,1].reshape(-1)
        feature_out[f"trial{trial+1}"] = data_rnn
        label_out[f"trial{trial+1}"] = label

    '''
    print("total cnn size:", data_inter_cnn.shape)
    print("total rnn size:", data_inter_rnn.shape)
    print("total label size:", label_inter.shape)
    '''

    # index = np.array(range(0, len(label_inter)))
    save_data = {'feature': feature_out, 'label': label_out}
    return save_data


def preprocessed_DEAP(dataset):
    begin = time.time()
    print("time begin:", time.localtime())
    dataset_dir = "E:/datasets/UnPreprocessed/DEAP/physiological recordings生理记录/data_preprocessed_matlab/"
    window_size = 384
    label_class = dataset  # arousal/valence/dominance
    suffix = 'yes'  # yes/no (remove baseline signals or not)
    record_list = [task for task in os.listdir(dataset_dir) if os.path.isfile(os.path.join(dataset_dir, task))]

    for record in record_list:
        file = os.path.join(dataset_dir, record)
        save_data = apply_mixup(file, window_size, label_class, suffix)
        subject_number = int(record[1:3])
        file_name = f"subject{subject_number}.mat"

        file_path = os.path.join(f'E:/datasets/DEAP_Time_Preprocessed_384/{dataset}/', file_name)
        # scio.savemat(file_path, save_data)

        end = time.time()
        print("end time:", time.localtime())
        print("time consuming:", (end - begin))


if __name__ == '__main__':
    dataset = 'Dominance'     #'Arousal'/'Valence'/'Dominance'
    preprocessed_DEAP(dataset)