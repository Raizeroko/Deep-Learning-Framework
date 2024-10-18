import os
import math
import numpy as np
import scipy.io as scio
from scipy.signal import butter, lfilter


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def read_file(file):
    data = scio.loadmat(file)
    data = data['data']
    return data


def compute_DE(signal):
    variance = np.var(signal, ddof=1)
    return math.log(2 * math.pi * math.e * variance) / 2


import numpy as np


def DLM_Inference(x, para):
    u0 = np.array([para['u0']]) if np.isscalar(para['u0']) else np.array(para['u0'])
    V0 = np.eye(u0.shape[0]) * para['V0']
    A = np.eye(u0.shape[0]) * para['A'] if np.isscalar(para['A']) else para['A']
    T = np.eye(u0.shape[0]) * para['T'] if np.isscalar(para['T']) else para['T']
    C = np.eye(u0.shape[0]) * para['C'] if np.isscalar(para['C']) else para['C']
    m_sigma = np.eye(C.shape[0]) * para['sigma'] if np.isscalar(para['sigma']) else para['sigma']
    m, n = x.shape
    P = np.zeros((u0.shape[0], u0.shape[0], n))
    u = np.zeros((u0.shape[0], n))
    V = np.zeros((u0.shape[0], u0.shape[0], n))
    K = np.zeros((u0.shape[0], C.shape[0], n))

    # Initialize state
    u[:, 0] = u0.flatten()
    V[:, :, 0] = V0

    for i in range(1, n):
        # Prediction step
        u_pred = np.dot(A, u[:, i - 1])
        V_pred = np.dot(A, np.dot(V[:, :, i - 1], A.T)) + T

        # Update step
        S = np.dot(C, np.dot(V_pred, C.T)) + m_sigma
        K[:, :, i] = np.dot(np.dot(V_pred, C.T), np.linalg.pinv(S))
        u[:, i] = u_pred + np.dot(K[:, :, i], (x[:, i] - np.dot(C, u_pred)))
        V[:, :, i] = np.dot(np.eye(u0.shape[0]) - np.dot(K[:, :, i], C), V_pred)

    Y = {'z': u, 'P': P, 'u': u, 'V': V, 'K': K}

    if 'givenAll' in para and para['givenAll'] == 1:
        uAll = np.zeros((u0.shape[0], n))
        VAll = np.zeros((u0.shape[0], u0.shape[0], n))
        J = np.zeros((u0.shape[0], u0.shape[0], n))

        uAll[:, n - 1] = u[:, n - 1]
        VAll[:, :, n - 1] = V[:, :, n - 1]

        for i in range(n - 2, -1, -1):
            J[:, :, i] = np.dot(V[:, :, i], np.dot(A.T, np.linalg.pinv(V_pred)))
            uAll[:, i] = u[:, i] + np.dot(J[:, :, i], (uAll[:, i + 1] - np.dot(A, u[:, i])))
            VAll[:, :, i] = V[:, :, i] + np.dot(np.dot(J[:, :, i], (VAll[:, :, i + 1] - np.dot(A, V[:, :, i]))),
                                                J[:, :, i].T)

        Y['uAll'] = uAll
        Y['J'] = J
        Y['VAll'] = VAll

    return Y


def lds(sequence):
    n, l, k = sequence.shape
    sequence_new = np.zeros((n, l, k))

    # Calculate average for each channel and frequency band
    ave = np.mean(sequence, axis=1)  # Shape: (n, k)

    for i in range(n):
        for j in range(k):
            X = sequence[i, :, j]
            X = X.reshape((40, 15))  # Reshape if necessary for your data
            para = {
                'u0': ave[i, j],
                'V0': 0.1,
                'A': 1,
                'T': 0.001,
                'C': np.eye(40),  # Observation matrix C
                'sigma': 1,  # Observation noise covariance
                'givenAll': 1
            }
            Y = DLM_Inference(X, para)
            X_smoothed = Y['z']
            sequence_new[i, :, j] = X_smoothed.flatten()

    return sequence_new

def decompose(file):
    start_index = 384  # 3s pre-trial signals
    data = read_file(file)
    frequency = 128
    segment_length = 3 * frequency  # 384

    all_trials_features = []
    for trial in range(40):
        trial_features = np.zeros((32, 20, 5))
        trial_signal = data[trial, :, 3 * 128:]
        for channel in range(32):
            delta = butter_bandpass_filter(trial_signal[channel, :], 1, 4, frequency, order=3)
            theta = butter_bandpass_filter(trial_signal[channel, :], 4, 7, frequency, order=3)
            alpha = butter_bandpass_filter(trial_signal[channel, :], 8, 14, frequency, order=3)
            beta = butter_bandpass_filter(trial_signal[channel, :], 14, 31, frequency, order=3)
            gamma = butter_bandpass_filter(trial_signal[channel, :], 31, 45, frequency, order=3)

            DE_delta = []
            DE_theta = []
            DE_alpha = []
            DE_beta = []
            DE_gamma = []

            for index in range(20):
                start = index * segment_length
                end = (index + 1) * segment_length

                segment1 = delta[start:end]
                segment2 = theta[start:end]
                segment3 = alpha[start:end]
                segment4 = beta[start:end]
                segment5 = gamma[start:end]

                de_1 = compute_DE(segment1)
                de_2 = compute_DE(segment2)
                de_3 = compute_DE(segment3)
                de_4 = compute_DE(segment4)
                de_5 = compute_DE(segment5)

                DE_delta.append(de_1)
                DE_theta.append(de_2)
                DE_alpha.append(de_3)
                DE_beta.append(de_4)
                DE_gamma.append(de_5)

            trial_features[channel, :, 0] = DE_delta
            trial_features[channel, :, 1] = DE_theta
            trial_features[channel, :, 2] = DE_alpha
            trial_features[channel, :, 3] = DE_beta
            trial_features[channel, :, 4] = DE_gamma
        # all_trials_features[f'trial{trial + 1}'] = trial_features
        all_trials_features.append(trial_features)

    all_trials_features = np.array(all_trials_features)#(40, 32, 20, 5)
    print("all_trials_features ", all_trials_features.shape)
    all_trials_features = np.swapaxes(all_trials_features, 0, 1)#all_trials_features  (32, 40, 20, 5)
    print("all_trials_features ",all_trials_features.shape)
    # 首先，将第 2 和第 3 维度展开为一维
    # flat_features = all_trials_features.reshape(all_trials_features.shape[0], -1, all_trials_features.shape[-1])
    # print("flat_features",flat_features.shape)#(32, 600, 5)
    # Perform LDS smoothing on the features
    #smoothed_features = lds(flat_features)

    return all_trials_features

def save_features(file, original_features, smoothed_features):
    scio.savemat(file, {
        'original_features': original_features,
        'smoothed_features': smoothed_features
    })


def get_labels(file):
    #0 valence, 1 arousal, 2 dominance, 3 liking
    valence_labels = scio.loadmat(file)["labels"][:,0]	# valence labels
    arousal_labels = scio.loadmat(file)["labels"][:,1]# arousal labels
    print("v",valence_labels)
    print("a", arousal_labels)
    for i in range(len(valence_labels)):
        if valence_labels[i] > 5:
            valence_labels[i] =1
        else :
            valence_labels[i]=0

        if arousal_labels[i] > 5:
            arousal_labels[i] =1
        else :
            arousal_labels[i]=0

    print("v2", valence_labels)
    print("a2", arousal_labels)
    # assert 1==0
    final_valence_labels = {}
    final_arousal_labels = {}
    print("len(valence_labels),len(arousal_labels)",len(valence_labels),len(arousal_labels))
    for i in range(len(valence_labels)):
        trial_valence_labels = np.empty([0])
        trial_arousal_labels = np.empty([0])
        for j in range(0, 20):
            trial_valence_labels = np.append(trial_valence_labels,valence_labels[i])
            trial_arousal_labels = np.append(trial_arousal_labels,arousal_labels[i])
        final_arousal_labels[f'trial{i+1}'] = trial_valence_labels
        final_valence_labels[f'trial{i+1}'] = trial_arousal_labels

    return final_arousal_labels, final_valence_labels

def moving_average(data, window_size):
    padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
    return np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')

def apply_moving_average(features, window_size):
    smoothed_features = np.zeros(features.shape)  # Maintain the same shape as the input
    for channel in range(features.shape[0]):
        for feature in range(features.shape[2]):
            smoothed_features[channel, :, feature] = moving_average(features[channel, :, feature], window_size)
    return smoothed_features


if __name__ == '__main__':
    dataset_dir = "E:/datasets/DEAP/physiological recordings生理记录/data_preprocessed_matlab"
    result_dir = "/home/pamL/PycharmProjects/deap_DE6/"
    os.makedirs(result_dir, exist_ok=True)

    for file in os.listdir(dataset_dir):
        file_path = os.path.join(dataset_dir, file)
        subject_number = file[1:3]  # 取出 '01'
        subject_number = int(subject_number)
        print(f"Processing: {file_path} ...")
        features = decompose(file_path)

        # Apply moving average to the features
        # window_size = 5
        # smoothed_features = apply_moving_average(flat_features, window_size)

        # print("smoothed_features",features.shape)
        # features = flat_features.reshape(flat_features.shape[0], 40, 20, flat_features.shape[-1])
        # 交换前两维
        all_features = features.transpose(1, 2, 3, 0)
        print("features", all_features.shape)
        feature = {}
        for i in range(all_features.shape[0]):
            feature[f'trial{i+1}'] = all_features[i]
        arousal_label, valence_label = get_labels(file_path)

        arousal_data = {'feature': feature, 'label': arousal_label}
        valence_data = {'feature': feature, 'label': valence_label}
        file_name = f"subject{subject_number}.mat"
        print("test")
        arousal_path = os.path.join('E:/datasets/DEAP_DE_Preprocessed_384/Arousal', file_name)
        valence_path = os.path.join('E:/datasets/DEAP_DE_Preprocessed_384/Valence', file_name)
        scio.savemat(arousal_path, arousal_data)
        scio.savemat(valence_path, valence_data)


