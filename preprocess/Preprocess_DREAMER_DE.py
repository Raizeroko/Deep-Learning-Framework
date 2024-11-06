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
    arousal_labels = scio.loadmat(file)["labels"][:,1] # arousal labels
    dominance_labels = scio.loadmat(file)["labels"][:,2]
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
        if dominance_labels[i] > 5:
            dominance_labels[i] = 1
        else:
            dominance_labels[i] = 0

    print("v2", valence_labels)
    print("a2", arousal_labels)
    # assert 1==0
    final_valence_labels = {}
    final_arousal_labels = {}
    final_dominance_labels = {}
    print("len(valence_labels),len(arousal_labels)",len(valence_labels),len(arousal_labels))
    for i in range(len(valence_labels)):
        trial_valence_labels = np.empty([0])
        trial_arousal_labels = np.empty([0])
        trial_dominance_labels = np.empty([0])
        for j in range(0, 20):
            trial_valence_labels = np.append(trial_valence_labels,valence_labels[i])
            trial_arousal_labels = np.append(trial_arousal_labels,arousal_labels[i])
            trial_dominance_labels = np.append(trial_dominance_labels,dominance_labels[i])
        final_arousal_labels[f'trial{i+1}'] = trial_valence_labels
        final_valence_labels[f'trial{i+1}'] = trial_arousal_labels
        final_dominance_labels[f'trial{i+1}'] = trial_dominance_labels

    return final_arousal_labels, final_valence_labels, final_dominance_labels

def moving_average(data, window_size):
    padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
    return np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')

def apply_moving_average(features, window_size):
    smoothed_features = np.zeros(features.shape)  # Maintain the same shape as the input
    for channel in range(features.shape[0]):
        for feature in range(features.shape[2]):
            smoothed_features[channel, :, feature] = moving_average(features[channel, :, feature], window_size)
    return smoothed_features

def preprocessed_DREAMER():
    data = scio.loadmat('E:/datasets/UnPreprocessed/DREAMER/DREAMER.mat')
    data = data['DREAMER']
    window_size = 384
    # 取出所有被试数据
    data = data['Data'][0][0][0]

    frequency = 128
    segment_length = 3 * frequency  # 384

    # 23个被试
    for sub in range(23):
        sub_data = data[sub]
        sub_eeg = sub_data['EEG'][0][0]
        sub_baseline = sub_eeg['baseline'][0][0]
        sub_stimuli = sub_eeg['stimuli'][0][0]
        sub_arousal = sub_data['ScoreArousal'][0][0]
        sub_valence = sub_data['ScoreValence'][0][0]
        sub_dominance = sub_data['ScoreDominance'][0][0]
        # 18个trial
        length = 0
        feature = {}
        arousal = {}
        valence = {}
        dominance = {}
        for trial in range(18):
            trial_stimuli = sub_stimuli[trial][0]
            trial_arousal = sub_arousal[trial][0]
            trial_valence = sub_valence[trial][0]
            trial_dominance = sub_dominance[trial][0]

            trial_arousal = 0 if trial_arousal <= 3 else 1
            trial_valence = 0 if trial_valence <= 3 else 1
            trial_dominance = 0 if trial_dominance <= 3 else 1

            T = trial_stimuli.shape[0]//window_size
            trial_stimuli = trial_stimuli[:T*window_size, :]
            trial_stimuli = trial_stimuli.transpose(1, 0)

            trial_features = np.zeros((14, T, 5))

            for channel in range(14):
                delta = butter_bandpass_filter(trial_stimuli[channel, :], 1, 4, frequency, order=3)
                theta = butter_bandpass_filter(trial_stimuli[channel, :], 4, 7, frequency, order=3)
                alpha = butter_bandpass_filter(trial_stimuli[channel, :], 8, 14, frequency, order=3)
                beta = butter_bandpass_filter(trial_stimuli[channel, :], 14, 31, frequency, order=3)
                gamma = butter_bandpass_filter(trial_stimuli[channel, :], 31, 45, frequency, order=3)

                DE_delta = []
                DE_theta = []
                DE_alpha = []
                DE_beta = []
                DE_gamma = []

                for index in range(T):
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


            # trial_feature = trial_stimuli.reshape(-1, window_size, 14)
            print(trial_features)
            trial_features = trial_features.transpose(1, 2, 0)
            feature[f'trial{trial + 1}'] = trial_features
            arousal[f'trial{trial + 1}'] = np.repeat(trial_arousal, trial_features.shape[0])
            valence[f'trial{trial + 1}'] = np.repeat(trial_valence, trial_features.shape[0])
            dominance[f'trial{trial + 1}'] = np.repeat(trial_dominance, trial_features.shape[0])

        save_data_arousal = {'feature': feature, 'label': arousal}
        save_data_valence = {'feature': feature, 'label': valence}
        save_data_dominance = {'feature': feature, 'label': dominance}

        file_name = f"subject{sub+1}.mat"
        arousal_path = os.path.join(f'E:/datasets/DREAMER_DE_Preprocessed_384/Arousal/', file_name)
        valence_path = os.path.join(f'E:/datasets/DREAMER_DE_Preprocessed_384/Valence/', file_name)
        dominance_path = os.path.join(f'E:/datasets/DREAMER_DE_Preprocessed_384/Dominance/', file_name)
        scio.savemat(arousal_path, save_data_arousal)
        scio.savemat(valence_path, save_data_valence)
        scio.savemat(dominance_path, save_data_dominance)

    print(data)


if __name__ == '__main__':
    preprocessed_DREAMER()


