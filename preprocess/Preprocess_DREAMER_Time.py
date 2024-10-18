import scipy.io as scio
import os
import numpy as np
import time
import scipy.io as scio


def preprocessed_DREAMER():
    data = scio.loadmat('E:/datasets/DREAMER/DREAMER.mat')
    data = data['DREAMER']
    # 取出所有被试数据
    data = data['Data'][0][0][0]
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
            trial_baseline = sub_baseline[trial][0]
            baseline = trial_baseline.reshape(-1, 128, 14)
            baseline = np.mean(baseline, axis=0)
            trial_stimuli = sub_stimuli[trial][0]
            trial_arousal = sub_arousal[trial][0]
            trial_valence = sub_valence[trial][0]
            trial_dominance = sub_dominance[trial][0]

            # 去除基线信号
            for i in range(int(trial_stimuli.shape[0]/128)):
                trial_stimuli[i * 128:(i + 1) * 128, 0:14] = trial_stimuli[i * 128:(i + 1) * 128, 0:14] - baseline
            length += trial_stimuli.shape[0]

            trial_arousal = 0 if trial_arousal <= 3 else 1
            trial_valence = 0 if trial_valence <= 3 else 1
            trial_dominance = 0 if trial_dominance <= 3 else 1

            trial_feature = trial_stimuli.reshape(-1, 128, 14)
            feature[f'trial{trial + 1}'] = trial_feature
            arousal[f'trial{trial + 1}'] = np.repeat(trial_arousal, trial_feature.shape[0])
            valence[f'trial{trial + 1}'] = np.repeat(trial_valence, trial_feature.shape[0])
            dominance[f'trial{trial + 1}'] = np.repeat(trial_dominance, trial_feature.shape[0])

        save_data_arousal = {'feature': feature, 'label': arousal}
        save_data_valence = {'feature': feature, 'label': valence}
        save_data_dominance = {'feature': feature, 'label': dominance}

        file_name = f"subject{sub+1}.mat"
        arousal_path = os.path.join(f'E:/datasets/DREAMER_Preprocessed/Arousal/', file_name)
        valence_path = os.path.join(f'E:/datasets/DREAMER_Preprocessed/Valence/', file_name)
        dominance_path = os.path.join(f'E:/datasets/DREAMER_Preprocessed/Dominance/', file_name)
        # scio.savemat(arousal_path, save_data_arousal)
        # scio.savemat(valence_path, save_data_valence)
        # scio.savemat(dominance_path, save_data_dominance)

    print(data)



if __name__ == '__main__':
    preprocessed_DREAMER()