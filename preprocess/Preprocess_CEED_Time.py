import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import scipy.io as scio

def print_base_information(raw):
    print('---------------------------------------------------')
    # 通道图
    # raw.plot_sensors(show_names=True)
    # # 脑电图
    # raw.plot(scalings='auto')
    # plt.show(block=True)
    # 打印数据基本信息
    print(f"信息: {raw.info}")
    print(f"通道数: {len(raw.ch_names)}")

    # 提取数据和时间
    data, times = raw.get_data(return_times=True)
    print(f"数据形状: {data.shape}")
    print(f"时间点: {times.shape}")

    # 提取标记事件
    events, event_id = mne.events_from_annotations(raw)
    print(f"事件: {events}")
    print(f"事件描述: {event_id}")

    print('---------------------------------------------------')
    return events, event_id

def read_brainvision_data(vhdr_file):
    """
    读取 BrainVision 数据 (.vhdr, .eeg, .vmrk)
    """
    # 读取 .vhdr 文件（会自动加载 .eeg 和 .vmrk 文件）
    raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

    # print_base_information(raw)
    # 去除坏道

    # 1. 带通滤波：4-47 Hz
    raw.filter(l_freq=4.0, h_freq=47.0)

    # 2. 平均重参考
    raw.set_eeg_reference('average', projection=True)
    # 确保投影被应用
    raw.apply_proj()

    # 3. 降采样至 128 Hz
    raw.resample(128)

    events, event_id = print_base_information(raw)
    return raw, events, event_id


def extract_trials(raw, events, event_id):
    window_size = 384
    """
    提取从 'Stimulus/S  1', 'Stimulus/S  2', 'Stimulus/S  3' 到 'Stimulus/S  5' 的脑电片段
    """
    # 定义开始和结束标记
    start_codes = [event_id['Stimulus/S  1'],
                   event_id['Stimulus/S  2'],
                   event_id['Stimulus/S  3']]
    end_code = event_id['Stimulus/S  5']

    # 存储 trial 数据
    trial_dict = {}  # 使用字典存储每个trial

    trial_labels = {}  # 存储每个 trial 的标签（1、2 或 3）

    # 遍历事件
    for i, (time, _, event_code) in enumerate(events):
        if event_code in start_codes:  # 找到一个 trial 的开始标记
            # 查找对应的结束标记
            for j in range(i + 1, len(events)):
                if events[j, 2] == end_code:  # 找到结束标记
                    start_time = time / raw.info['sfreq']  # 转为秒
                    end_time = events[j, 0] / raw.info['sfreq']  # 转为秒

                    # 提取时间段数据
                    trial_data = raw.copy().crop(tmin=start_time, tmax=end_time).get_data()

                    # 提取基线信号
                    baseline_data = raw.copy().crop(tmin=start_time - 3.0, tmax=start_time).get_data()
                    # 385长度变384
                    baseline_data = baseline_data[:, 1:]
                    # baseline_dict[f'trial{len(baseline_dict) + 1}'] = baseline_data
                    base_signal = (baseline_data[:, 0:128] + baseline_data[:, 128:256] + baseline_data[:, 256:384]) / 3
                    for i in range(int(trial_data.shape[0] / 128)):
                        trial_data[:, i * 128:(i + 1) * 128] = trial_data[:, i * 128:(i + 1) * 128] - base_signal

                    # 转为µV单位
                    trial_data = trial_data*1e6

                    T = trial_data.shape[1] // window_size
                    trial_data = trial_data[:, :T * window_size]
                    trial_data = trial_data.reshape(trial_data.shape[0], -1, 128 * 3).transpose(1, 2, 0)

                    # 将数据存入字典，键名为 'trial{i}'
                    trial_labels[f'trial{len(trial_dict) + 1}'] = np.repeat(event_code-1, trial_data.shape[0])
                    trial_dict[f'trial{len(trial_dict) + 1}'] = trial_data

                    break

    print(f"提取了 {len(trial_dict)} 个 trial")
    save_data = {'feature': trial_dict, 'label': trial_labels}
    return save_data

if __name__ == '__main__':
    window_size = 384
    # 示例调用
    vhdr_file = "E:/datasets/UnPreprocessed/CH（陈浩）/CH20230718.vhdr"
    preprocess_data, events, event_id = read_brainvision_data(vhdr_file)
    save_data = extract_trials(preprocess_data, events, event_id)
    file_path = f"E:/datasets/CEED_Time_Preprocessed_{window_size}/Subject1.mat"
    scio.savemat(file_path, save_data)


