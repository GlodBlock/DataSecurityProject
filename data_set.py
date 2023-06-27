import random

import dill as dill
import numpy as np
import torch
from torch.utils.data import Dataset

# 窗口大小
ITV = 30
# 包数量下限
MIS = 10
one_hot = {'VoIP': 0, 'Chat': 1, 'Email': 2, 'Streaming': 3, 'File': 4}


# 根据时间戳和大小生成FlowPic
def gen_img(time, payload):
    # 时间戳归一化
    time -= min(time)
    time = time / (max(time) + 0.0001) * 32
    time = np.clip(time, 0, 31)
    # 包大小归一化
    payload -= min(payload)
    payload = payload / (max(payload) + 0.0001) * 32
    payload = np.clip(payload, 0, 31)
    img = np.zeros((32, 32))
    for index in range(len(time)):
        img[int(time[index]), int(payload[index])] += 1
    return (img-np.min(img) + 0.0001)/(np.max(img)-np.min(img) + 0.0001)


# 数据集类
class VPNDataSet(Dataset):

    def __init__(self, data=None):
        self.inputs = []
        self.labels = []
        self.data = data

    # 将单向流安装30s进行切分，对每个块生成FlowPic
    def push(self, time_stamp, payload_size, ty, vpn):
        if len(time_stamp) < MIS:
            return
        for index in range(len(time_stamp)):
            time_stamp[index] = abs(time_stamp[index])
            payload_size[index] = abs(payload_size[index])
        index = 0
        while index < len(time_stamp):
            time_clip = []
            payload_clip = []
            start = time_stamp[index]
            while index < len(time_stamp) and start + ITV > time_stamp[index]:
                time_clip.append(time_stamp[index])
                payload_clip.append(payload_size[index])
                index += 1
            time_clip = np.array(time_clip)
            payload_clip = np.array(payload_clip)
            # 每个块中至少有10个包
            if len(time_clip) > MIS:
                self.inputs.append(gen_img(time_clip, payload_clip))
                self.labels.append((one_hot[ty], vpn))

    def __getitem__(self, index):
        if self.data is None:
            return self.inputs[index], self.labels[index]
        else:
            return self.data[index]

    def __len__(self):
        if self.data is None:
            return len(self.inputs)
        else:
            return len(self.data)


# 生成FlowPic
def gen_dataset():
    flow_info = open("flow.txt", "r")
    dataset = VPNDataSet()
    i = 0
    while 1:
        flow_raw = flow_info.readline().split(";")
        i += 1
        if flow_raw is None or len(flow_raw) <= 2:
            break
        dataset.push(eval(flow_raw[5]), eval(flow_raw[6]), flow_raw[7], eval(flow_raw[8]))
        print(f'{i / 9549 * 100}%')
    # 保存数据集
    torch.save(dataset.inputs, 'data1.pkl', pickle_module=dill)
    torch.save(dataset.labels, 'data2.pkl', pickle_module=dill)


# 把数据集切分成训练集和测试集
def get_train_test():
    all_sets = [[], [], [], [], []]
    train_sets = []
    test_sets = []
    inputs = torch.load('data1.pkl')
    outputs = torch.load('data2.pkl')

    for i in range(len(inputs)):
        ty = outputs[i][0]
        all_sets[ty].append((np.array([inputs[i]]), outputs[i][1]))

    # 打乱顺序
    random.shuffle(all_sets[0])
    random.shuffle(all_sets[1])
    random.shuffle(all_sets[2])
    random.shuffle(all_sets[3])
    random.shuffle(all_sets[4])

    # 切分
    tr = all_sets[0][0:8000]
    te = all_sets[0][8000:8608]
    train_sets.append(VPNDataSet(tr))
    test_sets.append(VPNDataSet(te))
    tr = all_sets[1][0:900]
    te = all_sets[1][900:1092]
    train_sets.append(VPNDataSet(tr))
    test_sets.append(VPNDataSet(te))
    tr = all_sets[2][0:750]
    te = all_sets[2][750:858]
    train_sets.append(VPNDataSet(tr))
    test_sets.append(VPNDataSet(te))
    tr = all_sets[3][0:850]
    te = all_sets[3][850:914]
    train_sets.append(VPNDataSet(tr))
    test_sets.append(VPNDataSet(te))
    tr = all_sets[4][0:900]
    te = all_sets[4][900:1091]
    train_sets.append(VPNDataSet(tr))
    test_sets.append(VPNDataSet(te))
    return train_sets, test_sets
