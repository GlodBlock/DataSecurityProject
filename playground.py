import torch
from torch import argmax
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_set import get_train_test
from model import VPN_CNN

model = VPN_CNN()

train_datas, test_datas = get_train_test()

for t in range(5):
    tr = train_datas[t]
    ts = test_datas[t]
    model.load_state_dict(torch.load('./anaylsi/File.pkl'))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_set = DataLoader(tr, batch_size=2, shuffle=True)
    test_set = DataLoader(ts, batch_size=2, shuffle=False)
    acc = 0
    for img, label in tqdm(train_set):
        img = img.float().to(device)
        label = label.float().to(device)
        logits, _ = model(img, label)
        r = argmax(logits, dim=1)
        acc += torch.eq(r, label.bool()).float().sum().item()

    print(t, acc / (len(tr)))


