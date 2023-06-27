import train
from data_set import get_train_test
from model import VPN_CNN

if __name__ == "__main__":
    # 初始化模型
    model = VPN_CNN()
    # 加载数据集
    train_datas, test_datas = get_train_test()
    # 训练
    log = train.train(model, train_datas[4], test_datas[4])
    # 输出训练日志
    with open('record.csv', 'w') as f:
        f.write(log)
