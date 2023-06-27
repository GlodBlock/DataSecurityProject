import torch
from torch import optim, argmax
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, tr, ts):
    # 设置cuda/cpu环境
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    # 装载数据集
    train_set = DataLoader(tr, batch_size=2, shuffle=True)
    test_set = DataLoader(ts, batch_size=2, shuffle=False)
    bst_record = 0

    flow_info = 'epoch,train_acc,loss,test_acc,precision,recall,f1\n'

    for epoch in range(30):
        # 训练
        loss_val = 0
        # 真正例
        tt = 0
        # 真负例
        tf = 0
        # 假正例
        ft = 0
        # 假负例
        ff = 0
        model.train()
        for img, label in tqdm(train_set):
            optimizer.zero_grad()
            img = img.float().to(device)
            label = label.float().to(device)
            logits, loss = model(img, label)
            loss.backward()
            optimizer.step()
            r = argmax(logits, dim=1)
            e = torch.eq(r, label.bool())
            ne = torch.not_equal(r, label.bool())
            tt += torch.where(label > 0, e, 0).float().sum().item()
            ft += torch.where(label > 0, ne, 0).float().sum().item()
            tf += torch.where(label <= 0, e, 0).float().sum().item()
            ff += torch.where(label <= 0, ne, 0).float().sum().item()
            loss_val += loss.item()
        flow_info += f'{epoch + 1},{(tt + tf)/(tt + tf + ft + ff)},{loss_val/(tt + tf + ft + ff)},'
        print(f'Epoch {epoch + 1}: Train ACC {(tt + tf)/(tt + tf + ft + ff)} | Loss {loss_val/(tt + tf + ft + ff)}')

        # 测试
        tt = 0
        tf = 0
        ft = 0
        ff = 0
        model.eval()
        for img, label in tqdm(test_set):
            img = img.float().to(device)
            label = label.float().to(device)
            logits, _ = model(img, label)
            r = argmax(logits, dim=1)
            e = torch.eq(r, label.bool())
            ne = torch.not_equal(r, label.bool())
            tt += torch.where(label > 0, e, 0).float().sum().item()
            ft += torch.where(label > 0, ne, 0).float().sum().item()
            tf += torch.where(label <= 0, e, 0).float().sum().item()
            ff += torch.where(label <= 0, ne, 0).float().sum().item()
        pr = tt / (tt + ft)
        rc = tt / (tt + ff)
        flow_info += f'{(tt + tf)/(tt + tf + ft + ff)},{pr},{rc},{(2 * pr * rc)/(pr + rc)}\n'
        print(f'Epoch {epoch + 1}: Tess ACC {(tt + tf)/(tt + tf + ft + ff)}')

        # 检查模型是否提升
        if (tt + tf)/(tt + tf + ft + ff) > bst_record:
            # 更新checkpoint
            bst_record = (tt + tf)/(tt + tf + ft + ff)
            torch.save(model.state_dict(), "bst.pkl")
        else:
            # 回滚
            model.load_state_dict(torch.load("bst.pkl"))

    return flow_info
