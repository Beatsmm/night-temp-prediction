import os
import torch
import torchvision
from torchvision.transforms import Compose,ToTensor,Normalize
from torch.utils.data import DataLoader
from torch import nn, optim
import torch.nn.functional as F
import numpy as np

TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000

# 获取dataloader
def get_dataloader(train=True,batch_size=TRAIN_BATCH_SIZE):
    # 图片处理
    transform_fn = Compose([
        ToTensor(),  # 转换成tensor，（1X28X28）
        Normalize(   # 归一化处理加入均值和标准差，因为是单通道所依元组里边元素个数为1
            (0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root="./data",train=train,download=True,transform=transform_fn)
    return DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)


# data = get_dataloader()
# for i in enumerate(data):
#     print(i)
#     break
#
# print(len(data))  # 查看数据总批次

class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 28)  # 构建网络层
        self.fc2 = nn.Linear(28, 10)

    def forward(self, data):  # data的形状：[batch_size,1,28,28]
        features = data.view(data.size(0), 1*28*28)  # 需要变形成 （批次图片数量，特征数）
        features = self.fc1(features)  # (batch_size,28)
        features = F.relu(features)  # (batch_size,28)
        out = self.fc2(features)    # (batch_size,10)
        # return out
        return F.log_softmax(out, dim=-1)


model = ImageNet()  # 模型实例化
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器
# 假设已有训练好的模型或者优化器权值
if os.path.exists('./models/model.pkl'):
    model.load_state_dict(torch.load('./models/model.pkl'))
    optimizer.load_state_dict(torch.load('./models/optimizer.pkl'))

# criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数


# 模型训练
def train(epoch):
    mode = True
    model.train(mode=mode)
    train_dataloader = get_dataloader(train=mode)
    for idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)  # 调用模型得到预测值
        # loss = criterion(output,target) #对数似然损失
        loss = F.nll_loss(output,target)  # 得到损失
        loss.backward()  # 反向传播
        optimizer.step()  # 权值更新
        if idx % 10 == 0:
            print('第%d轮次,损失值为%f' % (epoch, loss.item()))

        # 模型保存
        if idx % 100 == 0:
            torch.save(model.state_dict(), './models/model.pkl')  # 模型保存
            torch.save(optimizer.state_dict(), './models/optimizer.pkl')  # 权值保存


# 模型评估
def test():
    test_loss = []
    correct = []
    model.eval()
    test_dataloader = get_dataloader(train=False, batch_size=TEST_BATCH_SIZE)
    with torch.no_grad():  # 不对操作进行跟踪记录
        for data, target in test_dataloader:
            output = model(data)
            test_loss.append(F.nll_loss(output, target))  # 损失列表
            pred = output.data.max(dim=1)[1]  # 获取最大值的位置,[batch_size,1]
            correct.append(pred.eq(target).float().mean())  # 每个批次的平均准确率
    print('模型损失%f,平均准确率%f' % (np.mean(test_loss), np.mean(correct)))


if __name__ == '__main__':
	# 训练5轮次
    for i in range(5):
        train(i)
    test()
