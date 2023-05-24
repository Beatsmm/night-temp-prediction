import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置超参数
BATCH_SIZE = 512
EPOCHS = 10
DEVICE = torch.device("cuda")
lr = 0.01

# 加载数据
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='data',
        train=True,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
             transforms.Normalize((0.1307,),(0.3801,))
        ])),
    batch_size = BATCH_SIZE,
    shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='data',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
             transforms.Normalize((0.1307,),(0.3801,))
        ])),
    batch_size = BATCH_SIZE,
    shuffle=True)

# 定义模型
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = x.view(in_size, -1)
        out = self.fc1(out)  # 输入28*28=784 输出256
        out = F.relu(out)
        out = self.fc2(out)  # 输入256 输出10
        out = F.log_softmax(out, dim=1)
        return out


model_mlp = MLP().to(DEVICE)
optimizer_mlp = optim.SGD(model_mlp.parameters(), lr=lr)

# 定义训练函数

def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data,target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# 定义测试函数
def test(model,device,test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data,target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss  += F.nll_loss(output,target,reduction='sum').item()
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCHS + 1):
    train(model_mlp, DEVICE, train_loader, optimizer_mlp, epoch)
    test(model_mlp, DEVICE, test_loader)