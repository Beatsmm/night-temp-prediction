import torch
import torch.nn as nn
import torch.optim as optim

# 定义神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear = nn.Linear(2, 1)  # 输入特征维度为2，输出特征维度为1

    def forward(self, x):
        return self.linear(x)

# 创建神经网络实例
model = NeuralNetwork()

# 定义输入数据和目标标签
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y_true = torch.tensor([[3.0], [5.0]])

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 前向传播
y_pred = model(x)

# 计算损失
loss = criterion(y_pred, y_true)

# 反向传播
optimizer.zero_grad()
loss.backward()

# 更新参数
optimizer.step()

# 输出更新后的参数
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data)