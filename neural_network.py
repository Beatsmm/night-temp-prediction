import numpy as np


def logistic(y):
    return 1 / (1 + np.exp(-y))


def sgn(x):
    return np.where(x >= 0, 1, 0)

p = np.array([(0, 0), (1, 0), (0, 1), (1, 1)])
w1, w2, b = 1, -1, 0.5

def all_connection(x):
    y1 = np.dot(w1, x[0]) + np.dot(w2, x[1]) - b # dot对应元素相乘在求和
    y1 = sgn(y1)
    y2 = np.dot(w1, x[1]) + np.dot(w2, x[0]) -b
    y2 = sgn(y2)
    y = np.dot(w1, y1) + np.dot(w1, y2) - b
    y = sgn(y)
    return y

for input in p:
    output = all_connection(input)
    print(output)

import torch

# 创建一个三行四列的随机数填充的tensor变量
tensor_variable = torch.randn(3, 4)

# 通过属性获取tensor变量的形状
shape = tensor_variable.shape

print(tensor_variable)
print(shape)