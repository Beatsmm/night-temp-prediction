import numpy as np

a = np.arange(25).reshape(5, 5)
# a[[1, 2]] = a[[2, 1]]
# a[a % 2 == 1] = -1
np.fill_diagonal(a, 0)
b = np.arange(5).reshape(1, 5)
c = np.tile(b, (5, 1))
print(c)
# print(a)

a_2d = np.array([[3,3,3],[4,4,4],[5,5,5]])
b_1d = np.array([1,2,3])

# res = a_2d - b_1d.reshape(3, 1)
res = a_2d - b_1d
print('res'+res)

import pandas as pd

data = np.random.rand(6, 4)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
print(df)

d = {
 'LiYongzhu' : [95, 96, 90, 88],
 'LiuQingyuan' : [90, 93, 95, 90],
 'GuoYimeng' : [100, 100, 100, 100]
}
df = pd.DataFrame(d, columns=['LiYongzhu', 'LiuQingyuan', 'GuoYimeng'], index=['C', 'Java', 'Python', 'JS'])
print(df)