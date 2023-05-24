import pandas as pd
import matplotlib.pyplot as plt
'''
根据 babynames 数据集，解决以下问题： A. 分别统计男⽣和⼥⽣的出⽣⼈数 B. 有多少个不重复的名字 C. 横坐标为年份，纵坐标为数量，男⼥不同的⾛势图
'''
names = ['name', 'sex', 'births']
# names1880 = pd.read_csv('/Users/demons/Downloads/names/yob1880.txt', names=names)
# print(names1880[:10])
years = range(1880, 2021)
pieces = []
# 通过for循环将141个文件读入
for year in years:
    path = '/Users/demons/Downloads/names/yob%d.txt' % year
    frame = pd.read_csv(path, names=names)
    frame['year'] = year
    pieces.append(frame)
names = pd.concat(pieces, ignore_index=True)
# print(names)
# 1、分别统计男⽣和⼥⽣的出⽣⼈数
birth_by_sex = names.groupby('sex')['births'].sum()
print(birth_by_sex)
# 2、有多少个不同的名字
num_unique_names = names['name'].nunique()
print(num_unique_names)
# 3、横坐标为年份，纵坐标为数量，男⼥不同的⾛势图
birth_by_year_sex = names.groupby(['year', 'sex'])['births'].sum().unstack().plot()
birth_by_year_sex.plot(title = 'Number of Births by Year and Sex')

plt.xlabel('Year')
plt.ylabel('Number of births')
plt.show()
