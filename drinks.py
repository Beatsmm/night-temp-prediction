import pandas as pd

'''
根据 drinks 数据集，解决以下问题： A. 哪个国家的饮酒量最⼤ B. 哪个⼤洲的饮酒量最⼤
'''
path = ('/Users/demons/Downloads/drinks.csv')
drinks = pd.read_csv(path)
drinks.head()
#A.哪个国家的饮酒量最⼤
#方式1
print(drinks.groupby('country').total_litres_of_pure_alcohol.sum().sort_values(ascending=False).head(1))
#方式2
country_max = drinks.groupby('country')['total_litres_of_pure_alcohol'].sum().idxmax()
print(country_max)
#B.哪个大洲的饮酒量最大
#方式1
print(drinks.groupby('continent').total_litres_of_pure_alcohol.sum().sort_values(ascending=False).head(1))
#方式2
continent_max = drinks.groupby('continent')['total_litres_of_pure_alcohol'].sum().idxmax()
print(continent_max)
