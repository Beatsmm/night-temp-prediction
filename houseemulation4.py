import pandas as pd
import numpy as np
from keras import models
from keras import layers
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau


# keras.backend.clear_session()
np.seterr(divide='ignore', invalid='ignore')
np.set_printoptions(precision=0, threshold=999999999, formatter={'all': lambda x: str(x)})
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)


# x = 10 line env data(t .... t-10)  y = temperature (t)
def series_to_supervised(data):
    totalday = 0 # 总共天数
    step = 0 # 一天数据记录数

    for j, data_row in enumerate(data):
        if (data_row[0] == 6): # 即将新的一天开始
            totalday +=1
            if step == 0 :
                step = j + 1 # 第一天数据记录数

    target_col_size = data.shape[-1]  # 特征数

    print("totalday:", totalday , "step:" , step)
    # zeros 初始化创建多维数组,并且把所有元素初始化为0
    samples = np.zeros((totalday, step, target_col_size)) # 存储输入数据的样本
    targets = np.zeros((totalday, step))   # 目标值

    day_row_index = 0
    y_index = 4
    for j, data_row in enumerate(data):
        if (data_row[0] == 6):
            time_batch_x = np.zeros((step, target_col_size)) # 零矩阵
            time_batch_y = np.zeros((step,)) # step大小的零向量
            for n in range(step):
                time_batch_x[n] = data[j - (step - n -1)][:]
                time_batch_y[n] = time_batch_x[n][y_index]
                if n >= 1:
                    time_batch_x[n][y_index] = 0

            samples[day_row_index] = time_batch_x
            targets[day_row_index] = time_batch_y
            day_row_index += 1 # a new day,  新的一天起始从头计数，一天的第0行

    return samples, targets


#data title  时间，小时，户外温度，风速，湿度，棚内温度
data_file = "./data-0720401900000951.txt"
model_file = "./data-0720401900000951.h5"
'''
读取的数据文件以逗号分割,没有列标题,在读取数据后,第一列被删除,这是通过Pandas库中的drop实现,然后调用dropna函数来删除任何带有NaN值的行,
然后把处理后的数据存储在名为source_data的变量中,在处理数据后,它被转换为浮点型,训练数据和测试数据被划分,其中训练数据包括前150行的数据,
测试数据包括前150行之后的所有数据,这些数据是从source_data中提取的,train_data是训练数据的输入,train_targets是训练数据的目标值,predict_data
是测试数据的输入,actual_target是测试数据的目标值.接下来,使用series_to_supervised()函数将数据转换为监督学习
'''
df = pd.read_table(data_file, header=None, sep=',') # header=None, 表示数据文件中没有表头,所以数据的第一行也会被读取,sep=,表示数据文件中的数据以逗号分隔
df.drop(df.columns[[0]], axis=1,inplace=True) # drop time column
df.dropna(inplace=True)

trainStart = 0
tranEnd = 150  # train data 2021-12-01 to 2022-03-07


test_start = tranEnd  # test data two days 2022-03-08, 2022-03-09
test_end = len(df) # 数据集的长度

source_data = df.values
source_data = source_data.astype('float32')

# 训练数据
train_data = source_data[trainStart:tranEnd, :]
# 测试数据
train_targets = source_data[trainStart:tranEnd, -1]

predict_data = source_data[test_start:test_end, :]
actual_target = source_data[test_start:test_end, -1]


(train_sample, train_target) = series_to_supervised(train_data)

(predict_sample, actual_target) = series_to_supervised(predict_data)

print(df.shape)
print(train_sample.shape)
print(train_target.shape)
print(predict_sample.shape)

'''
创建一个包含两个回调函数的列表,用于在训练深度学习模型时进行模型调整和优化,
第一个回调函数是EarlyStopping,当监控指标mse在连续的patience轮训练中没有改善的时候,会停止训练过程,min_delta参数指定了所需的最小改善程度,避免过早停止训练
第二个回调函数是ReduceLROnPlateau,它可以在监控指标不再改善的时候动态的减小学习率,以帮助模型找到更优秀的局部最小值,factor参数指定了每次减小学习率的因子,
而patience参数指定了需要连续多少轮训练没有改善才会降低学习率
'''
callbacks_list = [
    EarlyStopping(
        monitor='mse',
        patience=10,
        mode='auto',
        min_delta=0.001
    ), ReduceLROnPlateau(
        monitor='mse',
        factor=0.1,
        patience=10,
        mode='auto'
    )
]

'''
下面代码使用Keras API构建一个具有两层的序列模型,并且使用方差作为损失函数
第一层是具有64个单元的门控递归单元GRU层,这是一种递归神经网络RNN层,可以捕获数据中的顺序依赖性,该层的输入形状被设置为训练数据的形状,训练数据具有三个纬度,  
样本数量、每个样本中的时间步长数量和每个时间步长中的特征数量
第二层是完全链接(密集)层，神经元的数量等于输出目标的数量,添加该层以基于GRU层的输出进行最终预测
'''
model = models.Sequential()
model.add(layers.GRU(64, input_shape=(train_sample.shape[1], train_sample.shape[2])))
model.add(layers.Dense(train_target.shape[1]))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

# train the model fit方法是keras中用于训练模型的方法，它接受训练数据和训练参数,自动完成模型的训练过程,返回训练过程中的历史信息
history = model.fit(train_sample, train_target, callbacks=callbacks_list, epochs=200, batch_size=5, verbose=1, shuffle=False)
# 可以用来打印出模型的概述信息,包括每个层的输出形状、参数数量等等,这对于了解模型的结构和调试模型非常有帮助
model.summary()
# save方法可以将训练好的模型保存到指定文件中,方便后续加载和使用,保存的文件包括模型的结构,权重
model.save(model_file)
# predict是keras中用于对新样本进行预测的方法,它接受输入数据并返回对应的预测结果，具体来说这个方法会对输入数据predict_sample进行预测,并返回预测结果,这个方法可以用于测试模型的性能,
# 或者在模型训练完成之后对新的数据进行预测
predict_target = model.predict(predict_sample)
print(actual_target)
print(predict_target)

predict_target=predict_target[:,0]

history_dict = history.history
print(history_dict.keys())
# history_dict['loss'][-3:]表示从history_dict字典中获取loss的最后三个值，这行代码的作用是打印出最后三个训练轮次中的损失函数值
print(history_dict['loss'][-3:])

'''
predict_target(预测目标值) actual_target(实际目标值)
np.abs(predict_target-actual_target)计算数组中每个元素的预测值和实际值之间的差值,然后取绝对值
np.mean(np.abs(predict_target-actual_target))计算数组中所有元素的平均值,即平均绝对误差
'''
target_mean = np.mean(np.abs(predict_target-actual_target))

#
# matplotlib.use('TkAgg')
# point = range(1, len(predict_target) + 1)
# plt.figure(figsize=(25,10))
# plt.plot(point, predict_target, 'bo', label='predict_temp')
# plt.plot(point, actual_target, 'b', label='actual_temp')
# plt.title('predict and actual')
# plt.xlabel('point')
# plt.ylabel('temp')
#
# plt.legend()
# plt.show()

if True:
    exit(-1)