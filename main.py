# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from concurrent import futures

import grpc
import nighttemp_pb2
import  nighttemp_pb2_grpc

class PythonServiceServicer(nighttemp_pb2_grpc.TempServiceServicer):
    def Mymethod(self, request, context):
        print('11111')
        temp_predictions = []
        for temp_prediction in request.tempPrediction:
            # Do something with each temp_prediction
            temp_predictions.append(temp_prediction)
        return nighttemp_pb2.TempServiceServicer(tempPrediction=temp_predictions)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nighttemp_pb2_grpc.add_TempServiceServicer_to_server(PythonServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    serve()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# 生成一个夜晚温度预测的模型 用于预测未来的温度
# 1.加载数据集
# 2.数据预处理
# 3.构建模型
# 4.训练模型
# 5.评估模型
# 6.预测模型
# 7.保存模型
# 8.模型部署



