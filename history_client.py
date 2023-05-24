import grpc
import historydata_pb2
import historydata_pb2_grpc

channel = grpc.insecure_channel('localhost:20882')
stub = historydata_pb2_grpc.HistoryDataServiceStub(channel)

request = historydata_pb2.HistoryDataRequest(
    deviceNum = '0719411600000015',
    cityCode = '北京市',
    dateBegin = 1675785600,
    dateEnd = 1683475200
)

response = stub.getHistoryData(request)

for data in response.historyData:
    print(data)