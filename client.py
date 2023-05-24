import grpc
import nighttemp_pb2
import  nighttemp_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = nighttemp_pb2_grpc.PythonServiceStub(channel)

request = nighttemp_pb2.Request()
request.message = "World"

response = stub.callPythonMethod(request)

print(response.message)