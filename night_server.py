from concurrent import futures
import time

import grpc

# from nightTemp import nighttemp_pb2, nighttemp_pb2_grpc
import nighttemp_pb2
import  nighttemp_pb2_grpc

class TempServiceServicer(nighttemp_pb2_grpc.TempServiceServicer):
    def GetTemp(self, request, context):

        print(f"Received request: {request}")
        return nighttemp_pb2.TempResponse(tempPrediction=request.tempPrediction)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nighttemp_pb2_grpc.add_TempServiceServicer_to_server(TempServiceServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()