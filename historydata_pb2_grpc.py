# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import historydata_pb2 as historydata__pb2


class HistoryDataServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.getHistoryData = channel.unary_unary(
                '/nightTempHistory.HistoryDataService/getHistoryData',
                request_serializer=historydata__pb2.HistoryDataRequest.SerializeToString,
                response_deserializer=historydata__pb2.HistoryDataResponse.FromString,
                )


class HistoryDataServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def getHistoryData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_HistoryDataServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'getHistoryData': grpc.unary_unary_rpc_method_handler(
                    servicer.getHistoryData,
                    request_deserializer=historydata__pb2.HistoryDataRequest.FromString,
                    response_serializer=historydata__pb2.HistoryDataResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'nightTempHistory.HistoryDataService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class HistoryDataService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def getHistoryData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/nightTempHistory.HistoryDataService/getHistoryData',
            historydata__pb2.HistoryDataRequest.SerializeToString,
            historydata__pb2.HistoryDataResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)