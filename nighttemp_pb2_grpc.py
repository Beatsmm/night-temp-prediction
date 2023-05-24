# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import nighttemp_pb2 as nighttemp__pb2


class TempServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetTemp = channel.unary_unary(
                '/nightTemp.TempService/GetTemp',
                request_serializer=nighttemp__pb2.TempRequest.SerializeToString,
                response_deserializer=nighttemp__pb2.TempResponse.FromString,
                )


class TempServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetTemp(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_TempServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetTemp': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTemp,
                    request_deserializer=nighttemp__pb2.TempRequest.FromString,
                    response_serializer=nighttemp__pb2.TempResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'nightTemp.TempService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class TempService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetTemp(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/nightTemp.TempService/GetTemp',
            nighttemp__pb2.TempRequest.SerializeToString,
            nighttemp__pb2.TempResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
