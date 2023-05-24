# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: historydata.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='historydata.proto',
  package='nightTempHistory',
  syntax='proto3',
  serialized_options=b'\n\033com.rain.dm.dw.service.grpcB\023HistoryTrainServiceP\001',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x11historydata.proto\x12\x10nightTempHistory\"]\n\x12HistoryDataRequest\x12\x11\n\tdeviceNum\x18\x01 \x01(\t\x12\x11\n\tdateBegin\x18\x02 \x01(\x03\x12\x0f\n\x07\x64\x61teEnd\x18\x03 \x01(\x03\x12\x10\n\x08\x63ityCode\x18\x04 \x01(\t\"*\n\x13HistoryDataResponse\x12\x13\n\x0bhistoryData\x18\x01 \x03(\t2u\n\x12HistoryDataService\x12_\n\x0egetHistoryData\x12$.nightTempHistory.HistoryDataRequest\x1a%.nightTempHistory.HistoryDataResponse\"\x00\x42\x34\n\x1b\x63om.rain.dm.dw.service.grpcB\x13HistoryTrainServiceP\x01\x62\x06proto3'
)




_HISTORYDATAREQUEST = _descriptor.Descriptor(
  name='HistoryDataRequest',
  full_name='nightTempHistory.HistoryDataRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='deviceNum', full_name='nightTempHistory.HistoryDataRequest.deviceNum', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dateBegin', full_name='nightTempHistory.HistoryDataRequest.dateBegin', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dateEnd', full_name='nightTempHistory.HistoryDataRequest.dateEnd', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='cityCode', full_name='nightTempHistory.HistoryDataRequest.cityCode', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=39,
  serialized_end=132,
)


_HISTORYDATARESPONSE = _descriptor.Descriptor(
  name='HistoryDataResponse',
  full_name='nightTempHistory.HistoryDataResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='historyData', full_name='nightTempHistory.HistoryDataResponse.historyData', index=0,
      number=1, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=134,
  serialized_end=176,
)

DESCRIPTOR.message_types_by_name['HistoryDataRequest'] = _HISTORYDATAREQUEST
DESCRIPTOR.message_types_by_name['HistoryDataResponse'] = _HISTORYDATARESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

HistoryDataRequest = _reflection.GeneratedProtocolMessageType('HistoryDataRequest', (_message.Message,), {
  'DESCRIPTOR' : _HISTORYDATAREQUEST,
  '__module__' : 'historydata_pb2'
  # @@protoc_insertion_point(class_scope:nightTempHistory.HistoryDataRequest)
  })
_sym_db.RegisterMessage(HistoryDataRequest)

HistoryDataResponse = _reflection.GeneratedProtocolMessageType('HistoryDataResponse', (_message.Message,), {
  'DESCRIPTOR' : _HISTORYDATARESPONSE,
  '__module__' : 'historydata_pb2'
  # @@protoc_insertion_point(class_scope:nightTempHistory.HistoryDataResponse)
  })
_sym_db.RegisterMessage(HistoryDataResponse)


DESCRIPTOR._options = None

_HISTORYDATASERVICE = _descriptor.ServiceDescriptor(
  name='HistoryDataService',
  full_name='nightTempHistory.HistoryDataService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_start=178,
  serialized_end=295,
  methods=[
  _descriptor.MethodDescriptor(
    name='getHistoryData',
    full_name='nightTempHistory.HistoryDataService.getHistoryData',
    index=0,
    containing_service=None,
    input_type=_HISTORYDATAREQUEST,
    output_type=_HISTORYDATARESPONSE,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
  ),
])
_sym_db.RegisterServiceDescriptor(_HISTORYDATASERVICE)

DESCRIPTOR.services_by_name['HistoryDataService'] = _HISTORYDATASERVICE

# @@protoc_insertion_point(module_scope)