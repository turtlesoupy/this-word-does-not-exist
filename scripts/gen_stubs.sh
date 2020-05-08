python -m grpc_tools.protoc \
    --include_imports \
    --include_source_info \
    --proto_path=word_service/word_service_proto \
    --python_out=word_service/word_service_proto \
    --grpc_python_out=word_service/word_service_proto \
    --descriptor_set_out=word_service/word_service_proto/api_descriptor.pb \
    wordservice.proto
sed -i -r 's/import (.+_pb2.*)/from . import \1/g' word_service/word_service_proto/*_pb2*.py
