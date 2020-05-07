python -m grpc_tools.protoc \
    --include_imports \
    --include_source_info \
    --proto_path=. \
    --python_out=. \
    --grpc_python_out=. \
    --descriptor_set_out=api_descriptor.pb \
    wordservice.proto
