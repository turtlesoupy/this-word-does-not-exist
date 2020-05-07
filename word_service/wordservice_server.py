import argparse
import traceback
from concurrent import futures
import time
import os

from google.protobuf import struct_pb2
import grpc

import wordservice_pb2
import wordservice_pb2_grpc
from contextlib import contextmanager

import grpc


@contextmanager
def context(grpc_context):
    """A context manager that automatically handles KeyError."""
    try:
        yield
    except KeyError as key_error:
        grpc_context.code(grpc.StatusCode.NOT_FOUND)
        grpc_context.details(
            'Unable to find the item keyed by {}'.format(key_error))


class WordServiceServicer(wordservice_pb2_grpc.WordServiceServicer):
    def __init__(self):
        pass

    def DefineWord(self, request, context):
        try:
            response = wordservice_pb2.DefineWordResponse(
                word=wordservice_pb2.WordDefinition(word="abc", definition="bbq",)
            )
            return response
        except Exception:
            traceback.print_exc()
            raise


def serve(port, shutdown_grace_duration):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    wordservice_pb2_grpc.add_WordServiceServicer_to_server(WordServiceServicer(), server)
    server.add_insecure_port("[::]:{}".format(port))
    server.start()

    print("Listening on port {}".format(port))

    try:
        while True:
            time.sleep(3600 * 24)
    except KeyboardInterrupt:
        server.stop(shutdown_grace_duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="The port to listen on."
        "If arg is not set, will listen on the $PORT env var."
        "If env var is empty, defaults to 8000.",
    )
    parser.add_argument(
        "--shutdown_grace_duration", type=int, default=5, help="The shutdown grace duration, in seconds"
    )

    args = parser.parse_args()

    port = args.port
    if not port:
        port = os.environ.get("PORT")
    if not port:
        port = 8000

    serve(port, args.shutdown_grace_duration)
