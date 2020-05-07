import argparse
from concurrent import futures
import time
import os

from google.protobuf import struct_pb2
import grpc

import bookstore
import wordservice_pb2
import wordservice_pb2_grpc
import status

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class WordServiceServicer(wordservice_pb2_grpc.WordServiceServicer):
    def __init__(self, store):
        self._store = store

    def ListShelves(self, unused_request, context):
        with status.context(context):
            response = wordservice_pb2.ListShelvesResponse()
            response.shelves.extend(self._store.list_shelf())
            return response

    def CreateShelf(self, request, context):
        with status.context(context):
            shelf, _ = self._store.create_shelf(request.shelf)
            return shelf

    def GetShelf(self, request, context):
        with status.context(context):
            return self._store.get_shelf(request.shelf)

    def DeleteShelf(self, request, context):
        with status.context(context):
            self._store.delete_shelf(request.shelf)
            return struct_pb2.Value()

    def ListBooks(self, request, context):
        with status.context(context):
            response = wordservice_pb2.ListBooksResponse()
            response.books.extend(self._store.list_books(request.shelf))
            return response

    def CreateBook(self, request, context):
        with status.context(context):
            return self._store.create_book(request.shelf, request.book)

    def GetBook(self, request, context):
        with status.context(context):
            return self._store.get_book(request.shelf, request.book)

    def DeleteBook(self, request, context):
        with status.context(context):
            self._store.delete_book(request.shelf, request.book)
            return struct_pb2.Value()


def create_sample_bookstore():
    """Creates a Bookstore with some initial sample data."""
    store = bookstore.Bookstore()

    shelf = wordservice_pb2.Shelf()
    shelf.theme = 'Fiction'
    _, fiction = store.create_shelf(shelf)

    book = wordservice_pb2.Book()
    book.title = 'REAMDE'
    book.author = "Neal Stephenson"
    store.create_book(fiction, book)

    shelf = wordservice_pb2.Shelf()
    shelf.theme = 'Fantasy'
    _, fantasy = store.create_shelf(shelf)

    book = wordservice_pb2.Book()
    book.title = 'A Game of Thrones'
    book.author = 'George R.R. Martin'
    store.create_book(fantasy, book)

    return store


def serve(port, shutdown_grace_duration):
    """Configures and runs the bookstore API server."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    store = create_sample_bookstore()
    wordservice_pb2_grpc.add_WordServiceServicer_to_server(
        WordServiceServicer(store), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()

    print('Listening on port {}'.format(port))

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(shutdown_grace_duration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--port', type=int, default=None,
        help='The port to listen on.'
             'If arg is not set, will listen on the $PORT env var.'
             'If env var is empty, defaults to 8000.')
    parser.add_argument(
        '--shutdown_grace_duration', type=int, default=5,
        help='The shutdown grace duration, in seconds')

    args = parser.parse_args()

    port = args.port
    if not port:
        port = os.environ.get('PORT')
    if not port:
        port = 8000

    serve(port, args.shutdown_grace_duration)
