import json
import jinja2
import aiohttp_jinja2
from aiohttp import web
import words
import argparse
from urllib.parse import quote_plus
from word_service.word_service_proto import wordservice_pb2
from word_service.word_service_proto import wordservice_grpc
from grpclib.client import Channel



def _word_to_dict(w: wordservice_pb2.WordDefinition):
    if not w.word:
        return None

    return {
        "word": w.word,
        "definition": w.definition,
        "examples": list(w.examples),
        "pos": w.pos
    }


def _dev_handlers():
    return Handlers(
        word_service_address="localhost",
        word_service_port=8000,
        word_index=words.WordIndex.load("./website/data/words.json"),
    )


class Handlers:
    def __init__(self, word_service_address, word_service_port, word_index):
        channel = Channel(word_service_address, word_service_port)
        self.word_service = wordservice_grpc.WordServiceStub(channel)
        self.word_index = word_index

    @aiohttp_jinja2.template("index.jinja2")
    async def index(self, request):
        w = self.word_index.random()
        return {"word": w}

    async def define_word(self, request):
        word = request.query["word"]
        response = await self.word_service.DefineWord(wordservice_pb2.DefineWordRequest(
            word=word
        ))
        return web.Response(
            text=json.dumps({
                "word": _word_to_dict(response.word),
            }),
            content_type="application/json",
        )

    async def favicon(self, request):
        return web.FileResponse("./website/static/favicon.ico")


def app(handlers=None):
    handlers = handlers or _dev_handlers()
    app = web.Application()
    app.add_routes([
        web.get('/', handlers.index),
        web.get('/define_word', handlers.define_word),
        web.get('/favicon.ico', handlers.favicon),
        web.static("/static", "./website/static"),
    ])
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader("./website/templates"),
        filters={
            'quote_plus': quote_plus
        },
    )
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Unix socket path")
    parser.add_argument("--word-service-address", type=str, help="Word service address", required=True)
    parser.add_argument("--word-service-port", type=int, help="Word service port", required=True)
    parser.add_argument("--word-index-path", type=str, help="Path to word index", required=True)
    args = parser.parse_args()

    wi = words.WordIndex.load(args.word_index_path)
    handlers = Handlers(
        args.word_service_address,
        args.word_service_port,
        word_index=wi,
    )

    web.run_app(app(handlers), path=args.path)
