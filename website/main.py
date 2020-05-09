import os
import json
import jinja2
import aiohttp_jinja2
import aiohttp
from aiohttp import web
import words
import argparse
from urllib.parse import quote_plus
from word_service.word_service_proto import wordservice_pb2
from word_service.word_service_proto import wordservice_grpc
from grpclib.client import Channel
import logging

logger = logging.getLogger(__name__)


def _word_to_dict(w: wordservice_pb2.WordDefinition):
    if not w.word:
        return None

    return {
        "word": w.word,
        "definition": w.definition,
        "examples": list(w.examples),
        "pos": w.pos,
    }


def _json_error(klass, message):
    return klass(text=json.dumps({"error": message}))


def _dev_handlers():
    logging.basicConfig(level=logging.INFO)
    return Handlers(
        word_service_address="localhost",
        word_service_port=8000,
        word_index=words.WordIndex.load("./website/data/words.json"),
        recaptcha_server_token=os.environ["RECAPTCHA_SERVER_TOKEN"],
    )


class Handlers:
    def __init__(
        self,
        word_service_address,
        word_service_port,
        word_index,
        recaptcha_server_token,
        captcha_timeout=10,
    ):
        self.captcha_timeout = captcha_timeout

        self.word_service_channel = Channel(word_service_address, word_service_port)
        self.word_service = wordservice_grpc.WordServiceStub(self.word_service_channel)
        self.word_index = word_index
        self.recaptcha_server_token = recaptcha_server_token

    async def on_startup(self, app):
        self.captcha_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(self.captcha_timeout)
        )

    async def on_cleanup(self, app):
        await self.captcha_session.close()
        self.word_service_channel.close()

    @aiohttp_jinja2.template("index.jinja2")
    async def index(self, request):
        w = self.word_index.random()
        return {"word": w}

    async def _verify_recaptcha(self, ip, token):
        async with self.captcha_session.post(
            "https://www.google.com/recaptcha/api/siteverify",
            params={
                "secret": self.recaptcha_server_token,
                "response": token,
                "remoteip": ip,
            },
        ) as response:
            d = await response.json()

            if d.get("error-codes"):
                logger.warning(f"Captcha server error {d['error-codes']}")
                return False

            return bool(d.get("success"))

    async def define_word(self, request):
        try:
            token = request.query["token"]
            word = request.query["word"]
        except KeyError:
            raise _json_error(web.HTTPBadRequest, "Expected token and word args")

        if not await self._verify_recaptcha(request.remote, token):
            raise _json_error(web.HTTPBadRequest, "Baddo")

        response = await self.word_service.DefineWord(
            wordservice_pb2.DefineWordRequest(word=word)
        )
        return web.Response(
            text=json.dumps({"word": _word_to_dict(response.word),}),
            content_type="application/json",
        )

    async def favicon(self, request):
        return web.FileResponse("./website/static/favicon.ico")


def app(handlers=None):
    handlers = handlers or _dev_handlers()
    app = web.Application()
    app.on_startup.append(handlers.on_startup)
    app.on_cleanup.append(handlers.on_cleanup)
    app.add_routes(
        [
            web.get("/", handlers.index),
            web.get("/define_word", handlers.define_word),
            web.get("/favicon.ico", handlers.favicon),
            web.static("/static", "./website/static"),
        ]
    )
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader("./website/templates"),
        filters={"quote_plus": quote_plus},
    )
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Unix socket path")
    parser.add_argument(
        "--word-service-address", type=str, help="Word service address", required=True
    )
    parser.add_argument(
        "--word-service-port", type=int, help="Word service port", required=True
    )
    parser.add_argument(
        "--word-index-path", type=str, help="Path to word index", required=True
    )
    parser.add_argument("--verbose", type=str, help="Verbose logging")
    args = parser.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl)

    wi = words.WordIndex.load(args.word_index_path)
    handlers = Handlers(
        args.word_service_address,
        args.word_service_port,
        word_index=wi,
        recaptcha_server_token=os.environ["RECAPTCHA_SERVER_TOKEN"],
    )

    web.run_app(app(handlers), path=args.path)
