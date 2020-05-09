import os
import json
import jinja2
import aiohttp_jinja2
import aiohttp
from aiohttp import web
import words
import argparse
from urllib.parse import quote_plus
from cryptography.fernet import Fernet
from word_service.word_service_proto import wordservice_pb2
from word_service.word_service_proto import wordservice_grpc
from grpclib.client import Channel
import logging

logger = logging.getLogger(__name__)


def _json_error(klass, message):
    return klass(text=json.dumps({"error": message}), content_type="application/json")


def _dev_handlers():
    logging.basicConfig(level=logging.INFO)
    return Handlers(
        word_service_address="localhost",
        word_service_port=8000,
        word_index=words.WordIndex.load("./website/data/words.json"),
        recaptcha_server_token=os.environ["RECAPTCHA_SERVER_TOKEN"],
        fernet_crypto_key=os.environ["FERNET_CRYPTO_KEY"],
    )


class Handlers:
    def __init__(
        self,
        word_service_address,
        word_service_port,
        word_index,
        recaptcha_server_token,
        fernet_crypto_key,
        captcha_timeout=10,
    ):
        self.fernet = Fernet(fernet_crypto_key)
        self.word_service_channel = Channel(word_service_address, word_service_port)
        self.word_service = wordservice_grpc.WordServiceStub(self.word_service_channel)
        self.word_index = word_index
        self.fernet_crypto_key = fernet_crypto_key
        self.recaptcha_server_token = recaptcha_server_token
        self.captcha_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(captcha_timeout))

    async def on_startup(self, app):
        pass

    async def on_cleanup(self, app):
        await self.captcha_session.close()
        self.word_service_channel.close()

    def _word_permalink(self, view_word):
        return self.fernet.encrypt(json.dumps(view_word.to_short_dict()).encode("utf-8")).decode("utf-8")

    @aiohttp_jinja2.template("index.jinja2")
    async def index(self, request):
        if "p" in request.query:
            w = words.Word.from_dict(
                json.loads(self.fernet.decrypt(request.query["p"].encode("utf-8")).decode("utf-8"))
            )
        else:
            w = self.word_index.random()
        return {
            "word": w, 
            "word_json": json.dumps(w.to_dict()), 
            "permalink": self._word_permalink(w)
        }

    async def _verify_recaptcha(self, ip, token):
        async with self.captcha_session.post(
            "https://www.google.com/recaptcha/api/siteverify",
            params={"secret": self.recaptcha_server_token, "response": token, "remoteip": ip,},
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

        response = await self.word_service.DefineWord(wordservice_pb2.DefineWordRequest(word=word))

        if not response.word or not response.word.word or not response.word.definition:
            raise _json_error(web.HTTPServerError, "Couldn't define")

        view_word = words.Word.from_protobuf(response.word)
        return web.Response(
            text=json.dumps({"word": view_word.to_dict(), "permalink": self._view_word_permalink(view_word)}), content_type="application/json",
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
        app, loader=jinja2.FileSystemLoader("./website/templates"), filters={
            "quote_plus": quote_plus,
            "escape_double": lambda x: x.replace('"', r'\"'),
        },
    )
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Unix socket path")
    parser.add_argument("--word-service-address", type=str, help="Word service address", required=True)
    parser.add_argument("--word-service-port", type=int, help="Word service port", required=True)
    parser.add_argument("--word-index-path", type=str, help="Path to word index", required=True)
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
        fernet_crypto_key=os.environ["FERNET_CRYPTO_KEY"],
    )

    web.run_app(app(handlers), path=args.path)
