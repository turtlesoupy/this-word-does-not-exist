import os
import json
import jinja2
import aiohttp_jinja2
import aiohttp
import backoff
from aiohttp import web
import argparse
from urllib.parse import quote_plus
import hashlib
import words
import hmac
from word_service.word_service_proto import wordservice_pb2
from word_service.word_service_proto import wordservice_grpc
from grpclib.client import Channel
from grpclib.exceptions import GRPCError
from grpclib.const import Status
import logging
import base64
from async_lru import alru_cache
from title_maker_pro.bad_words import grawlix

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
        permalink_hmac_key=os.environ["PERMALINK_HMAC_KEY"],
        gcloud_api_key=os.environ["GCLOUD_API_KEY"]
    )


def _grpc_nonretriable(e: GRPCError):
    return e.status in (
        Status.INVALID_ARGUMENT,
        Status.NOT_FOUND,
        Status.ALREADY_EXISTS,
        Status.PERMISSION_DENIED,
        Status.FAILED_PRECONDITION,
        Status.UNAUTHENTICATED,
    )



class Handlers:
    def __init__(
        self,
        word_service_address,
        word_service_port,
        word_index,
        recaptcha_server_token,
        permalink_hmac_key,
        gcloud_api_key,
        captcha_timeout=10,
    ):
        self.word_service_channel = Channel(word_service_address, word_service_port)
        self.word_service = wordservice_grpc.WordServiceStub(self.word_service_channel)
        self.word_index = word_index
        self.permalink_hmac_key = permalink_hmac_key.encode("utf-8")
        self.recaptcha_server_token = recaptcha_server_token
        self.captcha_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(captcha_timeout))
        self.gcloud_api_key = gcloud_api_key

    async def on_startup(self, app):
        pass

    async def on_cleanup(self, app):
        await self.captcha_session.close()
        self.word_service_channel.close()

    def _view_word_permalink(self, view_word):
        payload = base64.urlsafe_b64encode(json.dumps(view_word.to_short_dict()).encode("utf-8"))
        signature = base64.urlsafe_b64encode(hmac.new(self.permalink_hmac_key, payload, digestmod=hashlib.sha256).digest())
        permalink = f"{payload.decode('utf-8')}.{signature.decode('utf-8')}"
        return permalink

    def _index_response(self, word, word_in_title=False):
        return {
            "word": word, 
            "word_json": json.dumps(word.to_dict()), 
            "word_exists": bool(word.probably_exists),
            "permalink": self._view_word_permalink(word),
            "word_in_title": bool(word_in_title),
        }

    @aiohttp_jinja2.template("index.jinja2")
    async def index(self, request):
        return self._index_response(self.word_index.random())

    @aiohttp_jinja2.template("index.jinja2")
    async def word(self, request):
        _ = request.match_info["word"]
        encrypted = request.match_info["encrypt"]
        payload, signature = encrypted.split(".")
        h = hmac.new(self.permalink_hmac_key, payload.encode("utf-8"), digestmod=hashlib.sha256)
        if not hmac.compare_digest(h.digest(), base64.urlsafe_b64decode(signature)):
            raise _json_error(web.HTTPBadRequest, "Bad digest")

        word_dict = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
        w = words.Word.from_dict(word_dict)
        return self._index_response(w, word_in_title=True)

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

    @alru_cache(maxsize=65536)
    @backoff.on_exception(backoff.expo, ConnectionRefusedError, max_time=20)
    @backoff.on_exception(backoff.expo, GRPCError, max_time=20, giveup=_grpc_nonretriable)
    async def _cached_fetch_word_definition(self, word):
        return await self.word_service.DefineWord(
            wordservice_pb2.DefineWordRequest(word=word),
            metadata=(('x-api-key', self.gcloud_api_key),)
        )

    async def define_word(self, request):
        try:
            token = request.query["token"]
            word = request.query["word"].strip()
        except KeyError:
            raise _json_error(web.HTTPBadRequest, "Expected token and word args")

        if len(word) > 40:
            return _json_error(web.HTTPBadRequest, "Too long word")
        elif len(word) == 0:
            return _json_error(web.HTTPBadRequest, "Too short word")

        word = grawlix(word)

        if not await self._verify_recaptcha(request.remote, token):
            raise _json_error(web.HTTPBadRequest, "Baddo")

        try:
            response = await self._cached_fetch_word_definition(word)
        except Exception:
            logging.exception("Couldn't fetch word definition")
            raise

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
            web.get("/w/{word}/{encrypt}", handlers.word),
            web.get("/define_word", handlers.define_word),
            web.get("/favicon.ico", handlers.favicon),
            web.static("/static", "./website/static"),
        ]
    )
    aiohttp_jinja2.setup(
        app, loader=jinja2.FileSystemLoader("./website/templates"), filters={
            "quote_plus": quote_plus,
            "remove_period": lambda x: x.rstrip("."),
            "escape_double": lambda x: x.replace('"', r'\"'),
            "strip_quotes": lambda x: x.strip('"'),
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
        permalink_hmac_key=os.environ["PERMALINK_HMAC_KEY"],
        gcloud_api_key=os.environ["GCLOUD_API_KEY"],
    )

    web.run_app(app(handlers), path=args.path)
