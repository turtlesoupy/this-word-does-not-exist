import re
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
from word_service.word_service_proto.wordservice_pb2 import DatasetType
from word_service.word_service_proto import wordservice_grpc
from grpclib.client import Channel
from grpclib.exceptions import GRPCError
from grpclib.const import Status
import logging
import base64
from async_lru import alru_cache
from title_maker_pro.bad_words import grawlix
from pathlib import Path
from jinja2 import evalcontextfilter
from markupsafe import Markup, escape


logger = logging.getLogger(__name__)


def _load_word_indexes_from_base_dir(base_dir):
    base_dir = Path(base_dir)
    return {
        DatasetType.OED: words.WordIndex.load(base_dir / "words.json.gz"),
        DatasetType.UD_FILTERED: words.WordIndex.load_encrypted(
            base_dir / "words_ud_filtered.enc.gz", fernet_key=os.environ["FERNET_ENCRYPTION_KEY"]
        ),
        DatasetType.UD_UNFILTERED: words.WordIndex.load_encrypted(
            base_dir / "words_ud_unfiltered.enc.gz", fernet_key=os.environ["FERNET_ENCRYPTION_KEY"]
        ),
    }


def _json_error(klass, message):
    return klass(text=json.dumps({"error": message}), content_type="application/json")


def _dev_handlers():
    logging.basicConfig(level=logging.INFO)
    return Handlers(
        word_service_address="localhost",
        word_service_port=8000,
        word_indexes=_load_word_indexes_from_base_dir("./website/data"),
        recaptcha_server_token=os.environ["RECAPTCHA_SERVER_TOKEN"],
        permalink_hmac_key=os.environ["PERMALINK_HMAC_KEY"],
        gcloud_api_key=os.environ["GCLOUD_API_KEY"],
        firebase_api_key=os.environ["FIREBASE_API_KEY"],
    )


def _grpc_nonretriable(e: GRPCError):
    return e.status in (
        Status.INVALID_ARGUMENT,
        Status.UNKNOWN,
        Status.NOT_FOUND,
        Status.ALREADY_EXISTS,
        Status.PERMISSION_DENIED,
        Status.FAILED_PRECONDITION,
        Status.UNAUTHENTICATED,
    )


_paragraph_re = re.compile(r'(?:\r\n|\r|\n){2,}')


@evalcontextfilter
def nl2br(eval_ctx, value):
    result = u'\n\n'.join(p.replace('\n', Markup('<br>\n'))
                          for p in _paragraph_re.split(escape(value)))
    if eval_ctx.autoescape:
        result = Markup(result)
    return result


class Handlers:
    def __init__(
        self,
        word_service_address,
        word_service_port,
        word_indexes,
        recaptcha_server_token,
        permalink_hmac_key,
        gcloud_api_key,
        firebase_api_key,
        captcha_timeout=10,
    ):
        self.word_service_channel = Channel(word_service_address, word_service_port)
        self.word_service = wordservice_grpc.WordServiceStub(self.word_service_channel)
        self.word_indexes = word_indexes
        self.permalink_hmac_key = permalink_hmac_key.encode("utf-8")
        self.recaptcha_server_token = recaptcha_server_token
        self.captcha_timeout = captcha_timeout
        self.gcloud_api_key = gcloud_api_key
        self.firebase_api_key = firebase_api_key

    async def on_startup(self, app):
        self.captcha_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(self.captcha_timeout))
        self.firebase_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(self.captcha_timeout))

    async def on_cleanup(self, app):
        await self.captcha_session.close()
        await self.firebase_session.close()
        self.word_service_channel.close()

    @web.middleware
    async def dataset_middleware(self, request, handler):
        dataset = request.query.get("dataset")
        secret = request.query.get("secret")
        if dataset == "ud_filtered":
            request.dataset = DatasetType.UD_FILTERED
            if secret != os.environ["UD_SECRET"]:
                raise _json_error(web.HTTPBadRequest, "Bad")
            request.dataset_qs = f"dataset={dataset}&secret={os.environ['UD_SECRET']}"
        elif dataset == "ud_unfiltered":
            if secret != os.environ["UD_SECRET"]:
                raise _json_error(web.HTTPBadRequest, "Bad")
            request.dataset = DatasetType.UD_FILTERED
            request.dataset_qs = f"dataset={dataset}&secret={os.environ['UD_SECRET']}"
        else:
            request.dataset = DatasetType.OED
            request.dataset_qs = ""

        resp = await handler(request)
        return resp

    def _view_word_permalink(self, view_word):
        payload = base64.urlsafe_b64encode(json.dumps(view_word.to_short_dict()).encode("utf-8"))
        signature = base64.urlsafe_b64encode(hmac.new(self.permalink_hmac_key, payload, digestmod=hashlib.sha256).digest())
        permalink = f"{payload.decode('utf-8')}.{signature.decode('utf-8')}"
        return permalink

    def _full_permalink_url(self, view_word, permalink):
        return f"https://www.thisworddoesnotexist.com/w/{quote_plus(view_word.word)}/{permalink}"

    def _index_response(self, request, word, word_in_title=False):
        permalink = self._view_word_permalink(word)
        return {
            "word": word, 
            "word_json": json.dumps(word.to_dict()), 
            "word_exists": bool(word.probably_exists),
            "permalink": permalink,
            "full_url": self._full_permalink_url(word, permalink),
            "word_in_title": bool(word_in_title),
            "urban": request.dataset in (DatasetType.UD_FILTERED, DatasetType.UD_UNFILTERED),
            "dataset_qs": request.dataset_qs,
        }

    @aiohttp_jinja2.template("index.jinja2")
    async def index(self, request):
        word_index = self.word_indexes[request.dataset]
        return self._index_response(request, word_index.random())

    def _word_from_url(self, request):
        _ = request.match_info["word"]
        encrypted = request.match_info["encrypt"]
        payload, signature = encrypted.split(".")
        h = hmac.new(self.permalink_hmac_key, payload.encode("utf-8"), digestmod=hashlib.sha256)
        if not hmac.compare_digest(h.digest(), base64.urlsafe_b64decode(signature)):
            raise _json_error(web.HTTPBadRequest, "Bad digest")
        
        word_dict = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
        w = words.Word.from_dict(word_dict)

        if w.dataset_type and w.dataset_type != request.dataset:
            raise _json_error(web.HTTPBadRequest, "Mismatched word dataset")

        return w

    @aiohttp_jinja2.template("index.jinja2")
    async def word(self, request):
        w = self._word_from_url(request)
        return self._index_response(request, w, word_in_title=True)

    async def shorten_word_url(self, request):
        w = self._word_from_url(request)
        permalink = self._view_word_permalink(w)
        full_url = self._full_permalink_url(w, permalink)
        async with self.firebase_session.post(
            url=f"https://firebasedynamiclinks.googleapis.com/v1/shortLinks?key={self.firebase_api_key}",
            json={
                "longDynamicLink": f"https://l.thisworddoesnotexist.com/?link={full_url}",
                "suffix": {
                    "option": "SHORT",
                }
            }
        ) as response:
            d = await response.json()
            if "shortLink" not in d:
                logger.error(f"No shortlink from firebase: {d}")
                raise _json_error(web.HTTPException, "Unexpected response in url shortener")

            return web.Response(
                text=json.dumps({"url": d["shortLink"]}),
                content_type="application/json",
            )

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
    async def _cached_fetch_word_definition(self, word, dataset):
        return await self.word_service.DefineWord(
            wordservice_pb2.DefineWordRequest(
                word=word,
                dataset=dataset,
            ),
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
            response = await self._cached_fetch_word_definition(word, request.dataset)
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
        return web.FileResponse("./website/static/favicons/favicon.ico")


def app(handlers=None):
    handlers = handlers or _dev_handlers()
    app = web.Application(
        middlewares=[handlers.dataset_middleware]
    )
    app.on_startup.append(handlers.on_startup)
    app.on_cleanup.append(handlers.on_cleanup)
    app.add_routes(
        [
            web.get("/", handlers.index),
            web.get("/w/{word}/{encrypt}", handlers.word),
            web.get("/shorten_word_url/{word}/{encrypt}", handlers.shorten_word_url),
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
            "nl2br": nl2br,
        },
    )
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Unix socket path")
    parser.add_argument("--word-service-address", type=str, help="Word service address", required=True)
    parser.add_argument("--word-service-port", type=int, help="Word service port", required=True)
    parser.add_argument("--word-index-base-dir", type=str, help="Path to word index", required=True)
    parser.add_argument("--urban-word-index-path", type=str, help="Path to urban word index", required=True)
    parser.add_argument("--verbose", type=str, help="Verbose logging")
    args = parser.parse_args()

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl)

    word_indexes = _load_word_indexes_from_base_dir(args.word_index_base_dir)
    handlers = Handlers(
        args.word_service_address,
        args.word_service_port,
        word_indexes=word_indexes,
        recaptcha_server_token=os.environ["RECAPTCHA_SERVER_TOKEN"],
        permalink_hmac_key=os.environ["PERMALINK_HMAC_KEY"],
        gcloud_api_key=os.environ["GCLOUD_API_KEY"],
        firebase_api_key=os.environ["FIREBASE_API_KEY"],
    )

    web.run_app(app(handlers), path=args.path)
