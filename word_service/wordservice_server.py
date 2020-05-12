import re
import argparse
from concurrent import futures
import time
import os
import grpc
import logging
from word_service_proto import wordservice_pb2
from word_service_proto import wordservice_pb2_grpc
from contextlib import contextmanager

from title_maker_pro.word_generator import WordGenerator
from hyphen import Hyphenator

logger = logging.getLogger(__name__)


@contextmanager
def context(grpc_context):
    try:
        yield
    except KeyError as key_error:
        grpc_context.code(grpc.StatusCode.NOT_FOUND)
        grpc_context.details("Unable to find the item keyed by {}".format(key_error))


class WordServiceServicer(wordservice_pb2_grpc.WordServiceServicer):
    def __init__(self, word_generator, hyphenator):
        self.word_generator = word_generator
        self.hyphenator = hyphenator

    def gen_word_to_word_definition(self, gen_word):
        if gen_word is None:
            return wordservice_pb2.WordDefinition()
        else:
            definition = re.sub(gen_word.word, gen_word.word, gen_word.definition, flags=re.IGNORECASE)
            return wordservice_pb2.WordDefinition(
                word=gen_word.word,
                definition=definition,
                pos=gen_word.pos,
                examples=[gen_word.example],
                syllables=self.hyphenator.syllables(gen_word.word),
            )

    def GenerateWord(self, request, context):
        gen_word = self.word_generator.generate_word()
        return wordservice_pb2.GenerateWordResponse(word=self.gen_word_to_word_definition(gen_word))

    def WordFromDefinition(self, request, context):
        gen_word = self.word_generator.generate_word_from_definition(request.definition)
        return wordservice_pb2.WordFromDefinitionResponse(word=self.gen_word_to_word_definition(gen_word))

    def DefineWord(self, request, context):
        gen_word = self.word_generator.generate_definition(request.word)
        return wordservice_pb2.DefineWordResponse(word=self.gen_word_to_word_definition(gen_word))


def main(args):
    if args.quantize and args.device != "cpu":
        raise RuntimeError("Quantization only available on CPU devices")

    port = args.port or os.environ.get("PORT") or 8000
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=lvl)

    word_generator = WordGenerator(
        device=args.device,
        forward_model_path=args.forward_model_path,
        inverse_model_path=args.inverse_model_path,
        blacklist_path=args.blacklist_path,
        quantize=args.quantize,
    )
    h_en = Hyphenator('en_US')

    logging.info(f"Warming up with word generation")
    gen_word = word_generator.generate_word()
    logging.info(f"Generated {gen_word}")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    wordservice_pb2_grpc.add_WordServiceServicer_to_server(WordServiceServicer(word_generator, h_en), server)
    server.add_insecure_port("[::]:{}".format(port))
    server.start()

    logging.info(f"Listening on port {port}")

    try:
        while True:
            time.sleep(3600 * 24)
    except KeyboardInterrupt:
        server.stop(args.shutdown_grace_duration)


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
        "--shutdown_grace_duration", type=int, default=10, help="The shutdown grace duration, in seconds"
    )
    parser.add_argument(
        "--device", help="Force a certain device (cuda / cpu)", type=str,
    )
    parser.add_argument("--forward-model-path", help="Model path for (Word -> Definition)", type=str, required=True)
    parser.add_argument("--inverse-model-path", help="Model path for (Definition -> Word)", type=str, required=True)
    parser.add_argument(
        "--blacklist-path", help="Blacklist path for word generation", type=str, required=True,
    )
    parser.add_argument("--quantize", help="Perform quantization for models", action="store_true")
    parser.add_argument("--verbose", help="Verbose logging", action="store_true")

    args = parser.parse_args()
    main(args)
