import json
import time
import argparse
import logging
import os
from word_generator import WordGenerator
from google.cloud import storage

logger = logging.getLogger(__name__)


def upload_wotd(blob, word_generator):
    word = word_generator.generate_word()
    if not word:
        raise RuntimeError("Error during generation")

    blob.upload_from_string(json.dumps({
            "word": word.word,
            "part_of_speech": word.pos,
            "definition": word.definition,
            "example_usage": word.example,
            "topic": word.topic,
            "generated_at_ms": int(1000 * time.time()),
        }),
        content_type="application/jon"
    )


def main(args):
    gcloud_credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not gcloud_credentials: 
        raise RuntimeError("Expected to set GOOGLE_APPLICATION_CREDENTIALS env var")
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    lvl = logging.DEBUG if args.verbose else logging.INFO
    if args.log_file:
        print(f"Logging to {args.log_file}")
        logging.basicConfig(
            level=lvl, filename=args.log_file, filemode="a", format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(level=lvl)

    word_generator = WordGenerator(
        device=args.device,
        forward_model_path=args.forward_model_path,
        inverse_model_path=args.inverse_model_path,
        blacklist_path=args.blacklist_path,
        quantize=args.quantize,
    )

    logger.info("Uploading WOTD")
    client = storage.Client(project=args.gcloud_project)
    bucket = client.get_bucket(args.gcloud_bucket)
    blob = bucket.blob(args.gcloud_path)
    upload_wotd(blob, word_generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a wotd bot that uploads a wotd to a specified google bucket")
    parser.add_argument(
        "--device", help="Force a certain device (cuda / cpu)", type=str,
    )
    parser.add_argument("--forward-model-path", help="Model path for (Word -> Definition)", type=str, required=True)
    parser.add_argument("--inverse-model-path", help="Model path for (Definition -> Word)", type=str, required=True)
    parser.add_argument(
        "--blacklist-path", help="Blacklist path for word generation", type=str, required=True,
    )
    parser.add_argument("--quantize", help="Perform quantization for models", action="store_true")
    parser.add_argument("--log-file", type=str, help="Log to this file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--gcloud-project", type=str, required=True)
    parser.add_argument("--gcloud-bucket", type=str, required=True)
    parser.add_argument("--gcloud-path", type=str, default="wotd.json")
    args = parser.parse_args()

    try:
        main(args)
    except Exception:
        logger.exception("Uncaught error")
        raise
