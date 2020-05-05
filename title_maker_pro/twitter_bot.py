import re
import os
import time
import argparse
import tweepy
import torch
import datasets
import logging
import stanza
import random
from transformers import AutoModelWithLMHead, AutoTokenizer
import modeling

logger = logging.getLogger(__name__)

MAX_TWEET_LENGTH = 250


class WordGenerator:
    def __init__(self, forward_model_path, inverse_model_path, blacklist_path, quantize=False, device=None):
        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.stanza_pos_pipeline = stanza.Pipeline(
            lang="en", processors="tokenize,mwt,pos", use_gpu=("cpu" not in self.device.type)
        )

        logger.info(f"Using device {self.device}")

        logger.info(f"Loading word blacklist from {blacklist_path}...")
        self.blacklist = datasets.Blacklist.load(blacklist_path)
        logger.info(f"Loaded {len(self.blacklist)} words to blacklist")

        logger.info("Loading GPT2 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(datasets.SpecialTokens.special_tokens_dict())
        logger.info("Loaded tokenizer")

        ml = modeling.load_model
        if quantize:
            ml = modeling.load_quantized_model
            logger.info(f"Peforming quantization on models")

        logger.info(f"Loading forward model from {forward_model_path}")
        self.forward_model = ml(AutoModelWithLMHead, forward_model_path).to(self.device)
        logger.info("Loaded forward model")

        logger.info(f"Loading inverse model from {inverse_model_path}")
        self.inverse_model = ml(AutoModelWithLMHead, inverse_model_path).to(self.device)
        logger.info("Loaded inverse model")

        self.approx_max_length = 250

    def generate_word(self, user_filter=None):
        expanded, _ = datasets.ParsedDictionaryDefinitionDataset.generate_words(
            self.tokenizer,
            self.forward_model,
            num=1,
            max_iterations=5,
            blacklist=self.blacklist,
            generation_args=dict(
                top_k=300, num_return_sequences=10, max_length=self.approx_max_length, do_sample=True,
            ),
            num_expansion_candidates=20,
            example_match_pos_pipeline=self.stanza_pos_pipeline,
            user_filter=user_filter,
            dedupe_titles=True,
            filter_proper_nouns=True,
            use_custom_generate=True,
        )

        return expanded[0] if expanded else None

    def generate_definition(self, word, user_filter=None):
        prefix = f"{datasets.SpecialTokens.BOS_TOKEN}{word}{datasets.SpecialTokens.POS_SEP}"
        expanded, stats = datasets.ParsedDictionaryDefinitionDataset.generate_words(
            self.tokenizer,
            self.forward_model,
            num=1,
            prefix=prefix,
            max_iterations=3,
            generation_args=dict(top_k=75, num_return_sequences=6, max_length=self.approx_max_length, do_sample=True,),
            example_match_pos_pipeline=self.stanza_pos_pipeline,
            dedupe_titles=False,
            user_filter=user_filter,
            filter_proper_nouns=False,
            use_custom_generate=True,
        )

        logger.info(f"Generation stats: {stats}")

        if expanded:
            return expanded[0]
        else:
            return None

    def generate_word_from_definition(self, definition, user_filter=None):
        # Data peculiarity: definitions ending in a period are out of domain
        prefix = f"{datasets.SpecialTokens.BOS_TOKEN}{definition.rstrip('. ')}{datasets.SpecialTokens.DEFINITION_SEP}"
        expanded, stats = datasets.InverseParsedDictionaryDefinitionDataset.generate_words(
            self.tokenizer,
            self.inverse_model,
            blacklist=self.blacklist,
            num=1,
            prefix=prefix,
            max_iterations=5,
            generation_args=dict(top_k=200, num_return_sequences=5, max_length=self.approx_max_length, do_sample=True,),
            dedupe_titles=True,
            user_filter=user_filter,
        )

        logger.debug(stats)

        if expanded:
            return expanded[0]
        else:
            return None


def _inverse_definition_str(word_with_definition):
    return f"{word_with_definition.word}: {word_with_definition.definition}"


def _definition_str(word_with_definition):
    word_view_str = [word_with_definition.word]
    if word_with_definition.pos:
        word_view_str.append(f"/{word_with_definition.pos}/")

    if word_with_definition.topic:
        word_view_str.append(f"[{word_with_definition.topic}]")

    word_view_str.append(f"\n{word_with_definition.definition}")
    if word_with_definition.example:
        example_string = re.sub(
            word_with_definition.word, word_with_definition.word, word_with_definition.example, flags=re.IGNORECASE
        )  # Minor cleanup to correct the case of a word in the example text
        word_view_str.append(f'\n"{example_string}"')
    return " ".join(word_view_str)


def _formulate_reply_text(word_generator, text, author_name, max_len=250):
    inverse_mode_threshold = 3
    removed_mentions = re.sub(r"(@[\S]*)", "", text).strip()
    remove_word_define = re.sub(r"^(define |defn )", "", removed_mentions, flags=re.IGNORECASE).strip()

    warning = None
    inverse_mode = False
    if len(remove_word_define.split()) >= inverse_mode_threshold:
        inverse_mode = True
    elif len(remove_word_define) > 40:
        splits = remove_word_define.split()
        if len(splits) > 0:
            word = splits[0]
        else:
            word = remove_word_define[:40]
        warning = "Word too long! Here's:"
    elif len(remove_word_define) == 0:
        word = author_name
    elif remove_word_define == "me":
        word = author_name
    else:
        word = remove_word_define

    def final_text(word_with_definition):
        if inverse_mode:
            word_view_str = _inverse_definition_str(word_with_definition)
            emoji = "\U0001F643"
        else:
            word_view_str = _definition_str(word_with_definition)
            emoji = "\U0001F449"

        if warning:
            return f"@{author_name} {warning} {word_view_str}"
        else:
            return f"@{author_name} {emoji} {word_view_str}"

    start_time = time.time()
    if inverse_mode:
        word_with_definition = word_generator.generate_word_from_definition(
            remove_word_define,
            user_filter=(
                lambda w: (
                    len(final_text(w)) < 250
                    and len(w.word.split()) < inverse_mode_threshold
                    and datasets.SpecialTokens.DEFINITION_SEP not in w.word
                )
            ),
        )
    else:
        word_with_definition = word_generator.generate_definition(word, user_filter=lambda w: len(final_text(w)) < 250,)
    logger.info(f"Word generation ({'inverse' if inverse_mode else 'forward'}) took {time.time() - start_time:.2f}s")

    if not word_with_definition:
        return f"@{author_name} something went wrong on my end, sorry \U0001F61E\U0001F61E\U0001F61E"

    return final_text(word_with_definition)


def _formulate_wotd_text(word_with_definition, emoji):
    prefix = f"{emoji} Fake word of the day:"
    word_view_str = _definition_str(word_with_definition)
    return f"{prefix} {word_view_str}"


def tweet_wotd(me, api, word_generator):
    emoji = random.choice(
        (
            "\U0001F970",
            "\U0001F609",
            "\U0001F600",
            "\U0001F92A",
            "\U0001F917",
            "\U0001F92D",
            "\U0001F92B",
            "\U0001F914",
            "\U0001F636",
            "\U0001F60C",
            "\U0001F635",
            "\U0001F974",
            "\U0001F920",
            "\U0001F613",
            "\U0001F64A",
            "\U00002764",
            "\U0001F91E",
            "\U0001F44C",
            "\U0001F450",
            "\U0001F646",
            "\U0001F575",
            "\U0001F486",
            "\U0001F425",
            "\U0001F339",
            "\U0001F31E",
        )
    )
    word = word_generator.generate_word(user_filter=lambda w: len(_formulate_wotd_text(w, emoji)) < 250)
    if not word:
        raise RuntimeError("Error during generation")

    wotd_text = _formulate_wotd_text(word, emoji)
    api.update_status(wotd_text)
    logger.info(f"Tweeted {wotd_text}")


def bot_loop(me, api, word_generator, last_processed_id):
    fetch_count = 200
    while True:
        logger.debug("Querying...")
        items = list(reversed(api.mentions_timeline(since_id=last_processed_id, count=fetch_count, include_rts=0)))

        logger.debug(f"Starting {len(items)} items to reply to!")

        if len(items) == fetch_count:
            # If we are at capacity, heuristically dedupe to one per author
            logger.info("At capacity, deduping seen authors")
            new_items = []
            seen_users = set()
            for item in items:
                if item.author.id in seen_users:
                    continue
                new_items.append(item)
                seen_users.add(item.author_id)
            items = new_items

        items = [
            e
            for e in items
            if (e.in_reply_to_user_id != me.id or e.in_reply_to_status_id is None)
            and e.author.id != me.id
            and f"@{me.screen_name}" in e.text
        ]
        logger.debug(f"Found {len(items)} items to reply to!")

        for status in items:
            logger.debug(f"STATUS: {status.text}")
            reply = _formulate_reply_text(word_generator, status.text, status.author.screen_name)
            if len(reply) > MAX_TWEET_LENGTH:
                logger.warning(f"Reply to {status.id} (@{status.author.screen_name}) too long... truncating: {reply}")
                reply = reply[:MAX_TWEET_LENGTH]
            api.update_status(reply, in_reply_to_status_id=status.id)
            last_processed_id = status.id

        time.sleep(15)


def _fetch_last_processed_id(api, app_name):
    items = api.user_timeline(count=100)
    return next(e.in_reply_to_status_id for e in items if e.source == app_name and e.in_reply_to_status_id is not None)


def main(args):
    api_key = os.environ.get("TWITTER_API_KEY")
    api_secret = os.environ.get("TWITTER_API_SECRET")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
    access_secret = os.environ.get("TWITTER_ACCESS_SECRET")
    app_name = os.environ.get("TWITTER_APP_NAME")

    if not api_key:
        raise RuntimeError("Missing TWITTER_API_KEY environment variable")
    if not api_secret:
        raise RuntimeError("Missing TWITTER_API_SECRET environment variable")
    if not access_token:
        raise RuntimeError("Missing TWITTER_ACCESS_TOKEN environment variable")
    if not access_secret:
        raise RuntimeError("Missing TWITTER_ACCESS_SECRET environment variable")
    if not app_name:
        raise RuntimeError("Missing TWITTER_APP_NAME environment variable")

    stanza.download("en")

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

    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(
        auth,
        wait_on_rate_limit=True,
        wait_on_rate_limit_notify=True,
        retry_count=5,
        retry_delay=30,
        retry_errors=set([500, 503]),
    )
    word_generator = WordGenerator(
        device=args.device,
        forward_model_path=args.forward_model_path,
        inverse_model_path=args.inverse_model_path,
        blacklist_path=args.blacklist_path,
        quantize=args.quantize,
    )

    me = api.me()

    if args.wotd_mode:
        logger.info("Tweeting WOTD")
        tweet_wotd(me, api, word_generator)
    else:
        try:
            last_processed_id = _fetch_last_processed_id(api, app_name)
        except StopIteration:
            if args.bootstrap:
                logger.warning("Bootstrapping bot with no last replied to id")
                last_processed_id = None
            else:
                raise RuntimeError("Unable to determine last replied to id")

        logger.info(f"Entering bot loop - starting from {last_processed_id}")
        bot_loop(me, api, word_generator, last_processed_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a twitter bot that replies to tweets with definitions")
    parser.add_argument(
        "--bootstrap", help="Whether this is the first time we are starting the bot (no replies)", action="store_true",
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
    parser.add_argument("--log-file", type=str, help="Log to this file")
    parser.add_argument("--wotd-mode", action="store_true", help="Tweet a word of the day and quit")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    try:
        main(args)
    except Exception:
        logger.exception("Uncaught error")
        raise
