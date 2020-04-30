import re
import json
import os
import time
import argparse
import tweepy
import torch
import pickle
import itertools
import datasets
import logging
import stanza
import random
from transformers import AutoModelWithLMHead, AutoTokenizer
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

MAX_TWEET_LENGTH = 250


@dataclass
class BotState:
    path: str
    last_processed_id: Optional[int] = None

    @classmethod
    def read_from(cls, path):
        with open(path) as f:
            j = json.load(f)

        return cls(last_processed_id=j["last_processed_id"], path=path,)

    def write(self):
        with open(self.path, "w") as f:
            json.dump({"last_processed_id": self.last_processed_id}, f)


class WordGenerator:
    def __init__(self, forward_model_path, inverse_model_path, blacklist_path, device=None):
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

        logger.info(f"Loading forward model from {forward_model_path}")
        self.forward_model = AutoModelWithLMHead.from_pretrained(forward_model_path).to(self.device)
        logger.info("Loaded forward model")

        logger.info(f"Loading inverse model from {inverse_model_path}")
        self.inverse_model = AutoModelWithLMHead.from_pretrained(inverse_model_path).to(self.device)
        logger.info("Loaded inverse model")

        self.approx_max_length = 250

    def generate_word(self, user_filter=None):
        expanded, _ = datasets.ParsedDictionaryDefinitionDataset.generate_words(
            self.tokenizer,
            self.forward_model,
            num=1,
            max_iterations=5,
            blacklist=self.blacklist,
            do_example_expansion=True,
            generation_args=dict(
                top_k=300, num_return_sequences=10, max_length=self.approx_max_length, do_sample=True,
            ),
            expansion_generation_overrides=dict(top_k=50, num_return_sequences=20, do_sample=True,),
            num_expansion_candidates=20,
            example_match_pos_pipeline=self.stanza_pos_pipeline,
            user_filter=user_filter,
            dedupe_titles=True,
        )

        return expanded[0] if expanded else None

    def generate_definition(self, word, user_filter=None):
        prefix = f"{datasets.SpecialTokens.BOS_TOKEN}{word}{datasets.SpecialTokens.POS_SEP}"
        expanded, stats = datasets.ParsedDictionaryDefinitionDataset.generate_words(
            self.tokenizer,
            self.forward_model,
            num=1,
            prefix=prefix,
            max_iterations=1,
            do_example_expansion=True,
            generation_args=dict(top_k=75, num_return_sequences=5, max_length=self.approx_max_length, do_sample=True,),
            expansion_generation_overrides=dict(top_k=50, num_return_sequences=20, do_sample=True,),
            num_expansion_candidates=20,
            example_match_pos_pipeline=self.stanza_pos_pipeline,
            dedupe_titles=False,
            user_filter=user_filter,
            hail_mary_example=True,
        )

        logger.debug(stats)

        if expanded:
            return expanded[0]
        else:
            return None

    def generate_word_from_definition(self, definition, user_filter=None):
        prefix = f"{datasets.SpecialTokens.BOS_TOKEN}{definition}{datasets.SpecialTokens.DEFINITION_SEP}"
        expanded, stats = datasets.InverseParsedDictionaryDefinitionDataset.generate_words(
            self.tokenizer,
            self.inverse_model,
            blacklist=self.blacklist,
            num=1,
            prefix=prefix,
            max_iterations=1,
            generation_args=dict(top_k=75, num_return_sequences=20, max_length=self.approx_max_length, do_sample=True,),
            dedupe_titles=True,
            user_filter=user_filter,
        )

        logger.debug(stats)

        if expanded:
            return expanded[0]
        else:
            return None


def _definition_str(word_with_definition):
    word_view_str = [word_with_definition.word]
    if word_with_definition.pos:
        word_view_str.append(f"/{word_with_definition.pos}/")

    if word_with_definition.topic:
        word_view_str.append(f"[{word_with_definition.topic}]")

    word_view_str.append(f"\n{word_with_definition.definition}")
    if word_with_definition.example:
        word_view_str.append(f'\n"{word_with_definition.example}"')
    return " ".join(word_view_str)


def _formulate_reply_text(word_generator, text, author_name, max_len=250):
    removed_mentions = re.sub(r"(@[\S]*)", "", text).strip()
    remove_word_define = re.sub(r"^(define |defn )", "", removed_mentions, flags=re.IGNORECASE).strip()

    warning = None
    inverse_mode = False
    if len(remove_word_define.split()) >= 3:
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
        word_view_str = _definition_str(word_with_definition)
        if warning:
            return f"@{author_name} {warning} {word_view_str}"
        else:
            return f"@{author_name} \U0001F449 {word_view_str}"

    start_time = time.time()
    if inverse_mode:
        word_with_definition = word_generator.generate_word_from_definition(
            remove_word_define, user_filter=lambda w: len(final_text(w)) < 250,
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


def bot_loop(bot_state, me, api, word_generator):
    fetch_count = 200
    while True:
        logger.debug("Querying...")
        items = list(
            reversed(api.mentions_timeline(since_id=bot_state.last_processed_id, count=fetch_count, include_rts=0))
        )

        logger.debug(f"Starting {len(items)} items to reply to!")
        items = [e for e in items if e.in_reply_to_user_id != me.id or e.in_reply_to_status_id is None]
        logger.debug(f"Found {len(items)} items to reply to!")

        if len(items) == fetch_count:
            # If we are at capacity, heuristically dedupe to one per author
            new_items = []
            seen_users = set()
            for item in items:
                if item.author.id in seen_users:
                    continue
                new_items.append(item)
                seen_users.add(item.author_id)
            items = new_items

        for status in items:
            logger.debug(f"STATUS: {status.text}")
            reply = _formulate_reply_text(word_generator, status.text, status.author.screen_name)
            if len(reply) > MAX_TWEET_LENGTH:
                logger.warning(f"Reply to {status.id} (@{status.author.screen_name}) too long... truncating: {reply}")
                reply = reply[:MAX_TWEET_LENGTH]
            api.update_status(reply, in_reply_to_status_id=status.id)
            bot_state.last_processed_id = status.id
            bot_state.write()

        time.sleep(15)


def main(args):
    api_key = os.environ.get("TWITTER_API_KEY")
    api_secret = os.environ.get("TWITTER_API_SECRET")
    access_token = os.environ.get("TWITTER_ACCESS_TOKEN")
    access_secret = os.environ.get("TWITTER_ACCESS_SECRET")

    if not api_key:
        raise RuntimeError("Missing TWITTER_API_KEY environment variable")
    if not api_secret:
        raise RuntimeError("Missing TWITTER_API_SECRET environment variable")
    if not access_token:
        raise RuntimeError("Missing TWITTER_ACCESS_TOKEN environment variable")
    if not access_secret:
        raise RuntimeError("Missing TWITTER_ACCESS_SECRET environment variable")

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
    )

    me = api.me()

    if args.wotd_mode:
        logger.info("Tweeting WOTD")
        tweet_wotd(me, api, word_generator)
    else:
        if not args.state_file:
            raise RuntimeError("State mode must be specified")

        state_exists = os.path.exists(args.state_file)
        if not state_exists and not args.bootstrap:
            raise RuntimeError(f"Missing state file at {args.state_file}... did you mean to bootstrap?")
        elif state_exists and args.bootstrap:
            raise RuntimeError(f"Bootstrap specified and state file exists at {args.state_file}")
        elif args.bootstrap:
            bot_state = BotState(path=args.state_file)
        else:
            bot_state = BotState.read_from(args.state_file)

        logger.info("Entering bot loop")
        bot_loop(bot_state, me, api, word_generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a twitter bot that replies to tweets with definitions")
    parser.add_argument(
        "--bootstrap", help="Whether to create the state file, otherwise it is required", action="store_true",
    )
    parser.add_argument("--state-file", type=str, help="Path to the state file")
    parser.add_argument(
        "--device", help="Force a certain device (cuda / cpu)", type=str,
    )
    parser.add_argument("--forward-model-path", help="Model path for (Word -> Definition)", type=str, required=True)
    parser.add_argument("--inverse-model-path", help="Model path for (Definition -> Word)", type=str, required=True)
    parser.add_argument(
        "--blacklist-path", help="Blacklist path for word generation", type=str, required=True,
    )
    parser.add_argument("--log-file", type=str, help="Log to this file")
    parser.add_argument("--wotd-mode", action="store_true", help="Tweet a word of the day and quit")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    try:
        main(args)
    except Exception:
        logger.exception("Uncaught error")
        raise
