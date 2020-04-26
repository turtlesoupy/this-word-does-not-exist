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
from transformers import AutoModelWithLMHead, AutoTokenizer
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class WordGenerator:
    def __init__(self):
        parsed_dictionary_path = "data/en_dictionary_parsed_randomized.pickle"
        model_path = "models/en_dictionary_parsed_lr_00001/checkpoint-120000"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device {self.device}")

        logger.info(f"Loading word blacklist from {parsed_dictionary_path}...")
        self.blacklist = set(
            (
                x.lower()
                for x in itertools.chain.from_iterable(
                    [e.word] + e.derivatives
                    for e in pickle.load(open(parsed_dictionary_path, "rb"))
                )
            )
        )
        logger.info(f"Loaded {len(self.blacklist)} words to blacklist")

        logger.info("Loading GPT2 tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens(datasets.SpecialTokens.special_tokens_dict())
        logger.info("Loaded tokenizer")

        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelWithLMHead.from_pretrained(model_path).to(self.device)
        logger.info("Loaded model")

    def generate_definition(self, word, max_length=256):
        prefix = (
            f"{datasets.SpecialTokens.BOS_TOKEN}{word}{datasets.SpecialTokens.POS_SEP}"
        )
        expanded, _ = datasets.ParsedDictionaryDefinitionDataset.generate_words(
            self.tokenizer,
            self.model,
            num=1,
            prefix=prefix,
            max_iterations=1,
            do_example_expansion=True,
            generation_args=dict(
                top_k=300,
                num_return_sequences=3,
                max_length=max_length,
                do_sample=True,
            ),
            expansion_generation_overrides=dict(
                top_k=50, num_return_sequences=10, do_sample=True,
            ),
            num_expansion_candidates=10,
            filter_proper_nouns=True,
        )

        if expanded:
            return expanded[0]
        else:
            return None


def _formulate_reply_text(word_generator, text, author_name, max_defn_length=40):
    removed_mentions = re.sub("(@[\S]*)", "", text).strip()
    remove_word_define = re.sub(
        "^(define |defn )", "", removed_mentions, flags=re.IGNORECASE
    ).strip()

    warning = None
    if len(remove_word_define) > 40:
        splits = remove_word_define.split()
        if len(splits) > 0:
            word = splits[0]
        else:
            word = remove_word_define[:40]
        warning = "Your word was too long! Here is what I could do:"
    elif len(remove_word_define) == 0:
        warning = (
            "I couldn't figure out what you wanted to define, so here is your username:"
        )
        word = author_name
    elif remove_word_define == "me":
        word = author_name
    else:
        word = remove_word_define

    word_with_definition = word_generator.generate_definition(word)

    if not word_with_definition:
        return "Something went wrong on my end, sorry \U0001F61E\U0001F61E\U0001F61E"

    word_view_str = [word_with_definition.word]
    if word_with_definition.pos:
        word_view_str.append(f"/{word_with_definition.pos}/")

    if word_with_definition.topic:
        word_view_str.append(f"[{word_with_definition.topic}]")

    word_view_str.append(f"\n{word_with_definition.definition}")
    word_view_str.append(f'\n"{word_with_definition.example}"')

    if warning:
        reply = f"@{author_name} {warning} {' '.join(word_view_str)}"
    else:
        reply = f"@{author_name} {' '.join(word_view_str)}"

    return reply


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


def bot_loop(bot_state, api, word_generator):
    max_length = 260
    while True:
        cur = tweepy.Cursor(
            api.mentions_timeline, since_id=bot_state.last_processed_id, count=50
        )
        items = list(reversed(list(cur.items())))

        if len(items) > 0:
            logging.info(f"Found {len(items)} items to reply to!")

        for status in items:
            reply = _formulate_reply_text(
                word_generator, status.text, status.author.screen_name
            )
            if len(reply) > max_length:
                logging.warning(
                    f"Reply to {status.id} (@{status.author.screen_name}) too long... truncating: {reply}"
                )
                reply = reply[:max_length]
            api.update_status(reply, in_reply_to_status_id=status.id)
            bot_state.last_processed_id = status.id
            bot_state.write()

        time.sleep(10)


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

    state_exists = os.path.exists(args.state_file)
    if not state_exists and not args.bootstrap:
        raise RuntimeError(
            f"Missing state file at {args.state_file}... did you mean to bootstrap?"
        )
    elif state_exists and args.bootstrap:
        raise RuntimeError(
            f"Bootstrap specified and state file exists at {args.state_file}"
        )
    elif args.bootstrap:
        bot_state = BotState(path=args.state_file)
    else:
        bot_state = BotState.read_from(args.state_file)

    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True,)
    word_generator = WordGenerator()
    bot_loop(bot_state, api, word_generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a twitter bot that replies to tweets with definitions"
    )
    parser.add_argument(
        "--bootstrap",
        help="Whether to create the state file, otherwise it is required",
        action="store_true",
    )
    parser.add_argument(
        "--state-file", type=str, required=True, help="Path to the state file"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args)
