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
import stanza

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
    def __init__(self, model_path, blacklist_path, device=None):
        if not device:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.stanza_pos_pipeline = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')

        logger.info(f"Using device {self.device}")

        logger.info(f"Loading word blacklist from {blacklist_path}...")
        self.blacklist = set(
            (
                x.lower()
                for x in itertools.chain.from_iterable(
                    [e.word] + e.derivatives
                    for e in pickle.load(open(blacklist_path, "rb"))
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

    def generate_word(self, max_length=256):
        expanded, _ = datasets.ParsedDictionaryDefinitionDataset.generate_words(
            self.tokenizer,
            self.model,
            num=1,
            max_iterations=5,
            blacklist=self.blacklist,
            do_example_expansion=True,
            generation_args=dict(
                top_k=300,
                num_return_sequences=10,
                max_length=max_length,
                do_sample=True,
            ),
            expansion_generation_overrides=dict(
                top_k=50, num_return_sequences=20, do_sample=True,
            ),
            num_expansion_candidates=20,
            device=self.device,
            example_match_pos_pipeline=self.stanza_pos_pipeline,
        )

        return expanded[0] if expanded else None

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
                top_k=75, num_return_sequences=5, max_length=max_length, do_sample=True,
            ),
            expansion_generation_overrides=dict(
                top_k=50, num_return_sequences=10, do_sample=True,
            ),
            num_expansion_candidates=10,
            device=self.device,
            example_match_pos_pipeline=self.stanza_pos_pipeline,
        )

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
    word_view_str.append(f'\n"{word_with_definition.example}"')
    return " ".join(word_view_str)


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

    start_time = time.time()
    word_with_definition = word_generator.generate_definition(word)
    logging.info(f"Word generation took {time.time() - start_time:.2f}s")

    if not word_with_definition:
        return "Something went wrong on my end, sorry \U0001F61E\U0001F61E\U0001F61E"

    word_view_str = _definition_str(word_with_definition)
    if warning:
        reply = f"@{author_name} {warning} {word_view_str}"
    else:
        reply = f"@{author_name} {word_view_str}"

    return reply


def _formulate_wotd_text(word_with_definition):
    prefix = "Fake word of the day:"
    word_view_str = _definition_str(word_with_definition)
    return f"{prefix} {word_view_str}"


def tweet_wotd(api, word_generator):
    word = word_generator.generate_word()
    if not word:
        raise RuntimeError("Error during generation")

    wotd_text = _formulate_wotd_text(word)
    api.update_status(wotd_text)


def bot_loop(bot_state, api, word_generator):
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
            if len(reply) > MAX_TWEET_LENGTH:
                logging.warning(
                    f"Reply to {status.id} (@{status.author.screen_name}) too long... truncating: {reply}"
                )
                reply = reply[:MAX_TWEET_LENGTH]
            api.update_status(reply, in_reply_to_status_id=status.id)
            bot_state.last_processed_id = status.id
            bot_state.write()

        time.sleep(10)


def main(args):
    stanza.download('en')

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

    if args.log_file:
        print(f"Logging to {args.log_file}")
        logging.basicConfig(
            level=logging.INFO,
            filename=args.log_file,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(level=logging.INFO)

    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True,)
    word_generator = WordGenerator(
        device=args.device,
        model_path=args.model_path,
        blacklist_path=args.blacklist_path,
    )

    if args.wotd_mode:
        tweet_wotd(api, word_generator)
    else:
        if not args.state_file:
            raise RuntimeError("State mode must be specified")

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
    parser.add_argument("--state-file", type=str, help="Path to the state file")
    parser.add_argument(
        "--device", help="Force a certain device (cuda / cpu)", type=str,
    )
    parser.add_argument(
        "--model-path", help="Model path for word generation", type=str, required=True
    )
    parser.add_argument(
        "--blacklist-path",
        help="Blacklist path for word generation",
        type=str,
        required=True,
    )
    parser.add_argument("--log-file", type=str, help="Log to this file")
    parser.add_argument(
        "--wotd-mode", action="store_true", help="Tweet a word of the day and quit"
    )
    args = parser.parse_args()

    try:
        main(args)
    except Exception:
        logger.exception("Uncaught error")
        raise
