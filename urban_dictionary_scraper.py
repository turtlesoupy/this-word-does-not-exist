import os
import re
import torch
import logging
import datetime
import random
import time
import pickle
import hashlib
import requests
import requests_cache
import string
import urllib
import itertools
import asyncio
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from urllib3.exceptions import ProtocolError
from dataclasses import dataclass
from bs4 import BeautifulSoup
from collections import OrderedDict
from typing import List, Optional
from retry import retry

logger = logging.getLogger(__name__)
UD_ROOT = "https://www.urbandictionary.com"


@dataclass
class UrbanDictionaryWordURL:
    title: str
    url: str


@dataclass
class UrbanDictionaryIndexPage:
    url: str
    definition_urls: List[UrbanDictionaryWordURL]
    num_pages: Optional[int]


@dataclass
class UrbanDictionaryDefinition:
    word: str
    url: str

    meaning: str
    author: str
    examples: List[str]
    tags: List[str]

    outbound_links: List[UrbanDictionaryWordURL]

    creation_epoch: float
    upvotes: int
    downvotes: int


@dataclass
class UrbanDictionaryWord:
    url: str
    title: str
    definitions: List[UrbanDictionaryDefinition]


class StatusError(RuntimeError):
    def __init__(self, code, string):
        super().__init__(string)
        self.code = code


def make_throttle_hook(rand_timeout=1.0):
    """
    Returns a response hook function which sleeps for `timeout` seconds if
    response is not cached
    """

    def hook(response, *args, **kwargs):
        if not getattr(response, "from_cache", False):
            time.sleep(random.random() * rand_timeout)
        return response

    return hook


def get_session(throttle=0.1, expiry=24 * 3600):
    session = requests_cache.CachedSession("requests_cache", expire_after=expiry)
    session.hooks = {"response": make_throttle_hook(rand_timeout=throttle)}

    return session


@retry(
    exceptions=(OSError, ConnectionError, ProtocolError, requests.exceptions.Timeout),
    tries=20,
    delay=10,
    backoff=1.5,
    logger=logger,
)
def get_with_retries(session, url, timeout=20.0):
    ret = session.get(url, timeout=timeout)

    if ret.status_code != 200:
        raise StatusError(ret.status_code, f"Unexpected status code in {url}: {ret.status_code}")

    return ret


def fetch_letter_page(session, letter, page=1):
    url = f"{UD_ROOT}/browse.php?character={urllib.parse.quote(letter)}"
    if page > 1:
        url = f"{url}&page={page}"

    if random.random() < 0.1:
        logging.info(f"Fetching {url}")
    character_page = get_with_retries(session, url)
    parsed_page = BeautifulSoup(character_page.text, "html.parser")
    last_string = parsed_page.body.find("a", string=re.compile("Last ».*"))
    if not last_string:
        logger.warning(f"Unable to resolve location of last page in {url}")
        num_pages = None
    else:
        pages_match = re.search("page=(\d+)", last_string["href"])
        if not pages_match:
            raise RuntimeError(f"Unable to parse num pages from {last_string} in {url}")
        num_pages = int(pages_match.group(1))

    definitions = parsed_page.body.find_all("a", href=re.compile(".*define.php.*"), class_=None)
    if not definitions:
        raise RuntimeError(f"No definitions found for crawl of {url}!")

    definition_urls = [UrbanDictionaryWordURL(title=d.text, url=f"{UD_ROOT}{d['href']}") for d in definitions]

    return UrbanDictionaryIndexPage(url=url, definition_urls=definition_urls, num_pages=num_pages)


def fetch_all_letter_word_url(session, letter, limit=None):
    first_ip = fetch_letter_page(session, letter)

    if not first_ip.num_pages:
        raise RuntimeError(f"First page of {letter} lacks total number of pages!")

    all_definitions = OrderedDict((d.title, d) for d in first_ip.definition_urls)
    for i in range(2, first_ip.num_pages + 1):
        ip = fetch_letter_page(session, letter, page=i)
        all_definitions.update((d.title, d) for d in ip.definition_urls)

        if limit is not None and i > limit:
            break

    return all_definitions


def fetch_all_word_urls(session, limit=None):
    letters = list(string.ascii_uppercase) + ["*"]

    all_definitions = OrderedDict()
    for i, letter in enumerate(letters):
        logging.info(f"Starting fetch of words for {letter}")
        all_definitions.update(fetch_all_letter_word_url(session, letter))

        if limit is not None and i > limit:
            break

    return all_definitions


def _parse_definition_div(definition_div, url=None):
    word_as = definition_div.find_all("a", class_="word")
    if len(word_as) > 1:
        raise RuntimeError(f"Found more than one word in {url}")

    word_a = word_as[0]
    word_url = f'{UD_ROOT}{word_a["href"]}'
    word_title = word_a.text.strip()

    autolink_as = definition_div.find_all("a", class_="autolink")
    outbound_links = [UrbanDictionaryWordURL(url=f'{UD_ROOT}{a["href"]}', title=a.text.strip()) for a in autolink_as]

    meaning_divs = definition_div.find_all("div", class_="meaning")
    if len(meaning_divs) > 1:
        raise RuntimeError(f"Found more than one meaning in {url}")

    meaning = meaning_divs[0].get_text()

    tag_divs = definition_div.find_all("div", class_="tags")
    if len(tag_divs) > 1:
        raise RuntimeError(f"Found more than one tag div in {url}")
    elif len(tag_divs) == 0:
        tags = []
    else:
        tags = [e.text.strip() for e in tag_divs[0].find_all("a")]

    author_divs = definition_div.find_all("div", class_="contributor")
    if len(author_divs) > 1:
        raise RuntimeError(f"Found more than one author div in {url}")

    author_a = author_divs[0].find("a")
    author = author_a.text
    creation_date = author_a.next_sibling.strip()
    creation_epoch = datetime.datetime.strptime(creation_date, "%B %d, %Y").timestamp()

    example_divs = definition_div.find_all("div", class_="example")
    examples = [
        BeautifulSoup(re.sub("<br\s*?/?>", "\n", str(e)), "html.parser").get_text().replace("\r", "")
        for e in example_divs
    ]

    upvotes = int(definition_div.find("a", class_="up").find("span", class_="count").text)
    downvotes = int(definition_div.find("a", class_="down").find("span", class_="count").text)

    return UrbanDictionaryDefinition(
        word=word_title,
        url=word_url,
        meaning=meaning,
        outbound_links=outbound_links,
        tags=tags,
        author=author,
        creation_epoch=creation_epoch,
        examples=examples,
        upvotes=upvotes,
        downvotes=downvotes,
    )


def fetch_word(session, url):
    definition_page = get_with_retries(session, url)
    if definition_page.status_code != 200:
        raise RuntimeError("Unexpected status code")
    parsed_page = BeautifulSoup(definition_page.text, "html.parser")

    definitions = []
    definition_divs = parsed_page.find_all("div", class_="def-panel")
    definitions = [_parse_definition_div(d) for d in definition_divs]

    if len(definitions) == 0:
        raise RuntimeError(f"No definitions found for {url}")

    word = UrbanDictionaryWord(title=definitions[0].word, url=url, definitions=definitions)
    return word


def _fetch_word_lambda(session, word_url):
    try:
        return (word_url, fetch_word(session, word_url.url))
    except StatusError as e:
        logging.exception(f"Status error during scrape of {word_url.url}")
        return (None, None)


def fetch_all_definitions(
    session, to_fetch, already_done=None, save_interval=1000, save_path="all_words.pickle", executor=None
):
    already_done = already_done if already_done is not None else OrderedDict()
    fetch_list = list(to_fetch.values())
    pbar = tqdm(total=len(to_fetch) + len(already_done))
    pbar.update(len(already_done))

    mapper = executor.imap_unordered if executor else map
    for i, (word_url, word) in enumerate(mapper(partial(_fetch_word_lambda, session), fetch_list)):
        if word_url is None:
            logging.warning(f"Skipping due to upstream exception")
        elif word_url.title not in already_done and word_url.title not in to_fetch:
            logging.error(f"Warning: {word_url.title} from {word_url.url} missing from fetch / done list")
        else:
            already_done[word_url.title] = word
            del to_fetch[word_url.title]
        pbar.update()

        if i > 0 and i % save_interval == 0:
            with open(save_path, "wb") as f:
                pickle.dump(already_done, f, pickle.HIGHEST_PROTOCOL)
    pbar.close()
    return already_done


class SpecialTokens:
    BOS_TOKEN = "<|bod|>"
    EOS_TOKEN = "<|eod|>"
    PAD = "<|pad|>"

    TITLE_DEFINITION_SEP = "<|bt|>"
    DEFINITION_EXAMPLE_SEP = "<|et|>"

    @classmethod
    def special_tokens_dict(cls):
        return {
            "bos_token": cls.BOS_TOKEN,
            "eos_token": cls.EOS_TOKEN,
            "pad_token": cls.PAD,
            "additional_special_tokens": [cls.TITLE_DEFINITION_SEP, cls.DEFINITION_EXAMPLE_SEP],
        }


class UrbanDictionaryDataset(Dataset):
    @classmethod
    def _make_examples(cls, tokenizer, word):
        max_len = tokenizer.max_len_single_sentence

        # Adding prefix space to beginning as docs suggest
        bos_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(SpecialTokens.BOS_TOKEN))
        eos_token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(SpecialTokens.EOS_TOKEN))
        title_definition_sep_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(SpecialTokens.TITLE_DEFINITION_SEP)
        )
        definition_example_sep_ids = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(SpecialTokens.DEFINITION_EXAMPLE_SEP)
        )

        max_nonspecial_len = max_len - (
            len(bos_token_ids) + len(eos_token_ids) + len(title_definition_sep_ids) + len(definition_example_sep_ids)
        )

        examples = []

        for definition in word.definitions:
            # TODO: do I need this ?
            title_tokenization = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(definition.word, add_prefix_space=True)
            )
            meaning_tokenization = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(definition.meaning, add_prefix_space=True)
            )
            example_tokenization = tokenizer.convert_tokens_to_ids(
                tokenizer.tokenize(definition.examples[0], add_prefix_space=True)
            )

            if len(title_tokenization) > max_nonspecial_len:
                logger.error(f"Title of word '{definition.word}' too long to tokenize, skipping")
                continue

            max_len_meaning_example = max_nonspecial_len - len(title_tokenization)

            if len(meaning_tokenization) + len(example_tokenization) < max_len_meaning_example:
                max_meaning_len = len(meaning_tokenization)
                max_example_len = len(example_tokenization)
            elif len(example_tokenization) < max_len_meaning_example:
                max_meaning_len = max_len_meaning_example - len(example_tokenization)
                max_example_len = len(example_tokenization)
            elif len(meaning_tokenization) < max_len_meaning_example:
                max_meaning_len = len(meaning_tokenization)
                max_example_len = max_len_meaning_example - len(meaning_tokenization)
            else:
                logger.warning(
                    f"{definition.word}: both example and meaning exceed tokenization length, truncating both"
                )
                max_meaning_len = max_len_meaning_example // 2
                max_example_len = max_len_meaning_example - max_meaning_len

            example = list(
                itertools.chain(
                    bos_token_ids,
                    title_tokenization,
                    title_definition_sep_ids,
                    meaning_tokenization[:max_meaning_len],
                    definition_example_sep_ids,
                    example_tokenization[:max_example_len],
                    eos_token_ids,
                )
            )

            bool_mask = [
                bool(i >= len(bos_token_ids) and i < (len(bos_token_ids) + len(title_tokenization)))
                for i in range(len(example))
            ]

            assert len(example) <= max_len, f"Example should be less than max length: {len(example)} Vs. {max_len}"

            examples.append((example, bool_mask))

        return examples

    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, file_path: str, splits=(1.0), split_idx=0,
    ):
        assert os.path.isfile(file_path) or os.path.islink(file_path)
        directory, filename = os.path.split(file_path)

        cached_features_file = os.path.join(
            directory,
            args.model_type
            + "_cached_lm_splits_"
            + "_".join(str(e) for e in splits)
            + "_split_idx_"
            + str(split_idx)
            + "_"
            + filename,
        )

        splits_tensor = torch.tensor(splits)
        sum_splits = torch.cumsum(splits_tensor, 0)

        if sum_splits[-1] != 1.0:
            raise RuntimeError(f"Splits must sum to 1 (actual: {sum_splits[-1]})")
        elif split_idx >= len(sum_splits):
            raise RuntimeError(f"Invalid split index {split_idx} (must be less than {len(sum_splits)})")

        if split_idx == 0:
            start_range = 0.0
        else:
            start_range = sum_splits[split_idx - 1]

        end_range = sum_splits[split_idx]

        def in_split(ud_word):
            val = int(hashlib.md5(ud_word.title.encode("utf-8")).hexdigest(), 16,) % 10000 / 10000
            return (val >= start_range and val < end_range).item()

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.info("Loaded {len(self.examples)} features")
        else:
            logger.info(
                f"Cache at {cached_features_file} not found... creating features from dataset file at %s", directory
            )

            self.examples = []

            with open(file_path, "rb") as f:
                words = list(pickle.load(f).values())

            for word in words:
                if in_split(word):
                    self.examples.extend(self._make_examples(tokenizer, word))

            logger.info(f"Saving {len(self.examples)} features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (
            torch.tensor(self.examples[item][0], dtype=torch.long),
            torch.tensor(self.examples[item][1], dtype=torch.bool),
        )
