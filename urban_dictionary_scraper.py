import re
import logging
import datetime
import random
import time
import requests
import requests_cache
import string
import urllib
import asyncio

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


@retry(exceptions=(ConnectionError, requests.exceptions.Timeout), tries=5, delay=3, logger=logger)
def get_with_retries(session, url, timeout=10.0):
    ret = session.get(url, timeout=timeout)

    if ret.status_code != 200:
        raise RuntimeError(f"Unexpected status code in {url}")

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
