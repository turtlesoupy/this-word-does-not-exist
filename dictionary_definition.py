from struct import unpack
from zlib import decompress
import sys
import re
import hashlib
import bs4
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import logging
import os
import torch
import pickle
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


# Helpers for beautiful soup
def find_at_most_one(bs, *args, **kwargs):
    t = bs.find_all(*args, **kwargs)
    if not t:
        return t
    elif len(t) > 1:
        raise InvalidParseAssumptionError("Too many found!")
    else:
        return t[0]


def find_exactly_one(bs, *args, **kwargs):
    t = bs.find_all(*args, **kwargs)
    if not t:
        raise InvalidParseAssumptionError("Not enough tags found!")
    elif len(t) > 1:
        raise InvalidParseAssumptionError("Too many found!")
    else:
        return t[0]


def find_at_least_one(bs, *args, **kwargs):
    t = bs.find_all(*args, **kwargs)
    if not t:
        raise InvalidParseAssumptionError("Not enough tags found!")
    return t


class InvalidParseAssumptionError(RuntimeError):
    pass


@dataclass
class Pronounciation:
    text: str
    type: str


@dataclass
class Definition:
    pos_modifier: Optional[str]
    definition: str
    examples: List[str]
    topic: Optional[str]
    dates: List[str]


@dataclass
class ReferenceDefinition:
    pos_modifier: Optional[str]
    reference: str


@dataclass
class Sense:
    pos: Optional[str]
    definitions: List[Definition]


@dataclass
class Entry:
    word: str
    variant: Optional[int]
    senses: List[Sense]
    pronounciations: List[Pronounciation]
    phrases: List[Definition]
    phrasal_verbs: List[Tuple[str, List[Definition]]]
    origin: Optional[str]
    derivatives: List[str]
    notes: List[str]


@dataclass
class DictionaryDefinition:
    title: str
    entry_str: str
    parsed_entry: Optional[bs4.BeautifulSoup] = None

    @classmethod
    def gen_from_apple_dictionary(cls, f):
        f.seek(0x40)
        limit = 0x40 + unpack("i", f.read(4))[0]
        f.seek(0x60)
        while f.tell() < limit:
            (sz,) = unpack("i", f.read(4))
            buf = decompress(f.read(sz)[8:])
            for m in re.finditer(b"<d:entry[^\n]+", buf):
                entry = m.group().decode()
                title = re.search('d:title="(.*?)"', entry).group(1)
                title_soup = bs4.BeautifulSoup(title, features="html.parser")
                entry_soup = bs4.BeautifulSoup(entry, features="html.parser")

                title = title_soup.get_text()
                entry = entry_soup.get_text()

                if not title or not entry:
                    logger.warning(f"Invalid entry {title}: {entry}")
                    continue

                yield cls(
                    title=title_soup.get_text(), entry_str=entry_soup.get_text(), parsed_entry=entry_soup,
                )


class AppleDictParser:
    @classmethod
    def parse_pronounciations(cls, parsed_entry):
        pronounciation_encloses = [e for e in parsed_entry.find_all("span", class_="prx") if e.get_text().strip()]
        if len(pronounciation_encloses) == 0:
            pronounciation_encloses = [e for e in parsed_entry.find_all("span", class_="pr") if e.get_text().strip()]

        if not pronounciation_encloses:
            return None

        ret = []
        for pronounciation_enclose in pronounciation_encloses:
            pronounciations = pronounciation_enclose("span", class_="ph")
            if not pronounciations:
                raise InvalidParseAssumptionError(f"No pronounciations found")

            ret.extend([Pronounciation(text=p.get_text(), type=p["d:pr"]) for p in pronounciations])
        return ret

    @classmethod
    def parse_sense_definitions(cls, parsed_entry):
        global_pos_modifier_span = parsed_entry.find_all("span", class_="gg")
        global_pos_modifier_span = [
            e for e in global_pos_modifier_span if not e.find_parents("span", class_="msDict")
        ]  # Filter out local ones
        global_pos_modifier = global_pos_modifier_span[0].get_text().strip() if global_pos_modifier_span else None
        definitions = []
        entry_spans = find_at_least_one(parsed_entry, "span", class_="msDict")
        for entry_span in entry_spans:
            if not entry_span.get_text().strip().strip("•"):  # Some malformed entries, e.g. thrash
                continue

            definition_spans = entry_span.find_all("span", class_="df")
            xrg_spans = entry_span.find_all("span", class_="xrg")
            if definition_spans:
                for definition_span in definition_spans:
                    example_spans = entry_span("span", class_="ex")
                    topic_spans = [
                        e for e in entry_span("span", class_="lg") if not e.find_parents("span", class_="eg")
                    ]
                    if len(topic_spans) > 1:
                        logging.warning(f"Too many topics found: {topic_spans}, picking first one")

                    local_pos_modifier_span = entry_span.find("span", class_="gg", recursive=False)
                    local_pos_modifier = local_pos_modifier_span and local_pos_modifier_span.get_text().strip()

                    definition = definition_span.get_text().strip()
                    examples = [e.get_text().strip().strip(":").strip() for e in example_spans]
                    topic = topic_spans and topic_spans[0].get_text().strip()

                    date_spans = definition_span("span", class_="dg")
                    dates = [e.get_text().strip() for e in date_spans]

                    definitions.append(
                        Definition(
                            pos_modifier=local_pos_modifier or global_pos_modifier,
                            definition=definition,
                            examples=examples,
                            topic=topic,
                            dates=dates,
                        )
                    )
            elif xrg_spans:
                for xrg in xrg_spans:
                    referenced_terms = find_at_least_one(xrg, "span", class_="xr")

                    for referenced_term in referenced_terms:
                        reference = referenced_term.get_text().strip()
                        definitions.append(ReferenceDefinition(pos_modifier=global_pos_modifier, reference=reference,))
            elif entry_span.find("span", class_="ex"):
                logger.warning(f"Silently ignoring example without corresponding definition {entry_span}")
            else:
                raise InvalidParseAssumptionError(f"Weird span: {entry_span}")

        return definitions

    @classmethod
    def parse_sense(cls, parsed_entry):
        pos_spans = parsed_entry("span", class_="tg_pos")
        if len(pos_spans) > 1:
            pos = " ".join([e.get_text().strip() for e in pos_spans])
        elif not pos_spans:
            pos_span = find_at_most_one(parsed_entry, "span", class_="posg")
            pos = pos_span.get_text().strip() if pos_span else None
        else:
            pos = pos_spans[0].get_text().strip()

        if parsed_entry.findChildren("span", class_="se2"):
            sense_definitions = []
            for c in parsed_entry.children:
                if set(c["class"]) & set(("tg_pos", "posg", "x_xdh")):
                    continue
                elif not c.get_text().strip():
                    continue
                elif "se2" in c["class"]:
                    sense_definitions.extend(cls.parse_sense_definitions(c))
                elif "note" in c["class"]:
                    logger.warning(f"Dropping note in word sense {c}")
                elif "msDict" in c["class"]:
                    logger.warning(f"Dropping unexpected msDict in sense {c}")
                else:
                    raise InvalidParseAssumptionError(f"WEIRD TAG: {c}")
        else:
            sense_definitions = cls.parse_sense_definitions(parsed_entry)

        if not sense_definitions:
            raise InvalidParseAssumptionError("No sense definitions!")
        return Sense(pos=pos, definitions=sense_definitions)

    @classmethod
    def parse_derivatives(cls, parsed_entry):
        words = find_at_least_one(parsed_entry, "span", class_="l")
        return [e.get_text().strip() for e in words]

    @classmethod
    def parse_origin(cls, parsed_entry):
        etym_type = find_exactly_one(parsed_entry, "span", class_="tg_etym", recursive=False)
        if etym_type.get_text().strip() != "ORIGIN":
            raise InvalidParseAssumptionError(f"Unexpected etym type: {etym_type}")

        origin_span = find_exactly_one(parsed_entry, "span", class_="x_xo1")
        origin = origin_span.get_text().strip()
        return origin

    @classmethod
    def parse_phrasal_verbs(cls, parsed_entry):
        subentries = find_at_least_one(parsed_entry, "span", class_="subEntry")
        ret = []
        for subentry in subentries:
            word_span = subentry.find("span", class_="x_xoh") or subentry.find("span", class_="x_xot")
            if subentry.find("span", class_="msDict"):
                definitions = cls.parse_sense_definitions(subentry)
            else:
                definitions = []
            ret.append((word_span.get_text().strip(), definitions))
        return ret

    @classmethod
    def parse(cls, parsed_entry):
        entry = find_exactly_one(parsed_entry, "d:entry")
        head_entry = find_exactly_one(entry, "span", class_="hg")
        defn_entry = find_exactly_one(entry, "span", class_="sg")

        head_word_span = find_exactly_one(head_entry, "span", class_="hw")
        word = " ".join([t.strip() for t in head_word_span.contents if type(t) == bs4.element.NavigableString]).strip()

        variant_span = find_at_most_one(head_word_span, "span", class_="tg_hw")
        variant = int(variant_span.get_text()) if variant_span else None

        pronounciations = cls.parse_pronounciations(head_entry)

        senses = defn_entry("span", class_="se1")
        if len(senses) == 0:
            raise InvalidParseAssumptionError(f"No senses found!")

        senses = []
        for c in defn_entry.children:
            if "se1" in c["class"]:
                senses.append(cls.parse_sense(c))
            elif c.get_text().strip():
                raise InvalidParseAssumptionError(f"Weird tag found in definition: {c.prettify()}!")

        phrases = []
        origin = None
        subentries = entry.find_all("span", class_="t_derivatives")  # derivatives # TODO: verify
        derivatives = []
        phrasal_verbs = []
        notes = []

        for subentry in entry.children:
            if subentry == head_entry or subentry == defn_entry:
                continue
            elif "t_phrases" in subentry["class"]:
                phrases = cls.parse_sense_definitions(subentry)
            elif "t_derivatives" in subentry["class"]:
                derivatives = cls.parse_derivatives(subentry)
            elif "t_phrasalVerbs" in subentry["class"]:
                phrasal_verbs = cls.parse_phrasal_verbs(subentry)
            elif "etym" in subentry["class"]:
                origin = cls.parse_origin(subentry)
            elif "note" in subentry["class"]:
                notes.append(subentry.get_text())
            else:
                raise InvalidParseAssumptionError(f"Weird entry found: {subentry}")

        # TODO: determine other direct children types
        return Entry(
            word=word,
            variant=variant,
            pronounciations=pronounciations,
            senses=senses,
            phrases=phrases,
            phrasal_verbs=phrasal_verbs,
            origin=origin,
            derivatives=derivatives,
            notes=notes,
        )


def generate_words(
    tokenizer,
    model,
    allow_proper_nouns=True,
    blacklist=(),
    prefix="<title>",
    num=100,
    batch_size=50,
    max_length=400,
    max_iterations=20,
):
    ret = []
    num_iteration = 0

    input = tokenizer.encode(prefix, return_tensors="pt").to("cuda")

    while len(ret) < num and num_iteration < max_iterations:
        num_iteration += 1

        generated = model.generate(input, max_length=max_length, num_return_sequences=batch_size, temperature=1.0,)
        valid_i = 0

        for i in range(generated.size()[0]):
            sentence_tokens = generated[i, :].tolist()
            decoded = tokenizer.decode(sentence_tokens)
            m = re.search(r"<title>(.*?)</title>(.*)", decoded)
            if m:
                title = m.group(1).strip()
                if not allow_proper_nouns and title[:1].upper() == title[:1]:
                    continue
                elif title.upper() in blacklist or title.upper().rstrip("s") in blacklist:
                    continue
                else:
                    ret.append(DictionaryDefinition(title=title, entry_str=m.group(2).rstrip("!")))
            else:
                logger.warning(f'Unable to match regex in "{decoded}"')

    return ret
