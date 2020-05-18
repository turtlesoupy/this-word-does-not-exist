from dataclasses import dataclass
import os
import logging
import pickle
import hashlib
import itertools
import title_maker_pro.custom_modeling_utils as custom_modeling_utils
import title_maker_pro.dictionary_definition as dictionary_definition
import re
import torch
import random
import stanza
import time
import sys
from collections import Counter
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer
from typing import NamedTuple, List, Optional
from io import StringIO

logger = logging.getLogger(__name__)

oed_to_upos = {
    "exclamation": ("NOUN", "PROPN", "VERB", "NUM", "SYM", "X"),
    "abbreviation": ("X", "NOUN", "PROPN", "NUM", "SYM"),
    "noun": ("NOUN", "PROPN"),
    "adjective adverb": ("ADJ", "ADVERB"),
    "adjective": ("ADJ",),
    "verb": ("VERB"),
    "adverb": ("ADVERB"),
    "prefix": ("ADJ", "ADVERB"),
    "conjunction": ("AUX"),
    "pronoun": ("INTJ", "NOUN"),
}


def _access_zero_assert(item):
    if len(item) != 1:
        raise RuntimeError("Expected length 1 in item")

    return item[0]


def _read_in_chunks(stream, chunk_size=1 * 1024 * 1024):
    while True:
        data = stream.read(chunk_size)
        if not data:
            break
        yield data


@dataclass
class GeneratedWord:
    word: str
    pos: Optional[str]
    topic: Optional[str]
    definition: str
    example: Optional[str]
    decoded: Optional[str]
    decoded_tokens: Optional[List[int]]

    @classmethod
    def print_words(cls, words, f=sys.stdout):
        for word in words:
            word_str = [word.word]
            if word.pos:
                word_str.append(f"/{word.pos}/")
            if word.topic:
                word_str.append(f"[{word.topic}]")
            print(" ".join(word_str), file=f)
            print(f"\t{word.definition}{' |n| ' if word.example is None else ''}", file=f)
            if word.example:
                cleaned_example = word.example.replace("\n", "\n\t")
                print(f"\t\"{cleaned_example}\"", file=f)

            print("----------------", file=f)


@dataclass
class GeneratedWordCandidate:
    score: float
    candidate: GeneratedWord


class Blacklist:
    def __init__(self, blacklist_set):
        self.blacklist_set = blacklist_set

    def merge(self, other):
        self.blacklist_set |= other.blacklist_set
        return self

    def contains(self, word, recursive=True):
        word = word.strip().lower()
        return (
            word in self.blacklist_set
            or re.sub(r"('s|s|ing|')$", "", word) in self.blacklist_set
            or (recursive and all(self.contains(e, recursive=False) for e in word.split()))
            or (recursive and all(self.contains(e, recursive=False) for e in word.split("-")))
        )

    def collapse_hyphens(self):
        self.blacklist_set |= {"".join(e.split()) for e in self.blacklist_set}
        self.blacklist_set |= {"".join(e.split("-")) for e in self.blacklist_set}

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    @classmethod
    def from_text_lines(cls, stream):
        return cls(set(e.strip().lower() for e in stream))

    @classmethod
    def from_text_stream(cls, stream, min_threshold=3, chunk_size=1024 * 1024, use_gpu=False, sample_rate=1.0):
        cnt = Counter()
        for i, chunk in enumerate(_read_in_chunks(stream, chunk_size)):
            if sample_rate != 1.0 and random.random() > sample_rate:
                continue

            pipe = stanza.Pipeline(lang="en", processors="tokenize", use_gpu=use_gpu)
            res = pipe(chunk)
            two_prev = None
            prev = None
            for w in res.iter_words():
                cnt[w.text.lower()] += 1
                if prev:
                    cnt[f"{prev} {w.text}".lower()] += 1
                if two_prev:
                    cnt[f"{two_prev} {prev} {w.text}".lower()] += 1
                two_prev = prev
                prev = w.text

        ret = cls(set(k for k, v in cnt.items() if v > min_threshold))
        return ret

    @classmethod
    def from_parsed_dictionary(cls, path):
        blacklist = set(
            (
                x.lower()
                for x in itertools.chain.from_iterable([e.word] + e.derivatives for e in pickle.load(open(path, "rb")))
            )
        )
        return cls(blacklist)

    @classmethod
    def from_urban_dictionary(cls, path, loaded=None):
        class RenamingUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'urban_dictionary_scraper':
                    module = 'title_maker_pro.urban_dictionary_scraper'
                return super().find_class(module, name)

        with open(path, "rb") as f:
            ordered_dict = (loaded or RenamingUnpickler(f).load())
            blacklist = set(
                (
                    x.word.lower()
                    for x in itertools.chain.from_iterable(e.definitions for e in ordered_dict.values())
                )
            )
        return cls(blacklist)

    def dump(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.blacklist_set, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.blacklist_set)


def _len_range_overlap(x, y):
    start = max(x[0], y[0])
    end = min(x[-1], y[-1]) + 1
    return max(0, end - start)


def _split_range(splits, split_idx):
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

    return (start_range, end_range)


def _in_split_range(split_range, randomizer_str):
    start_range, end_range = split_range
    val = int(hashlib.md5(randomizer_str.encode("utf-8")).hexdigest(), 16,) % 100000 / 100000
    return (val >= start_range and val < end_range).item()


def _cache_path(class_name, base_directory, filename, **keys):
    path = [class_name]
    for k, v in keys.items():
        if isinstance(v, str):
            path.append(f"{k}-{v}")
            continue

        try:
            path.append(f"{k}-{'-'.join(str(e) for e in iter(v))}")
            continue
        except TypeError:
            pass

        path.append(f"{k}-{str(v)}")

    path.append(filename)
    return os.path.join(base_directory, "__".join(path))


class TokenGroup(NamedTuple):
    separator: List[int] = []
    payload: List[int] = []
    remove_if_truncated: bool = False


def _join_and_truncate(
    max_len: int, begin_tokens: List[int], token_groups: List[TokenGroup], end_tokens: List[int], min_append_size=5,
):
    if len(begin_tokens) + len(end_tokens) > max_len:
        raise RuntimeError("Length is too small for required tokens")

    running_max_len = max_len - len(begin_tokens) - len(end_tokens)

    ret = [begin_tokens]

    for token_group in token_groups:
        if len(token_group.separator) + len(token_group.payload) > running_max_len:
            if token_group.remove_if_truncated:
                break

            if running_max_len - len(token_group.separator) - len(token_group.payload) < min_append_size:
                break

            ret.append(token_group.separator)
            running_max_len -= len(token_group.separator)
            ret.append(token_group.payload[:running_max_len])
            running_max_len = 0
            break
        else:
            ret.append(token_group.separator)
            ret.append(token_group.payload)
            running_max_len -= len(token_group.separator) + len(token_group.payload)

    ret.append(end_tokens)
    return list(itertools.chain.from_iterable(ret))


class SpecialTokens:
    BOS_TOKEN = "<|bod|>"
    EOS_TOKEN = "<|eod|>"
    PAD = "<|pad|>"

    DEFINITION_SEP = "<|bd|>"
    EXAMPLE_SEP = "<|be|>"
    POS_SEP = "<|pos|>"
    TOPIC_SEP = "<|bto|>"

    @classmethod
    def special_tokens_dict(cls):
        return {
            "bos_token": cls.BOS_TOKEN,
            "eos_token": cls.EOS_TOKEN,
            "pad_token": cls.PAD,
            "additional_special_tokens": [cls.DEFINITION_SEP, cls.EXAMPLE_SEP, cls.POS_SEP, cls.TOPIC_SEP],
        }

@dataclass
class GenerationStats:
    num_iterations: int = 0

    num_items_considered: int = 0
    num_failed_match: int = 0
    num_blacklist_filtered: int = 0
    num_seen_filtered: int = 0
    num_proper_noun_filtered: int = 0

    num_example_missing: int = 0

    num_user_filtered: int = 0
    num_returned: int = 0
    num_example_pos_match_failed: int = 0
    num_example_missing_title: int = 0
    num_short_definitions: int = 0
    wall_time: float = 0.0
    wall_stanza_time: float = 0.0

    def __str__(self):
        return (
            f"iterations={self.num_iterations} time={self.wall_time} stanza_time={self.wall_stanza_time} | "
            + ", ".join(
                f"{k} {v / self.num_items_considered:.2f}@{v}"
                for k, v in (
                    ("items_considered", self.num_items_considered),
                    ("failed_match", self.num_failed_match),
                    ("blacklist_filtered", self.num_blacklist_filtered),
                    ("seen_filtered", self.num_seen_filtered),
                    ("proper_noun_filtered", self.num_proper_noun_filtered),
                    ("example_missing", self.num_example_missing),
                    ("short_definitions", self.num_short_definitions),
                    ("example_missing_title", self.num_example_missing_title),
                    ("example_pos_match_failed", self.num_example_pos_match_failed),
                    ("user_filtered", self.num_user_filtered),
                    ("returned", self.num_returned),
                )
            )
        )


class ParsedDictionaryDefinitionDataset(Dataset):
    @classmethod
    def _split_re(cls):
        split_re_pat = (
            f"^{re.escape(SpecialTokens.BOS_TOKEN)}(?P<title>.+?)"
            f"(?:{re.escape(SpecialTokens.POS_SEP)}(?P<pos>.+?))?"
            f"(?:{re.escape(SpecialTokens.TOPIC_SEP)}(?P<topic>.+?))?"
            f"{re.escape(SpecialTokens.DEFINITION_SEP)}(?P<definition>.+?)"
            f"(?:{re.escape(SpecialTokens.EXAMPLE_SEP)}(?P<example>.+?))*"
            f"{re.escape(SpecialTokens.EOS_TOKEN)}"
        )
        split_re = re.compile(split_re_pat, flags=re.MULTILINE | re.DOTALL)
        return split_re

    @classmethod
    def approx_pos(cls, nlp, sentence, lookup_idx, lookup_len):
        start_end_re = re.compile(r"start_char=(\d+)\|end_char=(\d+)")
        doc = nlp(sentence)
        uposes = Counter()

        for sentence in doc.sentences:
            for word in sentence.words:
                m = start_end_re.match(word.misc)
                if not m:
                    raise RuntimeError("Unable to extract start and end positions!")
                start_char = int(m.group(1))
                end_char = int(m.group(2))
                uposes[word.upos] += _len_range_overlap(
                    (lookup_idx, lookup_idx + lookup_len - 1), (start_char, end_char - 1)
                )

        ((tag, _),) = uposes.most_common(1)
        return tag

    @classmethod
    def evaluate_creativity(cls, tokenizer, model, blacklist, num_to_generate, batch_size, max_length):
        input = tokenizer.encode(SpecialTokens.BOS_TOKEN, return_tensors="pt").to(model.device)
        split_re = cls._split_re()
        num_generated = 0
        num_failed_match = 0
        num_succeeded_match = 0
        num_blacklisted = 0

        for i in tqdm(list(range(0, num_to_generate, batch_size)), desc="Evaluating Creativity"):
            generated = model.generate(
                input,
                max_length=max_length,
                num_return_sequences=batch_size,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for i in range(generated.size()[0]):
                sentence_tokens = generated[i, :].tolist()
                num_generated += 1
                decoded = tokenizer.decode(sentence_tokens)
                m = split_re.match(decoded)
                if not m:
                    num_failed_match += 1
                    continue

                num_succeeded_match += 1
                title = m.group("title")
                if blacklist and blacklist.contains(title):
                    num_blacklisted += 1

        return {
            "creative_words": 1 - (num_blacklisted / max(num_succeeded_match, 1)),
            "nonconforming": num_failed_match / num_generated,
        }

    @classmethod
    def generate_words(
        cls,
        tokenizer,
        model,
        prefix=SpecialTokens.BOS_TOKEN,
        num=100,
        max_iterations=10,
        generation_args={},
        blacklist=None,
        example_title_match=True,
        example_match_pos_pipeline=None,
        dedupe_titles=True,
        user_filter=None,
        filter_proper_nouns=False,
        use_custom_generate=True,
        min_definition_words=3,
    ):
        start = time.time()
        viable_candidates = []
        ret = []
        num_iteration = 0
        if isinstance(prefix, str):
            input = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
        else:
            input = torch.tensor([prefix], dtype=torch.long).to(model.device)

        pos_sep_id = _access_zero_assert(tokenizer.encode(SpecialTokens.POS_SEP))
        example_sep_id = _access_zero_assert(tokenizer.encode(SpecialTokens.EXAMPLE_SEP))
        topic_sep_id = _access_zero_assert(tokenizer.encode(SpecialTokens.TOPIC_SEP))
        definition_sep_id = _access_zero_assert(tokenizer.encode(SpecialTokens.DEFINITION_SEP))

        split_re = cls._split_re()
        seen_titles = set()
        stats = GenerationStats()
        t = tqdm(total=num)
        while len(ret) < num and num_iteration < max_iterations:
            num_iteration += 1
            stats.num_iterations += 1
            if not use_custom_generate:
                generated = model.generate(
                    input,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_args,
                )
            else:

                def partial_generation_transform(input_ids, tokens_to_add):
                    for i in range(tokens_to_add.size()[0]):
                        if blacklist and tokens_to_add[i] in (pos_sep_id, topic_sep_id, definition_sep_id):
                            word = tokenizer.decode(input_ids[i, :][1:])
                            if blacklist.contains(word):
                                tokens_to_add[i] = tokenizer.eos_token_id
                        elif tokens_to_add[i] == tokenizer.eos_token_id:
                            example_token_idxs = input_ids[i, :] == example_sep_id
                            if example_token_idxs.max() == 0:
                                tokens_to_add[i] = example_sep_id

                    return tokens_to_add

                generated = custom_modeling_utils.custom_generate(
                    model,
                    input,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    partial_generation_transform=partial_generation_transform,
                    **generation_args,
                )

            for i in range(generated.size()[0]):
                if len(ret) >= num:
                    break
                viable_candidates = viable_candidates[:1000]

                stats.num_items_considered += 1
                sentence_tokens = generated[i, :].tolist()
                decoded = tokenizer.decode(sentence_tokens)

                m = split_re.match(decoded)
                if not m:
                    stats.num_failed_match += 1
                    continue

                title = m.group("title")
                definition = m.group("definition")
                topic = m.group("topic")
                pos = m.group("pos")
                example = m.group("example")

                generated_word = GeneratedWord(
                    word=title and title.strip(),
                    definition=definition and definition.strip(),
                    example=example and example.strip(),
                    pos=pos and pos.strip(),
                    topic=topic and topic.strip(),
                    decoded=decoded,
                    decoded_tokens=sentence_tokens,
                )

                if blacklist and blacklist.contains(title):
                    stats.num_blacklist_filtered += 1
                    continue

                if dedupe_titles and title.strip().lower() in seen_titles:
                    stats.num_seen_filtered += 1
                    continue

                if filter_proper_nouns and title.strip()[:1].isupper():
                    stats.num_proper_noun_filtered += 1
                    continue

                if not example or not example.strip():
                    stats.num_example_missing += 1
                    viable_candidates.append(GeneratedWordCandidate(0.0, generated_word))
                    continue

                if len(definition.split()) < min_definition_words:
                    stats.num_short_definitions += 1
                    viable_candidates.append(GeneratedWordCandidate(0.2, generated_word))
                    continue

                t_rstrip = title.strip().lower().rstrip("s")
                l_example = example.lower()
                try:
                    start_title_idx = l_example.index(t_rstrip)
                    if pos and example_match_pos_pipeline:
                        pos_removed = re.sub(r"\[.*\]", "", pos).strip()
                        pos_removed = re.sub(r"plural", "", pos_removed).strip()
                        start_stanza = time.time()
                        pos_guess = cls.approx_pos(
                            example_match_pos_pipeline, l_example, start_title_idx, len(t_rstrip)
                        )
                        stats.wall_stanza_time += time.time() - start_stanza

                        if pos_removed not in oed_to_upos:
                            logger.warn(f"No UPOS mapping for {pos_removed} - {title} in '{example}'': {pos_guess}")
                            stats.num_example_pos_match_failed += 1
                            viable_candidates.append(GeneratedWordCandidate(0.9, generated_word))
                            continue
                        elif pos_guess not in oed_to_upos[pos_removed]:
                            stats.num_example_pos_match_failed += 1
                            viable_candidates.append(GeneratedWordCandidate(1.0, generated_word))
                            continue

                except ValueError:
                    stats.num_example_missing_title += 1
                    viable_candidates.append(GeneratedWordCandidate(0.5, generated_word))
                    continue

                if user_filter and not user_filter(generated_word):
                    stats.num_user_filtered += 1
                    continue
                else:
                    t.update()
                    ret.append(generated_word)
                    seen_titles.add(generated_word.word.lower())

        stats.num_returned = len(ret)
        stats.viable_candidates = viable_candidates
        stats.wall_time = time.time() - start
        return ret[:num], stats

    def _make_examples(self, tokenizer, entry: dictionary_definition.Entry):
        examples = []
        for sense in entry.senses:
            for definition in sense.definitions:
                if isinstance(definition, dictionary_definition.ReferenceDefinition):
                    continue

                token_groups = []
                token_groups.append(TokenGroup(separator=[], payload=tokenizer.encode(entry.word)))

                if sense.pos:
                    if definition.pos_modifier:
                        payload = tokenizer.encode(f"{sense.pos} {definition.pos_modifier}")
                    else:
                        payload = tokenizer.encode(sense.pos)

                    token_groups.append(TokenGroup(separator=self.pos_sep_ids, payload=payload))

                if definition.topic:
                    token_groups.append(
                        TokenGroup(separator=self.topic_sep_ids, payload=tokenizer.encode(definition.topic),)
                    )

                token_groups.append(
                    TokenGroup(
                        separator=self.definition_sep_ids, payload=tokenizer.encode(definition.definition.rstrip(". ")),
                    )
                )

                for example in definition.examples:
                    token_groups.append(
                        TokenGroup(
                            separator=self.example_sep_ids, payload=tokenizer.encode(example), remove_if_truncated=True,
                        )
                    )

                example = _join_and_truncate(
                    max_len=self.max_len,
                    begin_tokens=self.bos_token_ids,
                    end_tokens=self.eos_token_ids,
                    token_groups=token_groups,
                )

                assert (
                    len(example) <= self.max_len
                ), f"Example should be less than max length: {len(example)} Vs. {self.max_len}"

                examples.append(example)

        return examples

    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, splits=(1.0), split_idx=0):
        self.max_len = min(tokenizer.max_len_single_sentence, args.block_size)
        self.bos_token_ids = tokenizer.encode(SpecialTokens.BOS_TOKEN)
        self.eos_token_ids = tokenizer.encode(SpecialTokens.EOS_TOKEN)
        self.pos_sep_ids = tokenizer.encode(SpecialTokens.POS_SEP)
        self.definition_sep_ids = tokenizer.encode(SpecialTokens.DEFINITION_SEP)
        self.example_sep_ids = tokenizer.encode(SpecialTokens.EXAMPLE_SEP)
        self.topic_sep_ids = tokenizer.encode(SpecialTokens.TOPIC_SEP)

        assert os.path.isfile(file_path) or os.path.islink(file_path)
        directory, filename = os.path.split(file_path)

        cached_features_file = _cache_path(
            self.__class__.__name__,
            directory,
            filename,
            model_type=args.model_type,
            splits=splits,
            split_idx=split_idx,
            max_len=self.max_len,
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.info("Loaded {len(self.examples)} features")
        else:
            logger.info(
                f"Cache at {cached_features_file} not found... creating features from dataset file at %s", directory,
            )

            self.examples = []
            split_range = _split_range(splits, split_idx)

            with open(file_path, "rb") as f:
                entries = pickle.load(f)

            for entry in entries:
                if _in_split_range(split_range, entry.word):
                    self.examples.extend(self._make_examples(tokenizer, entry))

            logger.info(f"Saving {len(self.examples)} features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class InverseParsedDictionaryDefinitionDataset(Dataset):
    @classmethod
    def _split_re(self):
        split_re_pat = (
            f"^{re.escape(SpecialTokens.BOS_TOKEN)}(?P<definition>.+?)"
            f"{re.escape(SpecialTokens.DEFINITION_SEP)}(?P<title>.+?)"
            f"(?:{re.escape(SpecialTokens.POS_SEP)}(?P<pos>.+?))?"
            f"(?:{re.escape(SpecialTokens.TOPIC_SEP)}(?P<topic>.+?))?"
            f"(?:{re.escape(SpecialTokens.EXAMPLE_SEP)}(?P<example>.+?))*"
            f"{re.escape(SpecialTokens.EOS_TOKEN)}"
        )
        split_re = re.compile(split_re_pat, flags=re.MULTILINE | re.DOTALL)
        return split_re

    @classmethod
    def generate_words(
        cls,
        tokenizer,
        model,
        prefix=SpecialTokens.BOS_TOKEN,
        num=100,
        max_iterations=10,
        generation_args={},
        blacklist=(),
        dedupe_titles=True,
        user_filter=None,
    ):
        ret = []
        num_iteration = 0
        if isinstance(prefix, str):
            input = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
        else:
            input = torch.tensor([prefix], dtype=torch.long).to(model.device)

        split_re = cls._split_re()
        seen_titles = set()
        t = tqdm(total=num)
        while len(ret) < num and num_iteration < max_iterations:
            num_iteration += 1
            generated = model.generate(
                input,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **generation_args,
            )

            for i in range(generated.size()[0]):
                if len(ret) >= num:
                    break

                sentence_tokens = generated[i, :].tolist()
                decoded = tokenizer.decode(sentence_tokens)

                m = split_re.match(decoded)
                if not m:
                    continue

                title = m.group("title")
                definition = m.group("definition")
                topic = m.group("topic")
                pos = m.group("pos")
                example = m.group("example")

                generated_word = GeneratedWord(
                    word=title and title.strip(),
                    definition=definition and definition.strip(),
                    example=example and example.strip(),
                    pos=pos and pos.strip(),
                    topic=topic and topic.strip(),
                    decoded=decoded,
                    decoded_tokens=sentence_tokens,
                )

                if blacklist and blacklist.contains(title):
                    continue

                if dedupe_titles and title.strip().lower() in seen_titles:
                    continue

                if user_filter and not user_filter(generated_word):
                    continue
                else:
                    ret.append(generated_word)
                    seen_titles.add(generated_word.word.lower())
                    t.update()

        return ret[:num], None

    def _make_examples(self, tokenizer, entry: dictionary_definition.Entry):
        examples = []
        for sense in entry.senses:
            for definition in sense.definitions:
                if isinstance(definition, dictionary_definition.ReferenceDefinition):
                    continue

                token_groups = []
                token_groups.append(
                    TokenGroup(separator=[], payload=tokenizer.encode(definition.definition.rstrip(". ")))
                )

                token_groups.append(TokenGroup(separator=self.definition_sep_ids, payload=tokenizer.encode(entry.word)))

                if sense.pos:
                    if definition.pos_modifier:
                        payload = tokenizer.encode(f"{sense.pos} {definition.pos_modifier}")
                    else:
                        payload = tokenizer.encode(sense.pos)

                    token_groups.append(TokenGroup(separator=self.pos_sep_ids, payload=payload))

                if definition.topic:
                    token_groups.append(
                        TokenGroup(separator=self.topic_sep_ids, payload=tokenizer.encode(definition.topic),)
                    )

                for example in definition.examples:
                    token_groups.append(
                        TokenGroup(
                            separator=self.example_sep_ids, payload=tokenizer.encode(example), remove_if_truncated=True,
                        )
                    )

                example = _join_and_truncate(
                    max_len=self.max_len,
                    begin_tokens=self.bos_token_ids,
                    end_tokens=self.eos_token_ids,
                    token_groups=token_groups,
                )

                assert (
                    len(example) <= self.max_len
                ), f"Example should be less than max length: {len(example)} Vs. {self.max_len}"

                examples.append(example)

        return examples

    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, splits=(1.0), split_idx=0):
        self.max_len = min(tokenizer.max_len_single_sentence, args.block_size)
        self.bos_token_ids = tokenizer.encode(SpecialTokens.BOS_TOKEN)
        self.eos_token_ids = tokenizer.encode(SpecialTokens.EOS_TOKEN)
        self.pos_sep_ids = tokenizer.encode(SpecialTokens.POS_SEP)
        self.definition_sep_ids = tokenizer.encode(SpecialTokens.DEFINITION_SEP)
        self.example_sep_ids = tokenizer.encode(SpecialTokens.EXAMPLE_SEP)
        self.topic_sep_ids = tokenizer.encode(SpecialTokens.TOPIC_SEP)

        assert os.path.isfile(file_path) or os.path.islink(file_path)
        directory, filename = os.path.split(file_path)

        cached_features_file = _cache_path(
            self.__class__.__name__,
            directory,
            filename,
            model_type=args.model_type,
            splits=splits,
            split_idx=split_idx,
            max_len=self.max_len,
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.info("Loaded {len(self.examples)} features")
        else:
            logger.info(
                f"Cache at {cached_features_file} not found... creating features from dataset file at %s", directory,
            )

            self.examples = []
            split_range = _split_range(splits, split_idx)

            with open(file_path, "rb") as f:
                entries = pickle.load(f)

            for entry in entries:
                if _in_split_range(split_range, entry.word):
                    self.examples.extend(self._make_examples(tokenizer, entry))

            logger.info(f"Saving {len(self.examples)} features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class BinaryDictionaryDefinitionDataset(Dataset):
    @classmethod
    def title_tokenization(cls, title):
        return f"<title>{title}</title>"

    @classmethod
    def _make_example(cls, tokenizer, definition):
        max_len = cls.max_len

        m = re.match(r"\s*" + re.escape(definition.title) + r"\d*\s*(\|[^|]*\|)?\s*", definition.entry_str,)
        if m:
            trainable_entry = definition.entry_str[m.span()[1] :].strip()
            if not trainable_entry:
                raise RuntimeError(f"Bad entry for {definition.title}: '{definition.entry_str}'")
        else:
            raise RuntimeError(f"Couldn't match {definition.title} on '{definition.entry_str}'")

        tokenized_title = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(cls.title_tokenization(definition.title))
        )
        tokenized_entry = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(trainable_entry))

        if len(tokenized_title) + len(tokenized_entry) > max_len:
            logger.warn(f"Truncating long entry for '{definition.title}' (entry is {len(tokenized_entry)})")

        all_tokenized = (tokenized_title + tokenized_entry)[:max_len]
        example = tokenizer.build_inputs_with_special_tokens(all_tokenized)
        assert len(example) == len(all_tokenized), "If this fails our tokenizer is weird"

        return example

    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, file_path: str, splits=(1.0), split_idx=0,
    ):
        assert os.path.isfile(file_path) or os.path.islink(file_path)

        self.max_len = min(tokenizer.max_len_single_sentence, args.block_size)

        directory, filename = os.path.split(file_path)
        cached_features_file = _cache_path(
            self.__class__.__name__,
            directory,
            filename,
            model_type=args.model_type,
            splits=splits,
            split_idx=split_idx,
            max_len=self.max_len,
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.info("Loaded {len(self.examples)} features")
        else:
            logger.info("Creating features from dataset file at %s", directory)

            split_range = _split_range(splits, split_idx)
            self.examples = []

            with open(file_path, "rb") as f:
                for dd in dictionary_definition.DictionaryDefinition.gen_from_apple_dictionary(f):
                    if _in_split_range(split_range, dd.title):
                        self.examples.append(self._make_example(tokenizer, dd))

            logger.info(f"Saving {len(self.examples)} features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class UrbanDictionaryDataset(Dataset):
    @classmethod
    def _split_re(cls):
        split_re_pat = (
            f"^{re.escape(SpecialTokens.BOS_TOKEN)}(?P<title>.+?)"
            f"{re.escape(SpecialTokens.DEFINITION_SEP)}(?P<definition>.+?)"
            f"{re.escape(SpecialTokens.EXAMPLE_SEP)}(?P<example>.+?)"
            f"{re.escape(SpecialTokens.EOS_TOKEN)}"
        )
        split_re = re.compile(split_re_pat, flags=re.MULTILINE | re.DOTALL)
        return split_re

    @classmethod
    def generate_words(
        cls,
        tokenizer,
        model,
        prefix=SpecialTokens.BOS_TOKEN,
        num=100,
        max_iterations=10,
        generation_args={},
        blacklist=None,
        example_title_match=True,
        example_match_pos_pipeline=None,
        dedupe_titles=True,
        user_filter=None,
        filter_proper_nouns=False,
        use_custom_generate=True,
        min_definition_words=3,
    ):
        start = time.time()
        viable_candidates = []
        ret = []
        num_iteration = 0
        if isinstance(prefix, str):
            input = tokenizer.encode(prefix, return_tensors="pt").to(model.device)
        else:
            input = torch.tensor([prefix], dtype=torch.long).to(model.device)

        pos_sep_id = _access_zero_assert(tokenizer.encode(SpecialTokens.POS_SEP))
        example_sep_id = _access_zero_assert(tokenizer.encode(SpecialTokens.EXAMPLE_SEP))
        topic_sep_id = _access_zero_assert(tokenizer.encode(SpecialTokens.TOPIC_SEP))
        definition_sep_id = _access_zero_assert(tokenizer.encode(SpecialTokens.DEFINITION_SEP))

        split_re = cls._split_re()
        seen_titles = set()
        stats = GenerationStats()
        t = tqdm(total=num)
        while len(ret) < num and num_iteration < max_iterations:
            num_iteration += 1
            stats.num_iterations += 1
            if not use_custom_generate:
                generated = model.generate(
                    input,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    **generation_args,
                )
            else:

                def partial_generation_transform(input_ids, tokens_to_add):
                    for i in range(tokens_to_add.size()[0]):
                        if blacklist and tokens_to_add[i] in (pos_sep_id, topic_sep_id, definition_sep_id):
                            word = tokenizer.decode(input_ids[i, :][1:])
                            if blacklist.contains(word):
                                tokens_to_add[i] = tokenizer.eos_token_id
                        elif tokens_to_add[i] == tokenizer.eos_token_id:
                            example_token_idxs = input_ids[i, :] == example_sep_id
                            if example_token_idxs.max() == 0:
                                tokens_to_add[i] = example_sep_id

                    return tokens_to_add

                generated = custom_modeling_utils.custom_generate(
                    model,
                    input,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    partial_generation_transform=partial_generation_transform,
                    **generation_args,
                )

            for i in range(generated.size()[0]):
                if len(ret) >= num:
                    break
                viable_candidates = viable_candidates[:1000]

                stats.num_items_considered += 1
                sentence_tokens = generated[i, :].tolist()
                decoded = tokenizer.decode(sentence_tokens)

                m = split_re.match(decoded)
                if not m:
                    stats.num_failed_match += 1
                    continue

                title = m.group("title")
                definition = m.group("definition")
                example = m.group("example")

                generated_word = GeneratedWord(
                    word=title and title.strip(),
                    definition=definition and definition.strip(),
                    example=example and example.strip(),
                    pos=None,
                    topic=None,
                    decoded=decoded,
                    decoded_tokens=sentence_tokens,
                )

                if blacklist and blacklist.contains(title):
                    stats.num_blacklist_filtered += 1
                    continue

                if dedupe_titles and title.strip().lower() in seen_titles:
                    stats.num_seen_filtered += 1
                    continue

                if filter_proper_nouns and title.strip()[:1].isupper():
                    stats.num_proper_noun_filtered += 1
                    continue

                if not example or not example.strip():
                    stats.num_example_missing += 1
                    viable_candidates.append(GeneratedWordCandidate(0.0, generated_word))
                    continue

                if len(definition.split()) < min_definition_words:
                    stats.num_short_definitions += 1
                    viable_candidates.append(GeneratedWordCandidate(0.2, generated_word))
                    continue

                t_rstrip = title.strip().lower().rstrip("s")
                l_example = example.lower()
                try:
                    l_example.index(t_rstrip)
                except ValueError:
                    stats.num_example_missing_title += 1
                    viable_candidates.append(GeneratedWordCandidate(0.5, generated_word))
                    continue

                if user_filter and not user_filter(generated_word):
                    stats.num_user_filtered += 1
                    continue
                else:
                    t.update()
                    ret.append(generated_word)
                    seen_titles.add(generated_word.word.lower())

        stats.num_returned = len(ret)
        stats.viable_candidates = viable_candidates
        stats.wall_time = time.time() - start
        return ret[:num], stats

    def _make_examples(self, tokenizer, word):
        examples = []

        for definition in word.definitions:
            example = _join_and_truncate(
                max_len=self.max_len,
                begin_tokens=self.bos_token_ids,
                end_tokens=self.eos_token_ids,
                token_groups=[
                    TokenGroup(separator=[], payload=tokenizer.encode(definition.word)),
                    TokenGroup(separator=self.definition_sep_ids, payload=tokenizer.encode(definition.meaning),),
                    TokenGroup(separator=self.example_sep_ids, payload=tokenizer.encode(definition.examples[0]),),
                ],
            )

            assert (
                len(example) <= self.max_len
            ), f"Example should be less than max length: {len(example)} Vs. {self.max_len}"

            examples.append(example)

        return examples

    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, file_path: str, splits=(1.0), split_idx=0,
    ):
        self.max_len = min(tokenizer.max_len_single_sentence, args.block_size)
        self.bos_token_ids = tokenizer.encode(SpecialTokens.BOS_TOKEN)
        self.eos_token_ids = tokenizer.encode(SpecialTokens.EOS_TOKEN)
        self.definition_sep_ids = tokenizer.encode(SpecialTokens.DEFINITION_SEP)
        self.example_sep_ids = tokenizer.encode(SpecialTokens.EXAMPLE_SEP)

        assert os.path.isfile(file_path) or os.path.islink(file_path)
        directory, filename = os.path.split(file_path)

        cached_features_file = _cache_path(
            self.__class__.__name__,
            directory,
            filename,
            model_type=args.model_type,
            splits=splits,
            split_idx=split_idx,
            max_len=self.max_len,
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.info("Loaded {len(self.examples)} features")
        else:
            logger.info(
                f"Cache at {cached_features_file} not found... creating features from dataset file at %s", directory,
            )

            self.examples = []
            split_range = _split_range(splits, split_idx)

            with open(file_path, "rb") as f:
                words = list(pickle.load(f).values())

            for word in words:
                if _in_split_range(split_range, word.title):
                    self.examples.extend(self._make_examples(tokenizer, word))

            logger.info(f"Saving {len(self.examples)} features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


@dataclass
class WikiArticle:
    title: str
    text: str


class WikiArticleTitleDataset(Dataset):
    @classmethod
    def title_tokenization(cls, title):
        return f"<bot>{title}<eot>"

    @classmethod
    def refine_wikitext(cls, istream, limit=None):
        last_blank = False
        title_matcher = re.compile(r"^[\s]*= ([^=]*) =[\s]*$")
        last_title = None

        article_text = StringIO()
        for i, line in enumerate(istream):
            m = title_matcher.match(line)
            if m and last_blank:
                title = m.group(1)

                if last_title is not None:
                    yield WikiArticle(title=last_title, text=article_text.getvalue())
                last_title = title
                article_text = StringIO()
            else:
                cleaned_line = re.sub(re.escape(last_title), "TITLE", line, flags=re.IGNORECASE) if last_title else line
                article_text.write(cleaned_line)

            last_blank = re.match(r"^\s*$", line)

            if limit and i > limit:
                break

        yield WikiArticle(title=last_title, text=article_text.getvalue())

    @classmethod
    def generate_text_dataset(cls, istream, ostream, offset=0, stride=1024, limit=None):
        def _output_range(article, start, end):
            text = article.text[start:end]
            spaces = list(re.compile(r"\s+").finditer(text))
            if spaces:
                replace_idx = spaces[-1].span()[0]
                ostream.write(text[:replace_idx])
                ostream.write(cls.title_tokenization(article.title))
                ostream.write(text[replace_idx:])
            else:
                ostream.write(text)
                ostream.write(cls.title_tokenization(article.title))

        for article in cls.refine_wikitext(istream, limit=limit):
            if offset > 0:
                _output_range(article, 0, offset)

            for i in range(offset, len(article.text), stride):
                _output_range(article, i, i + stride)

    @staticmethod
    def _make_example(tokenizer, text_tokens, title_tokens):
        example = tokenizer.build_inputs_with_special_tokens(text_tokens + title_tokens)
        start_title_idx = next(i for i in reversed(range(len(example))) if example[i] == title_tokens[0])
        end_title_idx = start_title_idx + len(title_tokens)
        bool_mask = [bool(i > start_title_idx and i < end_title_idx) for i in range(len(example))]

        return (example, bool_mask)

    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename,
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []

            with open(file_path, encoding="utf-8") as f:
                for article in self.refine_wikitext(f):
                    tokenized_title = tokenizer.convert_tokens_to_ids(
                        tokenizer.tokenize(self.title_tokenization(article.title))
                    )
                    tokenized_article_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(article.text))

                    article_block_size = block_size - len(tokenized_title)
                    for i in range(0, len(tokenized_article_text) - article_block_size + 1, article_block_size,):
                        self.examples.append(
                            self._make_example(
                                tokenizer, tokenized_article_text[i : (i + article_block_size)], tokenized_title,
                            )
                        )

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return (
            torch.tensor(self.examples[item][0], dtype=torch.long),
            torch.tensor(self.examples[item][1], dtype=torch.bool),
        )
