from dataclasses import dataclass
import types
import os
import logging
import random
import pickle
import hashlib
import string
import itertools
import dictionary_definition
import re
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import NamedTuple, List, Optional

logger = logging.getLogger(__name__)


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


def _cache_path(base_directory, filename, **keys):
    path = []
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
    max_len: int, begin_tokens: List[int], token_groups: List[TokenGroup], end_tokens: List[int], min_append_size=5
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


class ParsedDictionaryDefinitionDataset(Dataset):
    @dataclass
    class GeneratedWord:
        word: str
        pos: Optional[str]
        topic: Optional[str]
        definition: str
        example: Optional[str]
        from_example_expansion: bool = False

    @classmethod
    def in_blacklist(cls, word, blacklist):
        word = word.strip().lower()
        return (
            word in blacklist
            or word.rstrip("s") in blacklist
            or word.rstrip("ing") in blacklist
            or all(e in blacklist for e in word.split())
        )

    @classmethod
    def generate_words(
        cls,
        tokenizer,
        model,
        prefix=SpecialTokens.BOS_TOKEN,
        num=100,
        max_iterations=10,
        batch_size=50,
        max_length=512,
        top_k=50,
        blacklist=(),
        do_example_expansion=False,
    ):
        ret = []
        num_iteration = 0
        if isinstance(prefix, str):
            input = tokenizer.encode(prefix, return_tensors="pt").to("cuda")
        else:
            input = torch.tensor([prefix], dtype=torch.long).to("cuda")

        eos_token_ids = tokenizer.encode(SpecialTokens.EOS_TOKEN)
        example_sep_ids = tokenizer.encode(SpecialTokens.EXAMPLE_SEP)
        split_re_pat = (
            f"^{re.escape(SpecialTokens.BOS_TOKEN)}(?P<title>.+?)"
            f"(?:{re.escape(SpecialTokens.POS_SEP)}(?P<pos>.+?))?"
            f"(?:{re.escape(SpecialTokens.TOPIC_SEP)}(?P<topic>.+?))?"
            f"{re.escape(SpecialTokens.DEFINITION_SEP)}(?P<definition>.+?)"
            f"(?:{re.escape(SpecialTokens.EXAMPLE_SEP)}(?P<example>.+?))*"
            f"{re.escape(SpecialTokens.EOS_TOKEN)}"
        )
        split_re = re.compile(split_re_pat, flags=re.MULTILINE | re.DOTALL)

        seen_titles = set()
        while len(ret) < num and num_iteration < max_iterations:
            num_iteration += 1
            generated = model.generate(
                input,
                max_length=max_length,
                num_return_sequences=batch_size,
                top_k=top_k,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_ids=tokenizer.eos_token_id,
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

                if cls.in_blacklist(title, blacklist) or title.strip().lower() in seen_titles:
                    continue

                if not example or not example.strip() or title.strip().lower().rstrip("s") not in example.lower():
                    if do_example_expansion:
                        eos_loc = max(
                            i
                            for i in range(len(sentence_tokens))
                            if sentence_tokens[i : (i + len(eos_token_ids))] == eos_token_ids
                        )
                        example_prefix = sentence_tokens[:eos_loc]
                        example_prefix.extend(example_sep_ids)

                        more_words = cls.generate_words(
                            tokenizer,
                            model,
                            num=1,
                            prefix=example_prefix,
                            max_iterations=1,
                            max_length=max_length,
                            top_k=top_k,
                            blacklist=blacklist,
                            do_example_expansion=False,
                        )
                        if more_words:
                            more_words[0].from_example_expansion = True
                            ret.append(more_words[0])
                            seen_titles.add(title.strip().lower())
                    else:
                        continue
                else:
                    ret.append(
                        cls.GeneratedWord(
                            word=title and title.strip(),
                            definition=definition and definition.strip(),
                            example=example and example.strip(),
                            pos=pos and pos.strip(),
                            topic=topic and topic.strip(),
                        )
                    )
                    seen_titles.add(title.strip().lower())

        return ret[:num]

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
                        TokenGroup(separator=self.topic_sep_ids, payload=tokenizer.encode(definition.topic))
                    )

                token_groups.append(
                    TokenGroup(separator=self.definition_sep_ids, payload=tokenizer.encode(definition.definition))
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

    def __init__(
        self, tokenizer: PreTrainedTokenizer, args, file_path: str, splits=(1.0), split_idx=0,
    ):
        self.max_len = min(tokenizer.max_len_single_sentence, args.block_size)
        self.bos_token_ids = tokenizer.encode(SpecialTokens.BOS_TOKEN)
        self.eos_token_ids = tokenizer.encode(SpecialTokens.EOS_TOKEN)
        self.pos_sep_ids = tokenizer.encode(SpecialTokens.POS_SEP)
        self.definition_sep_ids = tokenizer.encode(SpecialTokens.DEFINITION_SEP)
        self.example_sep_ids = tokenizer.encode(SpecialTokens.EXAMPLE_SEP)
        self.topic_sep_ids = tokenizer.encode(SpecialTokens.TOPIC_SEP)
        return

        assert os.path.isfile(file_path) or os.path.islink(file_path)
        directory, filename = os.path.split(file_path)

        cached_features_file = _cache_path(
            directory, filename, model_type=args.model_type, splits=splits, split_idx=split_idx, max_len=self.max_len,
        )

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
        max_len = self.max_len

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
            directory, filename, model_type=args.model_type, splits=splits, split_idx=split_idx, max_len=self.max_len,
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
                for dictionary_definition in DictionaryDefinition.gen_from_apple_dictionary(f):
                    if _in_split_range(split_range, dictionary_definition.title):
                        self.examples.append(self._make_example(tokenizer, dictionary_definition))

            logger.info(f"Saving {len(self.examples)} features into cached file {cached_features_file}")
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class UrbanDictionaryDataset(Dataset):
    def _make_examples(self, tokenizer, word):
        examples = []

        for definition in word.definitions:
            example = _join_and_truncate(
                max_len=self.max_len,
                begin_tokens=self.bos_token_ids,
                end_tokens=self.eos_token_ids,
                token_groups=[
                    TokenGroup(separator=[], payload=tokenizer.encode(definition.word)),
                    TokenGroup(separator=self.definition_sep_ids, payload=tokenizer.encode(definition.meaning)),
                    TokenGroup(separator=self.example_sep_ids, payload=tokenizer.encode(definition.examples[0])),
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
            directory, filename, model_type=args.model_type, splits=splits, split_idx=split_idx, max_len=self.max_len,
        )

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
        title_matcher = re.compile("^[\s]*= ([^=]*) =[\s]*$")
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

            last_blank = re.match("^\s*$", line)

            if limit and i > limit:
                break

        yield WikiArticle(title=last_title, text=article_text.getvalue())

    @classmethod
    def generate_text_dataset(cls, istream, ostream, offset=0, stride=1024, limit=None):
        def _output_range(article, start, end):
            text = article.text[start:end]
            spaces = list(re.compile("\s+").finditer(text))
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
                                tokenizer, tokenized_article_text[i : i + article_block_size], tokenized_title,
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
