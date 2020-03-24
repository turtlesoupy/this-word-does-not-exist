import types
import os
import logging
import random
import pickle
import hashlib
import string
import itertools
import dictionary_definition
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
