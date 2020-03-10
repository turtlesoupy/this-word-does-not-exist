from struct import unpack
from zlib import decompress
import sys
import re
import hashlib
from bs4 import BeautifulSoup
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import logging
import os
import torch

logger = logging.getLogger(__name__)


@dataclass
class DictionaryDefinition:
    title: str
    entry_str: str
    parsed_entry: BeautifulSoup

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
                title_soup = BeautifulSoup(title, features="html.parser")
                entry_soup = BeautifulSoup(entry, features="html.parser")

                title = title_soup.get_text()
                entry = entry_soup.get_text()

                if not title or not entry:
                    logger.warning(f"Invalid entry {title}: {entry}")
                    continue

                yield cls(
                    title=title_soup.get_text(),
                    entry_str=entry_soup.get_text(),
                    parsed_entry=entry_soup,
                )


class DictionaryDefinitionDataset(Dataset):
    @classmethod
    def title_tokenization(cls, title):
        return f"<title>{title}</title>"

    @classmethod
    def _make_example(cls, tokenizer, definition):
        max_len = tokenizer.max_len_single_sentence

        m = re.match(
            r"\s*" + re.escape(definition.title) + r"\d*\s*(\|[^|]*\|)?\s*",
            definition.entry_str,
        )
        if m:
            trainable_entry = definition.entry_str[m.span()[1] :].strip()
            if not trainable_entry:
                raise RuntimeError(
                    f"Bad entry for {definition.title}: '{definition.entry_str}'"
                )
        else:
            raise RuntimeError(
                f"Couldn't match {definition.title} on '{definition.entry_str}'"
            )

        tokenized_title = [tokenizer.bos_token_id] + tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(cls.title_tokenization(definition.title))
        )
        tokenized_entry = tokenizer.convert_tokens_to_ids(
            tokenizer.tokenize(trainable_entry)
        )

        if len(tokenized_title) + len(tokenized_entry) > max_len:
            logger.warn(
                f"Truncating long entry for '{definition.title}' (entry is {len(tokenized_entry)})"
            )

        all_tokenized = (tokenized_title + tokenized_entry)[:max_len]
        example = tokenizer.build_inputs_with_special_tokens(all_tokenized)
        assert len(example) == len(
            all_tokenized
        ), "If this fails our tokenizer is weird"
        bool_mask = [
            bool(i > 1 and i <= len(tokenized_title)) for i in range(len(example))
        ]

        return (example, bool_mask)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        args,
        file_path: str,
        splits=(1.0),
        split_idx=0,
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
            raise RuntimeError(
                f"Invalid split index {split_idx} (must be less than {len(sum_splits)})"
            )

        if split_idx == 0:
            start_range = 0.0
        else:
            start_range = sum_splits[split_idx - 1]

        end_range = sum_splits[split_idx]

        def in_split(dictionary_definiton):
            val = (
                int(
                    hashlib.md5(
                        dictionary_definition.title.encode("utf-8")
                    ).hexdigest(),
                    16,
                )
                % 10000
                / 10000
            )
            return (val >= start_range and val < end_range).item()

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
            logger.info("Loaded {len(self.examples)} features")
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []

            with open(file_path, "rb") as f:
                for (
                    dictionary_definition
                ) in DictionaryDefinition.gen_from_apple_dictionary(f):
                    if in_split(dictionary_definition):
                        self.examples.append(
                            self._make_example(tokenizer, dictionary_definition)
                        )

            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info(
                f"Saving {len(self.examples)} features into cached file {cached_features_file}"
            )
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
