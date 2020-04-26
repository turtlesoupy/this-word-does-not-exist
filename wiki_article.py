import os
import torch
import re
import sys
import logging
import pickle
from dataclasses import dataclass
from io import StringIO
from transformers import AutoModelWithLMHead, AutoTokenizer, PreTrainedTokenizer
from scipy import stats
from torch.nn.utils.rnn import pad_sequence
from typing import List
from torch.utils.data import SequentialSampler, DataLoader, Dataset

logger = logging.getLogger(__name__)


def title_perplexity(model, tokenizer, article, device="cuda"):
    max_length = model.config.n_positions
    article_tokens = tokenizer.tokenize(article.text)
    title_tokens = tokenizer.tokenize(title_tokenization(article.title))

    tokens = article_tokens[: (max_length - len(title_tokens) - 1)] + title_tokens
    token_ids = [tokenizer.eos_token_id] + tokenizer.convert_tokens_to_ids(tokens)

    with torch.no_grad():
        tensor_input = torch.tensor([token_ids], device=device)
        loss, logits, *_ = model(tensor_input, labels=tensor_input)

        # TODO: probably should just make this count actual title tokensstats
        title_offset = len(tokens) - len(title_tokens)
        lp = 0
        n = 0
        for i, input in enumerate(tensor_input[0][title_offset:]):
            predicted_score = logits[0, i]
            predicted_prob = torch.nn.functional.softmax(predicted_score, dim=0)
            lp += torch.log(predicted_prob[input])
            n += 1

        title_pp = -lp / n

    return title_pp.item()


def lm_eval(model, tokenizer, file_path, device="cuda", block_size=512, batch_size=1):
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(
            examples, batch_first=True, padding_value=tokenizer.pad_token_id
        )

    block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)
    eval_dataset = []
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        for i in range(
            0, len(tokenized_text) - block_size + 1, block_size
        ):  # Truncate in block of block_size
            tokenized = tokenizer.build_inputs_with_special_tokens(
                tokenized_text[i : i + block_size]
            )
            tensorized = torch.tensor(tokenized, dtype=torch.long)
            eval_dataset.append(tensorized)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=batch_size, collate_fn=collate
    )

    for batch in eval_dataloader:
        inputs, labels = (batch, batch)
        inputs = inputs.to(device)
        labels = labels.to(device)

    eval_loss = 0.0
    with torch.no_grad():
        outputs = model(inputs, labels=labels)
        lm_loss = outputs[0]
        eval_loss += lm_loss.mean().item()

    perplexity = torch.exp(torch.tensor(eval_loss))

    return perplexity


def perplexity(model, tokenizer, sentences, device="cuda", **fwd_args):
    with torch.no_grad():
        token_ids = [
            torch.tensor(
                [tokenizer.eos_token_id]
                + tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence))
            )
            for sentence in sentences
        ]
        padded_tokens = pad_sequence(token_ids, batch_first=True)
        tensor_input = padded_tokens.to(device)
        loss, logits, *_ = model(tensor_input, labels=tensor_input, **fwd_args)

    lp = 0
    n = 0
    for i, input in enumerate(tensor_input[0][1:]):
        masked_index = i
        predicted_score = logits[0, masked_index]
        predicted_prob = torch.nn.functional.softmax(predicted_score, dim=0)
        lp += torch.log(predicted_prob[input])
        n += 1

    return -loss


def run_title_evaluation(model, tokenizer, path, limit=None):
    title_pp = []
    with open(path) as f:
        for article in refine_wikitext(f, limit=limit):
            title_pp.append(title_perplexity(model, tokenizer, article))
    return stats.describe(title_pp)
