# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, Facebook AI Research authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import torch
from torch.nn import functional as F
from transformers.modeling_utils import top_k_top_p_filtering, calc_banned_ngram_tokens


@torch.no_grad()
def custom_generate(
    self,
    input_ids=None,
    max_length=None,
    min_length=None,
    do_sample=None,
    early_stopping=None,
    num_beams=None,
    temperature=None,
    top_k=None,
    top_p=None,
    repetition_penalty=None,
    bad_words_ids=None,
    bos_token_id=None,
    pad_token_id=None,
    eos_token_id=None,
    length_penalty=None,
    no_repeat_ngram_size=None,
    num_return_sequences=None,
    attention_mask=None,
    decoder_start_token_id=None,
    use_cache=True,
    partial_generation_transform=None,
):
    r""" Generates sequences for models with a LM head. The method currently supports greedy decoding, beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

    Adapted in part from `Facebook's XLM beam search code`_.

    .. _`Facebook's XLM beam search code`:
        https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


    Parameters:

        input_ids: (`optional`) `torch.LongTensor` of shape `(batch_size, sequence_length)`
            The sequence used as a prompt for the generation. If `None` the method initializes
            it as an empty `torch.LongTensor` of shape `(1,)`.

        max_length: (`optional`) int
            The max length of the sequence to be generated.  Between `min_length` and infinity. Default to 20.

        min_length: (`optional`) int
            The min length of the sequence to be generated.  Between 0 and infinity. Default to 0.

        do_sample: (`optional`) bool
            If set to `False` greedy decoding is used. Otherwise sampling is used. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

        early_stopping: (`optional`) bool
            if set to `True` beam search is stopped when at least `num_beams` sentences finished per batch. Defaults to `False` as defined in `configuration_utils.PretrainedConfig`.

        num_beams: (`optional`) int
            Number of beams for beam search. Must be between 1 and infinity. 1 means no beam search. Default to 1.

        temperature: (`optional`) float
            The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.

        top_k: (`optional`) int
            The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity. Default to 50.

        top_p: (`optional`) float
            The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. Must be between 0 and 1. Default to 1.

        repetition_penalty: (`optional`) float
            The parameter for repetition penalty. Between 1.0 and infinity. 1.0 means no penalty. Default to 1.0.

        pad_token_id: (`optional`) intcalc_banned_ngram_tokens

        bos_token_id: (`optional`) int
            BOS token. Defaults to `bos_token_id` as defined in the models config.

        eos_token_id: (`optional`) int
            EOS token. Defaults to `eos_token_id` as defined in the models config.

        length_penalty: (`optional`) float
            Exponential penalty to the length. Default to 1.

        no_repeat_ngram_size: (`optional`) int
            If set to int > 0, all ngrams of size `no_repeat_ngram_size` can only occur once.
        bad_words_ids: (`optional`) list of lists of int
            `bad_words_ids` contains tokens that are not allowed to be generated. In order to get the tokens of the words that should not appear in the generated text, use `tokenizer.encode(bad_word, add_prefix_space=True)`.

        num_return_sequences: (`optional`) int
            The number of independently computed returned sequences for each element in the batch. Default to 1.

        attention_mask (`optional`) obj: `torch.LongTensor` of same shape as `input_ids`
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
            Defaults to `None`.

        `What are attention masks? <../glossary.html#attention-mask>`__

        decoder_start_token_id=None: (`optional`) int
            If an encoder-decoder model starts decoding with a different token than BOS.
            Defaults to `None` and is changed to `BOS` later.

        use_cache: (`optional`) bool
            If `use_cache` is True, past key values are used to speed up decoding if applicable to model. Defaults to `True`.

    Return:

        output: `torch.LongTensor` of shape `(batch_size * num_return_sequences, sequence_length)`
            sequence_length is either equal to max_length or shorter if all batches finished early due to the `eos_token_id`

    Examples::

        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
        outputs = model.generate(max_length=40)  # do greedy decoding
        print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
        input_context = 'The dog'
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
        for i in range(3): #  3 output sequences were generated
            print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
        input_context = 'The dog'
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3)  # 3 generate sequences using by sampling
        for i in range(3): #  3 output sequences were generated
            print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
        input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
        print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

        tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
        input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
        bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
    """

    # We cannot generate if the model does not have a LM head
    if self.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
        )

    max_length = max_length if max_length is not None else self.config.max_length
    min_length = min_length if min_length is not None else self.config.min_length
    do_sample = do_sample if do_sample is not None else self.config.do_sample
    early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    num_beams = num_beams if num_beams is not None else self.config.num_beams
    temperature = temperature if temperature is not None else self.config.temperature
    top_k = top_k if top_k is not None else self.config.top_k
    top_p = top_p if top_p is not None else self.config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else self.config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
    length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
    )
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
    assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
        isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
        isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
        isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
        isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
    ), "`no_repeat_ngram_size` should be a positive integer."
    assert (
        isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."
    assert (
        bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
    ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device,)
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    # not allow to duplicate outputs when greedy decoding
    if do_sample is False:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    # create attention mask if necessary
    # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after
    # attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning("Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id))
        pad_token_id = eos_token_id

    # current position and vocab size
    if hasattr(self.config, "vocab_size"):
        vocab_size = self.config.vocab_size
    elif (
        self.config.is_encoder_decoder
        and hasattr(self.config, "decoder")
        and hasattr(self.config.decoder, "vocab_size")
    ):
        vocab_size = self.config.decoder.vocab_size

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if self.config.is_encoder_decoder:
        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

        assert (
            decoder_start_token_id is not None
        ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
        assert hasattr(self, "get_encoder"), "{} should have a 'get_encoder' function defined".format(self)
        assert callable(self.get_encoder), "{} should be a method".format(self.get_encoder)

        # get encoder and store encoder outputs
        encoder = self.get_encoder()

        encoder_outputs: tuple = encoder(input_ids, attention_mask=attention_mask)

    # Expand input ids if num_beams > 1 or num_return_sequences > 1
    if num_return_sequences > 1 or num_beams > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)

        input_ids = input_ids.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    if self.config.is_encoder_decoder:
        # create empty decoder_input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        cur_len = 1

        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (encoder_outputs[0].index_select(0, expanded_batch_idxs), *encoder_outputs[1:])

    else:
        encoder_outputs = None
        cur_len = input_ids.shape[-1]

    if num_beams > 1:
        raise RuntimeError("DIDN'T MODIFY BEAM SEARCH!")
    else:
        output = _generate_no_beam_search(
            self,
            input_ids,
            cur_len=cur_len,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            use_cache=use_cache,
            partial_generation_transform=partial_generation_transform,
        )

    return output


def _generate_no_beam_search(
    self,
    input_ids,
    cur_len,
    max_length,
    min_length,
    do_sample,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    bad_words_ids,
    bos_token_id,
    pad_token_id,
    eos_token_id,
    decoder_start_token_id,
    batch_size,
    encoder_outputs,
    attention_mask,
    use_cache,
    partial_generation_transform,
):
    """ Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
    """
    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    past = encoder_outputs  # defined for encoder-decoder models, None for decoder-only models

    goodies = []
    while cur_len < max_length:
        model_inputs = self.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache
        )

        outputs = self(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]

        # if model has past, then set the past variable to speed up decoding
        if use_cache:  # and self._use_cache(outputs, use_cache): !! removed:tdimson
            past = outputs[1]

        # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(next_token_logits, batch_size, 1, input_ids, repetition_penalty)

        if no_repeat_ngram_size > 0:
            # calculate a list of banned tokens to prevent repetitively generating the same ngrams
            # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
            banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
            for batch_idx in range(batch_size):
                next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

        if bad_words_ids is not None:
            # calculate a list of banned tokens according to bad words
            banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

            for batch_idx in range(batch_size):
                next_token_logits[batch_idx, banned_tokens[batch_idx]] = -float("inf")

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            next_token_logits[:, eos_token_id] = -float("inf")

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            # Top-p/top-k filtering
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # tdimson: add a partial generation transform that can manipulate as we generate
        if partial_generation_transform is not None:
            tokens_to_add = partial_generation_transform(input_ids, tokens_to_add)

        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len + 1)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

            # tdimson: stop computing for sentences that have terminated
            for i in range(input_ids.size()[0]):
                if unfinished_sents[i] == 0:
                    goodies.append(input_ids[i, :])

            idx = unfinished_sents == 1
            input_ids = input_ids[idx, :]
            attention_mask = attention_mask[idx, :]
            sent_lengths = sent_lengths[idx]
            new_past = []
            for item in past:
                new_past.append(item[:, idx, :, :, :])
            past = tuple(new_past)
            unfinished_sents = unfinished_sents[idx]

            if input_ids.size()[0] == 0:
                break

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if self.config.is_encoder_decoder is False:
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        cur_len = cur_len + 1

    # tdimson: take our finished sentences and roll
    for i in range(input_ids.size()[0]):
        goodies.append(input_ids[i])

    from torch.nn.utils.rnn import pad_sequence

    return pad_sequence(goodies, batch_first=True, padding_value=pad_token_id)

    # if there are different sentences lengths in the batch, some batches have to be padded
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`Pad_token_id` has to be defined if batches have different lengths"
        # finished sents are filled with pad_token
        decoded = input_ids.new(batch_size, sent_lengths.max().item()).fill_(pad_token_id)
    else:
        decoded = input_ids

    for hypo_idx, hypo in enumerate(input_ids):
        decoded[hypo_idx, : sent_lengths[hypo_idx]] = hypo[: sent_lengths[hypo_idx]]

    return decoded


def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens
