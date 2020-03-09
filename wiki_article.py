import torch
import re
import sys
from dataclasses import dataclass
from io import StringIO
from transformers import AutoModelWithLMHead, AutoTokenizer
from scipy import stats
from torch.nn.utils.rnn import pad_sequence
from typing import List
from torch.utils.data import SequentialSampler, DataLoader

@dataclass
class Article:
    title: str
    text: str
        
def title_tokenization(title):
    return f"<bot>{title}<eot>"
    
def refine_wikitext(istream, limit=None):
    last_blank = False
    title_matcher = re.compile("^[\s]*= ([^=]*) =[\s]*$")
    last_title = None
    
    
    article_text = StringIO()
    for i, line in enumerate(istream):
        m = title_matcher.match(line)
        if m and last_blank:
            title = m.group(1)
            
            if last_title is not None:
                yield Article(
                    title=last_title,
                    text=article_text.getvalue()
                )
            last_title = title
            article_text = StringIO()
        else:
            cleaned_line = re.sub(re.escape(last_title), "TITLE", line, flags=re.IGNORECASE) if last_title else line
            article_text.write(cleaned_line)
            
        last_blank = re.match("^\s*$", line)
        
        if limit and i > limit:
            break
    
    yield Article(
        title=last_title,
        text=article_text.getvalue()
    )
    
def generate_text_dataset(istream, ostream, offset=0, stride=1024, limit=None):
    def _output_range(article, start, end):
        text = article.text[start:end]
        spaces = list(re.compile("\s+").finditer(text))
        if spaces:
            replace_idx = spaces[-1].span()[0]
            ostream.write(text[:replace_idx])
            ostream.write(title_tokenization(article.title))
            ostream.write(text[replace_idx:])
        else:
            ostream.write(text)
            ostream.write(title_tokenization(article.title))       
    extract_dictionary
    for article in refine_wikitext(istream, limit=limit):
        if offset > 0:
            _output_range(article, 0, offset)
        
        for i in range(offset, len(article.text), stride):
            _output_range(article, i, i + stride)       
        
def title_perplexity(model, tokenizer, article, device='cuda'):
    max_length = model.config.n_positions
    article_tokens = tokenizer.tokenize(article.text)
    title_tokens = tokenizer.tokenize(title_tokenization(article.title))
    
    tokens = article_tokens[:(max_length - len(title_tokens) - 1)] + title_tokens
    token_ids = [tokenizer.eos_token_id] + tokenizer.convert_tokens_to_ids(tokens)
    
    
    with torch.no_grad():
        tensor_input = torch.tensor([token_ids], device=device)
        loss, logits, *_  = model(tensor_input, labels=tensor_input)
        
        # TODO: probably should just make this count actual title tokensstats
        title_offset = len(tokens) - len(title_tokens) 
        lp = 0
        n = 0
        for i, input in enumerate(tensor_input[0][title_offset:]):
            predicted_score = logits[0, i]
            predicted_prob = torch.nn.functional.softmax(predicted_score, dim=0)
            lp += torch.log(predicted_prob[input])
            n += 1
        
        title_pp = -lp /  n
    
    return title_pp.item()


def lm_eval(model, tokenizer, file_path, device='cuda', block_size=512, batch_size=1):
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)
    eval_dataset = []
    with open(file_path, encoding="utf-8") as f:
        text = f.read()
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            tokenized =  tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
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

        
        
        


def perplexity(sentence):
    with torch.no_grad():
        tokenize_input = tokenizer.tokenize(sentence)
        tensor_input = torch.tensor([
            [tokenizer.eos_token_id] + 
            tokenizer.convert_tokens_to_ids(tokenize_input)
        ])
        loss, logits, *_  = model(tensor_input, labels=tensor_input)
        
        

    lp = 0
    n = 0
    for i, input in enumerate(tensor_input[0][1:]):
        masked_index = i
        predicted_score = logits[0, masked_index]
        predicted_prob = torch.nn.functional.softmax(predicted_score, dim=0)
        lp += torch.log(predicted_prob[input])
        n += 1

    
    print(loss)
    print(- lp /  n)

    return -loss

def run_title_evaluation(model, tokenizer, path, limit=None):
    title_pp = []
    with open(path) as f:
        for article in refine_wikitext(f, limit=limit):
            title_pp.append(title_perplexity(model, tokenizer, article))
    return stats.describe(title_pp)
