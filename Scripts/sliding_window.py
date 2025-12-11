
#############################
# IMPORTS
#############################

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
import spacy
import torch
from tqdm import tqdm
import pandas as pd
import math
import collections
import os
import xml.etree.ElementTree as ET
import gc
from spacy.tokens import Doc
import time
from itertools import batched

WINDOW = 4
STEP = 1
BATCH = 16



def get_sliding_window(data_list, batch_num):
    
    if len(data_list) <= WINDOW:
        yield data_list
    else: 
        for start_pos in range((batch_num*WINDOW), (len(data_list)-WINDOW+STEP), STEP):
            batch_start = min(len(data_list) - 1, start_pos)
            batch_end = min(len(data_list), start_pos + WINDOW)
            yield data_list[batch_start:batch_end]




def surprisal_calc_sliding_window(sentence_tuples, tokenizer, model, accelerator, batch_num):
    sentences = [sent[0] for sent in sentence_tuples]

    all_token_surprisals = []

    window_gen = get_sliding_window(sentences, batch_num)

    for i, window in enumerate(batched(window_gen, n=BATCH)):
        joined_sentences = [tokenizer.bos_token + " ".join(sentences) for sentences in window]

        context_only = [tokenizer.bos_token + " ".join(sentences[:-1])] # To track idx of final word
        context_encoding = tokenizer(context_only, padding = False)
        context_lengths = [len(x) for x in context_encoding['input_ids']]


        inputs = tokenizer(joined_sentences, return_tensors='pt', padding=True).to(accelerator.device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        shift_logits = log_probs[:,:-1,:]
        shift_ids = inputs['input_ids'][:, 1:]

        surprisals = -torch.gather(shift_logits, 2, shift_ids.unsqueeze(-1)).squeeze(-1).cpu()

        input_ids = inputs['input_ids'].cpu()

        for i, start_idx in enumerate(context_lengths):
            slice_start = start_idx - 1

            seq_len = shift_ids[i].ne(tokenizer.pad_token_id).sum().item()

            target_surprisals = surprisals[i, slice_start : seq_len]
            target_tokens = input_ids[i, start_idx:ln(target_surprisals)]

            for t_id, val in zip(target_tokens, target_surprisals):
                all_token_surprisals.append({
                    "token_id" : t_id.item(),
                    "token": tokenizer.decode([t_id.item()]),
                    'surprisal' : val.item()
                })
        del inputs, outputs, logits, log_probs, shift_logits, shift_ids

        if i % 10 == 0:
            gc.collect()
            empty_gpu_cache()

    return all_token_surprisals