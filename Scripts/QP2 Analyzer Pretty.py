#########################################
"""
VINCENT MARIANI
11 DECEMBER 2025

Analysis script for QP 2; 
Takes an XML corpus (in this case the BNC1994) and outputs CSV files with surprisal calculations and NLP characteristics.
"""

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

############################
# CONFIG
############################

"""
This block configures the script parameters: 

    - BATCH: The number of sentences to manage at once for surprisal. Context builds from first token to last within the batch. (NOTE: No file in the BNC contains >600 sentences, setting to a higher value will process the entire file at once).

    - CONTEXT: The number of preceding sentences to use as context at the beginning of a non-initial batch. The first sentence in a non-initial batch will begin with this many previous sentences as context, and the context will build from the first to last token of the batch. 

    - TOKEN_LIM: Limit of tokens to simultaneously calculate surprisal for. This will not effect context; each token will receive the normal amount of context, the calculation will just proceed consecutively rather than concurrently.

    - SPACY_BATCH: The number of sentences to handle at once for SpaCy. Affects performance but not data. 

    - OVERWRITE: Whether to overwrite existing files in the given OUTPUT_DIR or to keep and skip existing output files. Setting to 0 essentially resumes from where the calculation left off. 

    - INPUT_DIR: The directory of the input XML files, in this case the BNC1994 corpus. May be a hierarchical nested file tree. 

    - OUTPUT_DIR: The directory of the output CSV files. Note that each input XML will have its own output CSV. They must be merged later, usually in R. 

    - SPACY_MOD: The SpaCy NLP model to be used. "en_core_web_trf" is recommended.

    - TRANSFORMER_MOD: The transformer LLM model to be used. Currently programmed to use HuggingFace. "meta-llama/Llama-3.2-1B" is recommended, but a larger or smaller model may be used depending on computing resources. 
"""
BATCH = 1024 
CONTEXT = 32 
TOKEN_LIM = 16384 

SPACY_BATCH = 256

OVERWRITE = 1 

INPUT_DIR = "D:/BNC Full Data/BNCFiles/Full BNC1994/download/Texts" # The directory of the input XML files
OUTPUT_DIR = "D:/BNC Full Data/12-11_11AM Run/CSV" # The directory of the output CSV files

SPACY_MOD = "en_core_web_trf" # The SpaCy model to use
TRANSFORMER_MOD = "meta-llama/Llama-3.2-1B" # The transformer model to use

############################
# HELPERS
############################

def timer(start_time, end_time):
    """
    Times a function. Mostly used for debugging and tuning.
    
    For example:

    start_time = time.time()
    {some function}
    end_time = time.time()

    timer(start_time, end_time)

    """
    duration = end_time - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    hhmmss_duration = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    print(f"Total processing time: {hhmmss_duration}.")

####

def empty_gpu_cache():
    """
    A platform agnostic function to clear the GPU cache.
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

####

def get_filepaths(inputDir):
    """
    Gathers sorted list of XMLs from a directory (recursive).
    """

    filepaths = []
    if not os.path.isdir(inputDir):
        print("Directory doesn't exist.")
        return [] # Exits if directory does not exist.
    
    for root, dirs, files in os.walk(inputDir): # Walks recursively through file tree
        files.sort()
        dirs.sort()
        for filename in files:
            if filename.endswith('.xml'): # Selects only XMLs
                filepaths.append(os.path.join(root, filename)) # Creates list of filepaths

    if not filepaths:
        print("No XMLs found")
        return [] # Exits if no XMLs exist
    
    return filepaths

####

def initialize_models(spacy_model, hf_model):
    """
    Activates NLP and LLM models.
    """
    
    
    # Prepare SpaCy #

    if torch.cuda.is_available():
        spacy.require_gpu()
        spacy_device = "GPU"
    else:
        spacy_device = "CPU"

    nlp = spacy.load(spacy_model)

    
    # Prepare HuggingFace Transformer Model (using Accelerator) #

    accelerator = Accelerator() # Memory management tool

    tokenizer = AutoTokenizer.from_pretrained(hf_model,use_fast=True) # Creates tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    model = AutoModelForCausalLM.from_pretrained( # Creates PyTorch model and pushes to GPU
        hf_model, device_map="auto", torch_dtype=torch.float16
    )
    model.eval() # Sets model to evaluation mode
    model = accelerator.prepare(model) # Stability wrapper; handles data types and memory allocation

    print(f"Transformer initialized on device {accelerator.device}.\nSpaCy model initialized on {spacy_device}.")

    return nlp, accelerator, tokenizer, model
    
####
    
def compute_iou(a_start, a_end, b_start, b_end):
    """
    Intersection over Union
    """
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    if inter == 0: return 0.0
    union = (a_end - a_start) + (b_end - b_start) - inter
    return inter / union if union > 0 else 0.0

############################
# DATA PROCESSING
############################

def XML_tupler(filepath):
    """
    Parses one XML into a list of (text, context) tuples.
    """

    sentence_tuples = [] # Creates holding list for tuples
    base_filename = os.path.basename(filepath) # Gathers filename w/out path
    filename_no_ext = os.path.splitext(base_filename)[0] # Gathers filename w/out .xml extension
    sentence_counter = 0 # For forming sentence ID number

    try:
        tree = ET.parse(str(filepath))
        root = tree.getroot() # Finds root of XML tree
    except ET.ParseError as e: # Sanity check for invalid XML
        print(f" Parse error {e}. Skipping file.")
        return []

    for modality, tag_type in [("written", ".//p")]: # Filters for written modality. Add ("spoken", ".//u") to also get spoken
            for element in root.findall(tag_type):
                for sentence_tag in element.findall(".//s"): # Extracts sentences
                    words = [
                        child.text.strip() # Removes extra spaces
                        for child in sentence_tag
                        if child.tag in ['w', 'c'] and child.text is not None # Removes empty words
                    ]
                    
                    if words:
                        sentence_text = ' '.join(words).strip() # Joins words into a sentence with only one space between words and removes empty sentences
                        sentence_counter += 1 
                        FSID = f"{filename_no_ext}_{sentence_counter:04d}"  # FSID = File Sentence ID, with leading zeroes

                        metadata = {
                            "FSID" : FSID,
                            "filename" : base_filename,
                            "modality" : modality
                        }
                        sentence_tuples.append((sentence_text, metadata))

    return sentence_tuples

####

def surprisal_calc(sentence_tuples, tokenizer, model, accelerator, batch_num):
    """
    Calculates surprisal values for each token in the text using HuggingFace model. 
        - Uses BATCH, CONTEXT, TOKEN_LIM, and TRANSFORMER_MOD config options. 
    """

    ### SETUP ###

    all_surprisals = []

    # Lists of sentences and sentence metadata
    sents = [sentence[0] for sentence in sentence_tuples]
    sents_meta = [sentence[1] for sentence in sentence_tuples]

    # Start and end idx of batch and context window
    batch_start = batch_num * BATCH
    batch_start_tok_idx = 0
    batch_end = min(((batch_num * BATCH) + BATCH), len(sents))
    context_start = max(0, (batch_start - CONTEXT))

    # Sentences in batch and context
    batch_context_sents = sents[context_start:batch_end]
    context_len = batch_start - context_start

    # Holding list for token ids
    flat_tok_ids = []
    targets = []

    # BOS token ID number
    bos_id = tokenizer.bos_token.id
    if isinstance(bos_id, list): # Sometimes returns as a list, this forces an integer
        bos_id = bos_id[0]

    # Loop over sentences in batch and context
    for idx, sent in enumerate(batch_context_sents):

        if batch_num == 0 and idx == 0: # If first token of first batch
            current_sent = sent # No initial space
        else:
            current_sent = " " + sent # initial space
        
        # Tokenizer ids for current sent
        sent_tok_output = tokenizer(current_sent, add_special_tokens = False)['input_ids']

        if sent_tok_output and isinstance(sent_tok_output[0], list):
            sent_tok_ids = [tok for sent in sent_tok_output for tok in sent] # Sentence token ids in single list
        else:
            sent_tok_ids = sent_tok_output

        # Adds <BOS> token if first item in batch:
        if idx == 0: 
            flat_tok_ids.append(bos_id)

        # Extends rest of sentence to flattened list
        flat_tok_ids.extend(sent_tok_ids)

        if idx >= context_len: # If sentence is after context window
            target_entry = {
                'text' : current_sent,
                'global_idx' : context_start + idx,
                'token_len' : len(sent_tok_ids)
            }
            targets.append(target_entry)

        if idx < context_len: # If sentence is within context window
            batch_start_tok_idx = len(flat_tok_ids)

    ### PYTORCH ###

    tensor = torch.tensor(flat_tok_ids, device = accelerator.device) # Creates 1D tensor of token id stream
    tensor_len = len(tensor)

    target_start_offset = max(1, batch_start_tok_idx)

    for i in tqdm(range(target_start_offset, tensor_len, TOKEN_LIM), desc = "Calculating Surprisal", position = 1, leave = False):

        # Start and end points for slicing
        target_start = i # Start of target range == current item
        target_end = min(i + TOKEN_LIM, tensor_len) # End of target; either full tensor or the token limit
        target_context_start = max(0, target_start - CONTEXT) # preceding context window, if available

        # Slice of context + batch
        slice = tensor[target_context_start : target_end].unsqueeze(0).to(accelerator.device) 

        # Surprisal calculation

        with torch.no_grad(): # Use model in eval mode rather than training mode

            # Calculating probability
            outputs = model(slice) # Forward pass of slice
            logits = outputs.logits

            # Transform to log probability
            logit_start_idx = (target_start - target_context_start) - 1 # Beginning bound of relevant slice
            logit_end_idx = (target_end - target_context_start) # End bound of relevant slice
            relevant_logits = logits[:, logit_start_idx:logit_end_idx, :] # Relevant logit slice
            log_probs = torch.log_softmax(relevant_logits, dim=-1) # Convert to log probability

            # Grab only probabilities for actual appearing words in corpus
            target_words = tensor[target_start:target_end].unsqueeze(0).unsqueeze(-1) # Converts words from text into 3D tensor
            gathered_probs = torch.gather(log_probs, dim=2, index = target_words) # Discards probs for non-appearing words
            
            # Send final values back to the CPU as a 1D half precision tensor
            final_probs = gathered_probs.squeeze().to(torch.float16).cpu()

            # Convert to surprisal
            surprisal = (-final_probs / math.log(2))

            # Append to list
            if surprisal.dim() == 0:
                all_surprisals.append(surprisal.item())
            else: 
                all_surprisals.extend(surprisal.tolist())

        # Cleanup
        del slice, outputs, logits, log_probs, target_words, gathered_probs, final_probs
    
    ### PACKAGE ###

    surp_results = [] # Holding list for surprisal results
    current_surp_idx = 0 # Pointer for surprisal values

    for target in targets:
        n_tokens = target['token_len'] 

        sent_surps = all_surprisals[current_surp_idx : current_surp_idx + n_tokens]
        current_surp_idx += n_tokens

        global_idx = target['global_idx'] # Globally stable idx pointer for each token

        # Sentence Metadata
        current_meta = sents_meta[global_idx].copy()
        current_meta['alignment_text'] = target['text']

        # Result tuple output
        result_tuple = (
            sents[global_idx],
            current_meta,
            sent_surps
        )

        surp_results.append(result_tuple)

    del tensor, flat_tok_ids, all_surprisals
    gc.collect()
    empty_gpu_cache()

    return surp_results

####

def spacy_streamer(sentence_tuples, nlp):
    """
    Processes sentences into SpaCy docs and passes them one at a time.
    """

    # Save sentence metadata and surprisals into a doc entry
    if not Doc.has_extension("sent_meta"):
        Doc.set_extension("sent_meta", default = None)
    if not Doc.has_extension("sent_surps"):
        Doc.set_extension("sent_surps", default = None)

    # Compress triple tuples into double tuples so SpaCy can handle them: 
    compressed_tuples = []
    for item in sentence_tuples:
        text = item[0]
        context = (item[1], item[2])
        compressed_tuples.append((text, context))

    # Run NLP
    doc_pipe = nlp.pipe(compressed_tuples, as_tuples = True, batch_size = SPACY_BATCH)

    # Embed metadata and surprisal into the Doc
    for doc, context in doc_pipe:
        metadata, surprisals = context
        doc._.sent_meta = metadata
        doc._.sent_surps = surprisals

        # Yield doc for further processing
        yield doc

