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

    - BATCH: The number of sentences to manage at once. Context builds from first token to last within the batch. (NOTE: No file in the BNC contains >600 sentences, setting to a higher value will process the entire file at once).

    - CONTEXT: The number of preceding sentences to use as context at the beginning of a non-initial batch. The first sentence in a non-initial batch will begin with this many previous sentences as context, and the context will build from the first to last token of the batch. 

    - TOKEN_LIM: Limit of tokens to simultaneously calculate surprisal for. This will not effect context; each token will receive the normal amount of context, the calculation will just proceed consecutively rather than concurrently.

    - OVERWRITE: Whether to overwrite existing files in the given OUTPUT_DIR or to keep and skip existing output files. Setting to 0 essentially resumes from where the calculation left off. 

    - INPUT_DIR: The directory of the input XML files, in this case the BNC1994 corpus. May be a hierarchical nested file tree. 

    - OUTPUT_DIR: The directory of the output CSV files. Note that each input XML will have its own output CSV. They must be merged later, usually in R. 

    - SPACY_MOD: The SpaCy NLP model to be used. "en_core_web_trf" is recommended.

    - TRANSFORMER_MOD: The transformer LLM model to be used. Currently programmed to use HuggingFace. "meta-llama/Llama-3.2-1B" is recommended, but a larger or smaller model may be used depending on computing resources. 
"""
BATCH = 1024 
CONTEXT = 32 
TOKEN_LIM = 16384 

OVERWRITE = 1 

INPUT_DIR = "D:/BNC Full Data/BNCFiles/Full BNC1994/download/Texts" # The directory of the input XML files
OUTPUT_DIR = "D:/BNC Full Data/12-11_11AM Run/CSV" # The directory of the output CSV files

SPACY_MOD = "en_core_web_trf" # The SpaCy model to use
TRANSFORMER_MOD = "meta-llama/Llama-3.2-1B" # The transformer model to use