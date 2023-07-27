from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd

import torch
import os
os.environ['TRANSFORMERS_CACHE'] = '/project/SDS/research/christ_research/Llama 2/llama2-70b/cache'
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer

)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from dotenv import load_dotenv
import os

# Load the environmental variables from the .env file
load_dotenv()

token= os.getenv('huggingface_token')

from huggingface_hub import login
login(token = token)

dataset = load_dataset('csv', data_files = "app/feedback.csv")

# 80% train, 20% test + validation
train_testvalid = dataset.train_test_split(test=0.2)
# Split the 10% test + valid in half test, half valid
test_valid = train_test_dataset['test'].train_test_split(test=0.5)
# gather everyone if you want to have a single DatasetDict
train_test_valid_dataset = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'valid': test_valid['train']})

train_test_valid_dataset['train'][1]