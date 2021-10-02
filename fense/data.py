from collections import defaultdict
import os
import re
import torch
from transformers import AutoTokenizer
from collections import defaultdict

def infer_preprocess(tokenizer, texts, max_len):
    batch = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        batch[k] = torch.LongTensor(batch[k])
    return batch
