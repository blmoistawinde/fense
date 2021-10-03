from collections import defaultdict
import os
import re
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from scipy.io import loadmat
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from collections import defaultdict

class FlatDataset(Dataset):
    def __init__(self, input_dir, tokenizer, max_len, split="train"):
        assert split in {"train", "val", "test"}
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = []
        self.labels = []
        df = pd.read_csv(os.path.join(input_dir, f"synthetic_{split}.csv"), index_col=0)
        for rid, row in df.iterrows():
            sample = {}
            sample["text"] = row["caption"]
            tokenized = tokenizer(sample["text"], truncation=True, padding='max_length', max_length=max_len)
            for k, v in tokenized.items():
                sample[k] = v
            self.data.append(sample)
            self.labels.append(row.values[1:].astype(int))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index], self.labels[index]

def my_collate_flat(data):
    labels = []
    processed_batch = defaultdict(list)
    for item, label in data:
        for k, v in item.items():
            processed_batch[k].append(v)
        labels.append(label)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        processed_batch[k] = torch.LongTensor(processed_batch[k])
    labels = torch.FloatTensor(np.array(labels))
    return processed_batch, labels

class FlatDataModule(pl.LightningDataModule):
    def __init__(self, bs, input_dir, tokenizer, max_len):
        super().__init__()
        self.bs = bs
        self.input_dir = input_dir
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def setup(self, stage):
        if stage == "fit":
            self.train_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "train")
            self.val_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "val")
        elif stage == "test":
            self.test_set = FlatDataset(self.input_dir, self.tokenizer, self.max_len, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.bs, collate_fn=my_collate_flat, shuffle=True, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.bs, collate_fn=my_collate_flat, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.bs, collate_fn=my_collate_flat, pin_memory=True, num_workers=4)


def infer_preprocess(tokenizer, texts, max_len):
    batch = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
    for k in ['input_ids', 'attention_mask', 'token_type_ids']:
        batch[k] = torch.LongTensor(batch[k])
    return batch

if __name__ == "__main__":
    model_id = 'prajjwal1/bert-tiny'
    # note that we need to specify the number of classes for this task
    # we can directly use the metadata (num_classes) stored in the dataset
    # model = AutoModelForSequenceClassification.from_pretrained(model_id, 
                # num_labels=train_dataset.features["label"].num_classes)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    data_module = FlatDataModule(4, "./dataset", tokenizer, max_len=64)
    data_module.setup("fit")
    train_loader = data_module.train_dataloader()
    batch0, labels0 = next(iter(train_loader))
    print(batch0)
    print(labels0)
    import pdb; pdb.set_trace()