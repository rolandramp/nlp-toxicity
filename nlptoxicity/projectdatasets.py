import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class NeuroToxicBertDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, content='Comment', label='Label'):
        self.tokenizer = tokenizer
        self.data = df[[content, label]].reset_index()
        self.text = content
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[self.text][idx]
        label = self.data[self.label][idx]
        encoding = self.tokenizer(text, return_tensors='pt', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label)}


class NeuroToxicBertNerDataset(Dataset):

    def __init__(self, raw_data, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.data = self.do_alignment(raw_data=self.raw_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return {'input_ids': data['input_ids'],
                'token_type_ids': data['token_type_ids'],
                'attention_mask': data['attention_mask'],
                'labels': data['labels']}

    def __align_labels_with_tokens(self, labels, word_ids):
        new_labels = labels
        new_labels.append(0)
        new_labels.insert(0, 0)
        # pad to needed length
        new_labels = new_labels + [0 for _ in range(len(word_ids) - len(new_labels))]
        return new_labels

    def __tokenize_and_align_labels(self, examples):
        tokenized_inputs = self.tokenizer(examples["Comment"], truncation=True, padding=True)
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(self.__align_labels_with_tokens(labels, word_ids))
        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    def do_alignment(self, raw_data):
        tokenized_datasets = raw_data.map(
            self.__tokenize_and_align_labels,
            batched=True,
            remove_columns=raw_data.column_names,
        )
        return tokenized_datasets
