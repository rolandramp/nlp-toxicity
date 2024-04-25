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
    def __init__(self, data, tokenizer, labels_to_ids):
        self.labels_to_ids = labels_to_ids
        # lb = [i.split() for i in data['ner_label'].values.tolist()]
        lb = data['ner_label'].values
        txt = data['Comment'].values.tolist()
        self.texts = [tokenizer(str(i),
                                padding='max_length', max_length=100, truncation=True, return_tensors="pt") for i in
                      txt]
        self.labels = [self.align_label(i, j, tokenizer, self.labels_to_ids) for i, j in zip(txt, lb)]

    def __len__(self):
        return len(self.labels)

    def get_batch_data(self, idx):
        return self.texts[idx]

    def get_batch_labels(self, idx):
        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):
        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels

    def align_label(self, texts, labels, tokenizer, labels_to_ids):
        tokenized_inputs = tokenizer(texts, padding='max_length', max_length=100, truncation=True)

        word_ids = tokenized_inputs.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(labels_to_ids[labels[word_idx]])
                except:
                    label_ids.append(-100)
            else:
                try:
                    label_ids.append(labels_to_ids[labels[word_idx]] if True else -100)
                except:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        return label_ids