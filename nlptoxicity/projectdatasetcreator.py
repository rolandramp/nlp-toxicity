import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

from nlptoxicity.utils import load_vulgarities_entries, prepare_ner_data, prepare_ner_data_sentence_wise
from datasets import Dataset, ClassLabel, Sequence, DatasetDict


class NeuroToxicBertNerConllDatasetCreator:
    def __init__(self, path, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.tokenized_data_vulgarity_data_set = self.prepare_ner_data_conllu(load_vulgarities_entries(path))

    def get_train_dataset(self):
        return self.tokenized_data_vulgarity_data_set['train']

    def get_test_dataset(self):
        return self.tokenized_data_vulgarity_data_set['test']

    def get_val_dataset(self):
        return self.tokenized_data_vulgarity_data_set['val']

    def __tokenize_and_align_labels_new(self, examples):
        tokenized_inputs = self.tokenizer(examples["Comment"], truncation=True)

        labels = []
        for i, label in enumerate(examples["ner_label"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            for word_idx in word_ids:
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(0)
                else:
                    try:
                        label_ids.append(label[word_idx])
                    except IndexError:
                        print(word_idx)
                        label_ids.append(0)
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def prepare_ner_data_conllu(self, vulgarity_df: pd.DataFrame):
        data_vulgarity = prepare_ner_data_sentence_wise(vulgarity_df)

        train_data_vulgarity, test_data_vulgarity = train_test_split(data_vulgarity, test_size=.2, random_state=42)
        test_data_vulgarity, val_data_vulgarity = train_test_split(train_data_vulgarity, test_size=0.5, random_state=42)

        train_data_vulgarity_data_set = self.create_data_set_from_dataframe(train_data_vulgarity)
        test_data_vulgarity_data_set = self.create_data_set_from_dataframe(test_data_vulgarity)
        val_data_vulgarity_data_set = self.create_data_set_from_dataframe(val_data_vulgarity)

        return_dict = {'train': train_data_vulgarity_data_set,
                       'test': test_data_vulgarity_data_set,
                       'val': val_data_vulgarity_data_set}

        return DatasetDict(return_dict)

    def create_data_set_from_dataframe(self, train_data_vulgarity):
        train_data_vulgarity_data_set = Dataset.from_pandas(train_data_vulgarity)
        train_data_vulgarity_data_set = train_data_vulgarity_data_set.cast_column('ner_label',
                                                                                  Sequence(
                                                                                      ClassLabel(names=['O', 'Vul'])))
        train_data_vulgarity_data_set = train_data_vulgarity_data_set.map(self.__tokenize_and_align_labels_new,
                                                                          batched=True)
        return train_data_vulgarity_data_set


class NeuroToxicBertNerDatasetCreator:
    def __init__(self, path, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        self.raw_data = prepare_ner_data(load_vulgarities_entries(path), self.tokenizer)

    def get_train_dataset(self):
        return self.raw_data['train']

    def get_test_dataset(self):
        return self.raw_data['test']

    def get_val_dataset(self):
        return self.raw_data['val']
