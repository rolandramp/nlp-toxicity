import datetime
import os
import pickle
import random
import re
import string
from itertools import chain

import nltk
import pandas as pd
import torch
from datasets import Dataset, ClassLabel, Sequence, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import BertModel, RobertaModel, AutoTokenizer, AutoModelForSequenceClassification, \
    PreTrainedTokenizer, AutoModelForTokenClassification, BertForTokenClassification

from nlptoxicity import logger

import csv
import pickle
import time
from typing import List

import pandas as pd
import stanza
import torch
import torch.optim as optim
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split as split
from torch import nn
from torch.utils.data import DataLoader

stanza.download('de')

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
from tqdm import tqdm

import csv

import nltk
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from stanza.utils.conll import CoNLL



def load_and_split_data_set(data_path, test_size=0.2, splits_path=None, kind="pruned"):
    df = __prepare_data(data_path)
    if splits_path:
        splits = pickle.load(open(splits_path, "rb"))
        train_idx = splits[kind]["train"]
        test_idx = splits[kind]["test"]
        val_idx = splits[kind]["val"]
        df_train = df.loc[train_idx]
        df_test = df.loc[test_idx]
        df_val = df.loc[val_idx]
    else:
        df_train, df_test = train_test_split(df, test_size=test_size, stratify=df[['Label']])
        df_test, df_val = train_test_split(df_test, test_size=0.5, stratify=df_test[['Label']])
    return df_train, df_test, df_val


def load_conllu_data_set(data_path, splits_path, kind='full'):
    splits = pickle.load(open(splits_path, "rb"))
    doclist, infolist = read_docs_from_conllu(data_path)
    corpus = [' '.join(convert_stanza_doc_to_list_of_lemmas(document)) for document in doclist]
    train_tuple_list = [(corpus[idx], infolist[idx]['label']) for idx in splits[kind]['train']]
    train_df = pd.DataFrame(train_tuple_list, columns=['Comment', 'Label'])
    test_tuple_list = [(corpus[idx], infolist[idx]['label']) for idx in splits[kind]['test']]
    test_df = pd.DataFrame(test_tuple_list, columns=['Comment', 'Label'])
    val_tuple_list = [(corpus[idx], infolist[idx]['label']) for idx in splits[kind]['val']]
    val_df = pd.DataFrame(val_tuple_list, columns=['Comment', 'Label'])
    return train_df, test_df, val_df


def check_if_cuda_is_available():
    """
    Just to check the availability  of CUDA.
    """
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))


def get_german_bert():
    """
    Loads and returns the GBERT base model form huggingface
    """
    german_bert = BertModel.from_pretrained('bert-base-german-cased')
    return german_bert


def get_bert_sequence_classification_model(freeze_first_n_layers=None, model_name='deepset/gbert-base'):
    """
    Loads a given transformer model (GBERT base by default), freezes the desired number of layers and adding a classification layer on top.
    :param freeze_first_n_layers: numbers of layers to freeze from top transformer layer on
    :param model_name: model to use
    """
    gbert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    if freeze_first_n_layers:
        modules = [gbert.bert.embeddings, *gbert.bert.encoder.layer[:freeze_first_n_layers]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        # for name, param in gbert.named_parameters():
        #    print(f'{name}  {param.requires_grad}')
    return gbert


def get_trained_sequence_classification_model(model_path: str):
    """
    Loads a pretrained transformer model with classification head from given path
    :param model_path: location of the model
    """
    if os.path.exists(model_path):
        return AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    else:
        logger.error(f'model does not exist on path {model_path}')
        return None


class GBertNerModel(torch.nn.Module):
    def __init__(self, freeze_first_n_layers=None):
        super(GBertNerModel, self).__init__()
        label_ids = list(['O', 'Vul'])
        label2id = {label: id for id, label in enumerate(label_ids)}
        id2label = {id: label for label, id in label2id.items()}
        self.bert = BertForTokenClassification.from_pretrained('deepset/gbert-base', num_labels=len(label_ids))
        if freeze_first_n_layers:
            modules = [self.bert.bert.embeddings, *self.bert.bert.encoder.layer[:freeze_first_n_layers]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

    def forward(self, input_id, mask, label):
        output = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        return output


def get_deepset_gbert_base_mode_ner2(freeze_first_n_layers=None):
    return GBertNerModel(freeze_first_n_layers)


def get_transformer_model_tokenizer(model_name='deepset/gbert-base'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def get_xlm_roberta_base():
    xlm_rpberta_base = RobertaModel.from_pretrained('xlm-roberta-base')
    return xlm_rpberta_base


def __prepare_data(data_path):
    return remove_duplicate_comments(decode_binary_comments(read_docs_from_json(data_path)))


def load_vulgarities_entries(data_path):
    all_data_df = read_docs_from_json(data_path)
    all_data_with_tags = all_data_df[all_data_df.Tags.apply(len) > 0]
    all_data_vulgarity = all_data_with_tags[
        (all_data_with_tags.Tags.apply(lambda x: [d['Tag'] for d in x if d['Tag'] == 'Vulgarity']).apply(len) > 0)]
    vulgarities = all_data_vulgarity.loc[:, 'Tags'].apply(
        lambda x: [d['Token'] for d in x if d['Tag'] == 'Vulgarity'])
    all_data_vulgarity.loc[:, 'Vulgarity'] = vulgarities
    return all_data_vulgarity


def prepare_ner_data(all_data_vulgarity: pd.DataFrame, tokenizer: PreTrainedTokenizer):
    comment_sentences = all_data_vulgarity.reset_index().loc[:, 'Comment']
    vulgarity_words = all_data_vulgarity.reset_index().loc[:, 'Vulgarity']
    raw_dict = {}
    idx = 0
    for single_comment, vulgarity in zip(comment_sentences, vulgarity_words):
        raw_dict[idx] = {}
        v_enc = []
        for v in vulgarity:
            v_enc.append(tokenizer.encode(v))
        vuls = list(set(list(chain.from_iterable(v_enc))))
        single_comment_encoded = tokenizer(single_comment, return_tensors='pt', truncation=True)
        ner_labels = ['Vul' if token in vuls else 'O' for token in single_comment_encoded.input_ids[0]][
                     1:-1]  # remove first and last token (CLS SEP)
        raw_dict[idx]['Comment'] = single_comment
        raw_dict[idx]['ner_tags'] = ner_labels
        idx = idx + 1

    data_list = []
    for idx, data in raw_dict.items():
        data_list.append({
            'id': idx,
            'Comment': data['Comment'],
            'ner_tags': data['ner_tags']
        })
    # Convert the list to a Hugging Face Dataset
    train_data_list, test_data_list = train_test_split(data_list, test_size=.2, random_state=42)
    test_data_list, val_data_list = train_test_split(test_data_list, test_size=0.5, random_state=42)

    train_dataset = Dataset.from_dict({k: [d[k] for d in train_data_list] for k in train_data_list[0]})
    train_dataset = train_dataset.cast_column('ner_tags', Sequence(ClassLabel(names=['O', 'Vul'])))

    test_data_set = Dataset.from_dict({k: [d[k] for d in test_data_list] for k in test_data_list[0]})
    test_data_set = test_data_set.cast_column('ner_tags', Sequence(ClassLabel(names=['O', 'Vul'])))

    val_data_set = Dataset.from_dict({k: [d[k] for d in val_data_list] for k in val_data_list[0]})
    val_data_set = val_data_set.cast_column('ner_tags', Sequence(ClassLabel(names=['O', 'Vul'])))

    # Create a DatasetDict
    raw_data = DatasetDict({"train": train_dataset,
                            "test": test_data_set,
                            "val": val_data_set})
    return raw_data


def prepare_ner_data_conllu(vulgarity_df: pd.DataFrame):
    data_vulgarity = vulgarity_df.loc[:, ['Comment', 'Vulgarity']]
    comment_words_list = extract_text(vulgarity_df['Comment'])
    vulgarity_word_list = extract_vul(vulgarity_df['Vulgarity'])
    vulgarity_df['Comment_list'] = comment_words_list
    vulgarity_df['Vulgarity'] = vulgarity_word_list
    vulgarity_df = vulgarity_df.reset_index()
    ner_labels_list = []
    comment_list = []
    vulgarity_list = []
    # split up to single sentences
    for idx in range(len(data_vulgarity)):
        comment = vulgarity_df.loc[idx, 'Comment_list']
        vulgarity = vulgarity_df.loc[idx, 'Vulgarity']
        for c in comment:
            ner_labels_list.append(['Vul' if token in vulgarity else 'O' for token in c])
            comment_list.append(c)
            vulgarity_list.append(vulgarity)
    vulgarity1_df2 = pd.DataFrame(list(zip(comment_list, ner_labels_list, vulgarity_list)),
                                  columns=['Comment', 'ner_label', 'Vulgarity'])
    vulgarity1_df2 = vulgarity1_df2.loc[:, ['Comment', 'ner_label']]
    vulgarity1_df2['Comment'] = vulgarity1_df2['Comment'].str.join(' ')
    vulgarity1_df2['ner_label'] = vulgarity1_df2['ner_label'].str.join(' ')
    # only keep examples with vulgarities
    vulgarity1_df2 = vulgarity1_df2[vulgarity1_df2["ner_label"].str.contains("Vul")]
    vulgarity1_df2['ner_label'] = vulgarity1_df2['ner_label'].str.split()
    return vulgarity1_df2


def create_folder_with_timestamp_and_random(base_path: str, model_type: str) -> str:
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    random_number = random.randint(1, 1000)
    folder_name = f"{timestamp}_{model_type}_{random_number}"
    try:
        return_folder = os.path.join(base_path, folder_name)
        os.makedirs(return_folder)
        logger.info(f"Folder '{folder_name}' created successfully.")
        return return_folder
    except OSError as e:
        logger.error(f"Failed to create folder. Error: {e}")
    return None


def clean_and_augment_data(data: pd.DataFrame, clean: bool = True, augment: bool = True):
    '''Helper function to perform simple data cleaning as well as data augmentation.
    Intended to be used for training data only.'''

    nltk.download("stopwords")
    stopwords = nltk.corpus.stopwords.words('german')

    if clean:
        # replace punctuation with empty string, removing them
        # re.sub is passed a set of punctuation, each of which is escaped as they are often special characters for regex
        data["Comment"] = data["Comment"].apply(
            lambda x: re.sub("[" + "\\".join(list(string.punctuation)) + "]", "", x))

        # remove stopwords
        for word in stopwords:
            data["Comment"] = data["Comment"].str.replace(word, "")

    if augment:
        # perform class balancing
        max_size = data["Label"].value_counts().max()  # amount of majority class
        data_list = [data]  # list containing original data
        for class_index, group in data.groupby('Label'):
            data_list.append(
                group.sample(max_size - len(group), replace=True))  # sampling the difference from minority class
        new_data = pd.concat(data_list).sample(frac=1)  # recombine, and shuffle just in case
        data = new_data

    return data


def prepare_ner_data_sentence_wise(vulgarity_df):
    data_vulgarity = vulgarity_df.loc[:, ['Comment', 'Vulgarity']]
    comment_words_list = extract_text(vulgarity_df['Comment'])
    vulgarity_word_list = extract_vul(vulgarity_df['Vulgarity'])
    vulgarity_df['Comment_list'] = comment_words_list
    vulgarity_df['Vulgarity'] = vulgarity_word_list
    vulgarity_df = vulgarity_df.reset_index()
    ner_labels_list = []
    comment_list = []
    vulgarity_list = []
    # split up to single sentences
    for idx in range(len(data_vulgarity)):
        comment = vulgarity_df.loc[idx, 'Comment_list']
        vulgarity = vulgarity_df.loc[idx, 'Vulgarity']
        for c in comment:
            ner_labels_list.append(['Vul' if token in vulgarity else 'O' for token in c])
            comment_list.append(c)
            vulgarity_list.append(vulgarity)
    vulgarity1_df2 = pd.DataFrame(list(zip(comment_list, ner_labels_list, vulgarity_list)),
                                  columns=['Comment', 'ner_label', 'Vulgarity'])
    vulgarity1_df2 = vulgarity1_df2.loc[:, ['Comment', 'ner_label']]
    vulgarity1_df2['Comment'] = vulgarity1_df2['Comment'].str.join(' ')
    vulgarity1_df2['ner_label'] = vulgarity1_df2['ner_label'].str.join(' ')
    # only keep examples with vulgarities
    vulgarity1_df2 = vulgarity1_df2[vulgarity1_df2["ner_label"].str.contains("Vul")]
    vulgarity1_df2['ner_label'] = vulgarity1_df2['ner_label'].str.split()
    data_vulgarity = vulgarity1_df2
    return data_vulgarity


def read_docs_from_json(filename: str) -> pd.DataFrame:
    df = pd.read_json(filename)
    return df


def remove_duplicate_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    If there are multiple comments with different label, the majority of this labels will be used.
    If number of labels is even on toxic or not we go for the toxic label.
    :param df: data frame with number of duplicate comment entries
    :return: df without any duplicate comments
    """
    reduced_comments_df = df.groupby(['Article_title', 'Comment'])['Label'].agg(
        Label=lambda x: list(pd.Series.mode(x))).reset_index()
    reduced_comments_df['Label'] = reduced_comments_df['Label'].map(max)
    return reduced_comments_df


def read_docs_from_csv(filename):
    docs = []
    with open(filename) as csvfile:
        reader = csv.reader(csvfile)
        for text, label in tqdm(reader):
            words = nltk.word_tokenize(text)
            docs.append((words, label))

    return docs


def read_docs_from_conllu(filename, comment_split_token="NEW\n", info_split_token=";;"):
    """Reads in concatenated CoNLLu output as generated by the write_docs_to_conllu function below. """
    doclist = []
    infolist = []
    with open(filename, mode="r", encoding='utf-8') as f:
        commentlist = f.read().split(comment_split_token)
        for comment in commentlist:
            if len(comment) < 1:
                # empty line at the start
                continue
            lines = comment.strip().split("\n")
            id, title, label = lines[0].split(info_split_token)
            doc = CoNLL.conll2doc(input_str="\n".join(lines[1:]))
            doclist.append(doc)
            infolist.append({"id": int(id), "title": title, "label": int(label)})

    return doclist, infolist


def search_toxic_comments(pattern, df):
    return df[df['Comment'].str.contains(pattern, case=False, na=False, regex=True)]


def search_party(pattern, df, r=50):
    for index, row in df.iterrows():
        text = row['Comment']
        matches = list(re.finditer(pattern, text, flags=re.I))
        if not matches:
            continue
        article_title = row['Article_title']
        comment = row['Comment']
        for match in matches:
            i, j = match.span()
            start = max(i - r, 0)
            end = j + r
            context = comment[start:end]

            print(f"Article Title: {article_title}\n\n...{context}...\n\n")


def find_names_in_comments(df):
    pattern = r'\b[A-Z][a-z]+\b'
    # A new column to store the extracted names
    df['Extracted_Names'] = df['Comment'].apply(lambda text: re.findall(pattern, text))
    return df


def reduce_repeating_characters_except_specific(comment):
    # Regex pattern to match repeating characters (3 or more repetitions) within words
    pattern = r'\b(\w*(?![.!?])\w)\1{2,}\b'

    # Reducing repeating characters to a maximum of two repetitions
    def reduce(match):
        return match.group(1) * 2

    reduced_comment = re.sub(pattern, reduce, comment)
    return reduced_comment


def decode_binary_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Searches for colums only containing binary data and decodes them back to text
    :param df: DataFrame containing binary data
    :return: df with decoded fields
    """
    temp_df = df.copy()
    temp_df.loc[temp_df['Comment'].str.isdigit(), 'Comment'] = temp_df[temp_df['Comment'].str.isdigit()][
        'Comment'].map(lambda x: convert_binary_to_ascii(x))
    return temp_df


def convert_binary_to_ascii(binary: str) -> str:
    """
    reads a binary input and decodes the text
    :param binary: binary input
    :return: decodes text
    """
    if len(binary) % 8 == 0:
        chunks = [binary[0 + i:8 + i] for i in range(0, len(binary), 8)]
        return ''.join([chr(int(i, 2)) for i in chunks])
    else:
        return binary


def write_docs_to_conllu(docs: list, filename: str, df: pd.DataFrame = None, comment_split_token: str = "NEW\n",
                         info_split_token: str = ";;"):
    '''Writes a list of stanza docs in a concatenated CoNLLu format, preserving information as well as additional information
    such as the title of the article and the label given by annotators.'''
    ist = info_split_token
    with open(filename, "w", encoding="utf-8") as f:
        for n, c in enumerate(docs):
            # write demarcation of individual comments to be able to piece them together again more "easily"
            f.write(f"{comment_split_token}{n}{ist}{df.iloc[n]['Article_title']}{ist}{df.iloc[n]['Label']}\n")
            CoNLL.write_doc2conll(c, f)
        f.write("\n")
    print(f"output written to {filename}.")


def split_train_dev_test(docs, train_ratio=0.8, dev_ratio=0.1):
    np.random.seed(2022)
    np.random.shuffle(docs)
    train_size = int(len(docs) * train_ratio)
    dev_size = int(len(docs) * dev_ratio)
    return (
        docs[:train_size],
        docs[train_size: train_size + dev_size],
        docs[train_size + dev_size:],
    )


def calculate_tp_fp_fn(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[0, 0] + cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = 2 * precision * recall / (precision + recall)

    return tp, fp, fn, precision, recall, fscore


class WordcountDataset:
    def __init__(self, corpus):
        '''corpus is a dictionary with index:text elements.'''
        # could make this more robust by allowing lists etc but for this purpose hardcoding is sufficient
        assert isinstance(corpus, dict), "the corpus is required to be a dictionary in index:text format."
        self.index = list(corpus.keys())
        self.values, self.features = self.vectorize(corpus.values())
        self._current = 0
        self.len = len(self.index)

    def vectorize(self, texts):
        vectorizer = CountVectorizer()
        vectorizer.fit(texts)
        return vectorizer.transform(texts), vectorizer.vocabulary

    def __getitem__(self, i):
        return self.values[i]

    def __len__(self):
        return self.len


def convert_stanza_doc_to_list_of_lemmas(document):
    """
    method to transform stanza document to list of text tokens.
    the lamma part is used as token
    :param document: stanza document
    :return: list of tokens
    """
    words_in_doc = []
    for sentence in document.sentences:
        for word in sentence.words:
            words_in_doc.append(word.lemma)
    return words_in_doc


def convert_stanza_doc_to_sentences_with_list_of_words(document):
    """
    method to transform stanza document to list of sentences.
    the text part is used
    :param document: stanza document
    :return: list of sentences containing tokens
    """
    sentences = []
    for sentence in document.sentences:
        words_in_doc = []
        for word in sentence.words:
            words_in_doc.append(word.text)
        sentences.append(words_in_doc)
    return sentences


def convert_stanza_doc_to_list_of_words(document):
    """
    method to transform stanza document to list of text tokens.
    the lamma part is used as token
    :param document: stanza document
    :return: list of tokens
    """
    words_in_doc = []
    for sentence in document.sentences:
        for word in sentence.words:
            words_in_doc.append(word.text)
    return words_in_doc


def convert_stanza_doc_to_list_of_words_cleaned(document):
    """
    method to transform stanza document to list of text tokens.
    the lamma part is used as token
    :param document: stanza document
    :return: list of tokens
    """
    words_in_doc = []
    for sentence in document.sentences:
        for word in sentence.words:
            if word.upos != "PUNCT" and word.upos != "NUM":
                words_in_doc.append(word.lemma)
    return words_in_doc


def extract_text(doc):
    nlp = stanza.Pipeline('de', processors='tokenize')
    return_list = []
    for d in doc:
        if isinstance(d, list):
            d = " ".join(d)
        return_list.append(convert_stanza_doc_to_sentences_with_list_of_words(nlp(d)))
    return return_list


def extract_vul(doc):
    nlp = stanza.Pipeline('de', processors='tokenize')
    return_list = []
    for d in doc:
        if isinstance(d, list):
            d = " ".join(d)
        return_list.append(convert_stanza_doc_to_list_of_words(nlp(d)))
    return return_list


def __create_document_vectorization(corpus):
    """
    method to create count vectorizer on document corpus
    :param corpus: all documents
    :return: vectorizer to transform documents into vectors
    """
    vectorizer: CountVectorizer = CountVectorizer()
    return vectorizer.fit(corpus)


def transform_stanza_doc_2_count_vectorizer(doclist: List):
    """
    method to create a count vectorizer on all documents
    :param doclist: list of stanza documents
    :return: vectorizer to transform documents into vectors
    """
    corpus = [' '.join(convert_stanza_doc_to_list_of_lemmas(document)) for document in doclist]
    return __create_document_vectorization(corpus)


# This is just for measuring training time!
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


class NeuroToxicDataset:
    def __init__(self, data_path, batch_size=64):
        self.data_path = data_path
        # Initialize the correct device
        # It is important that every array should be on the same device or the training won't work
        # A device could be either the cpu or the gpu if it is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = batch_size
        self.OUT_DIM = 1

        (self.train_iterator,
         self.valid_iterator,
         self.test_iterator,
         self.VOCAB_SIZE
         ) = self.load_toxic_data_set(self.data_path)

    def load_toxic_data_set(self, path, batch_size=64):
        """
        method to load data with prepared toxicity data
        :param path: paht of the pickled file
        :param batch_size: batch size of iterator
        :return: train val and test iterator
        """
        dataset_loaded = pickle.load(open(path, "rb"))
        vocab_size = dataset_loaded['train']['features'].shape[1]
        train_data = [
            (sample, label) for sample, label in
            zip(torch.from_numpy(dataset_loaded['train']['features'].toarray()).float().to(self.device),
                torch.from_numpy(dataset_loaded['train']['labels']).float().to(self.device))
        ]
        test_data = [
            (sample, label) for sample, label in
            zip(torch.from_numpy(dataset_loaded['test']['features'].toarray()).float().to(self.device),
                torch.from_numpy(dataset_loaded['test']['labels']).float().to(self.device))
        ]
        val_data = [
            (sample, label) for sample, label in
            zip(torch.from_numpy(dataset_loaded['val']['features'].toarray()).float().to(self.device),
                torch.from_numpy(dataset_loaded['val']['labels']).float().to(self.device))
        ]
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True)
        test_loader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False)
        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False)
        return train_loader, val_loader, test_loader, vocab_size


class NeuroToxicDatasetSklearn:
    def __init__(self, data_path):
        self.data_path = data_path
        (self.X_train,
         self.y_train,
         self.X_test,
         self.y_test,
         self.X_val,
         self.y_val
         ) = self.load_toxic_data_set(self.data_path)

    def load_toxic_data_set(self, path):
        dataset_loaded = pickle.load(open(path, "rb"))
        X_train = dataset_loaded['train']['features']
        y_train = dataset_loaded['train']['labels']
        X_test = dataset_loaded['test']['features']
        y_test = dataset_loaded['test']['labels']
        X_val = dataset_loaded['val']['features']
        y_val = dataset_loaded['val']['labels']
        return X_train, y_train, X_test, y_test, X_val, y_val


class Trainer:
    def __init__(
            self,
            dataset: NeuroToxicDataset,
            model: nn.Module,
            model_path: str = None,
            test: bool = False,
            lr: float = 0.001,
    ):
        self.dataset = dataset
        self.model = model

        # The optimizer will update the weights of our model based on the loss function
        # This is essential for correct training
        # The _lr_ parameter is the learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
        # self.criterion = nn.NLLLoss()

        # Copy the model and the loss function to the correct device
        self.model = self.model.to(dataset.device)
        self.criterion = self.criterion.to(dataset.device)

    def calculate_performance(self, preds, y):
        """
        Returns precision, recall, fscore per batch
        """
        # Get the predicted label from the probabilities
        rounded_preds = preds.round().detach().cpu().numpy()

        # Calculate the correct predictions batch-wise and calculate precision, recall, and fscore
        # WARNING: Tensors here could be on the GPU, so make sure to copy everything to CPU
        precision, recall, fscore, support = precision_recall_fscore_support(
            rounded_preds, y.cpu()
        )

        return precision[1], recall[1], fscore[1]

    def train(self, iterator):
        # We will calculate loss and accuracy epoch-wise based on average batch accuracy
        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0

        # You always need to set your model to training mode
        # If you don't set your model to training mode the error won't propagate back to the weights
        self.model.train()

        # We calculate the error on batches so the iterator will return matrices with shape [BATCH_SIZE, VOCAB_SIZE]
        for batch in iterator:
            text_vecs = batch[0]
            labels = batch[1]

            # We reset the gradients from the last step, so the loss will be calculated correctly (and not added together)
            self.optimizer.zero_grad()

            # This runs the forward function on your model (you don't need to call it directly)
            predictions = self.model(text_vecs)

            # Calculate the loss and the accuracy on the predictions (the predictions are log probabilities, remember!)
            loss = self.criterion(predictions, labels.reshape(-1, 1))

            prec, recall, fscore = self.calculate_performance(predictions, labels.reshape(-1, 1))

            # Propagate the error back on the model (this means changing the initial weights in your model)
            # Calculate gradients on parameters that requries grad
            loss.backward()
            # Update the parameters
            self.optimizer.step()

            # We add batch-wise loss to the epoch-wise loss
            epoch_loss += loss.item()
            # We also do the same with the scores
            epoch_prec += prec.item()
            epoch_recall += recall.item()
            epoch_fscore += fscore.item()
        return (
            epoch_loss / len(iterator),
            epoch_prec / len(iterator),
            epoch_recall / len(iterator),
            epoch_fscore / len(iterator),
        )

    # The evaluation is done on the validation dataset
    def evaluate(self, iterator):

        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0
        # On the validation dataset we don't want training so we need to set the model on evaluation mode
        self.model.eval()

        # Also tell Pytorch to not propagate any error backwards in the model or calculate gradients
        # This is needed when you only want to make predictions and use your model in inference mode!
        with torch.no_grad():
            # The remaining part is the same with the difference of not using the optimizer to backpropagation
            for batch in iterator:
                text_vecs = batch[0]
                labels = batch[1]

                predictions = self.model(text_vecs)
                loss = self.criterion(predictions, labels.reshape(-1, 1))

                prec, recall, fscore = self.calculate_performance(predictions, labels.reshape(-1, 1))

                epoch_loss += loss.item()
                epoch_prec += prec.item()
                epoch_recall += recall.item()
                epoch_fscore += fscore.item()

        # Return averaged loss on the whole epoch!
        return (
            epoch_loss / len(iterator),
            epoch_prec / len(iterator),
            epoch_recall / len(iterator),
            epoch_fscore / len(iterator),
        )

    def training_loop(self, train_iterator, valid_iterator, epoch_number=15):
        # Set an EPOCH number!
        N_EPOCHS = epoch_number

        best_valid_loss = float("inf")

        # We loop forward on the epoch number
        for epoch in range(N_EPOCHS):

            start_time = time.time()

            # Train the model on the training set using the dataloader
            train_loss, train_prec, train_rec, train_fscore = self.train(train_iterator)
            # And validate your model on the validation set
            valid_loss, valid_prec, valid_rec, valid_fscore = self.evaluate(valid_iterator)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # If we find a better model, we save the weights so later we may want to reload it
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss

            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(
                f"\tTrain Loss: {train_loss:.3f} | Train Prec: {train_prec * 100:.2f}% | Train Rec: {train_rec * 100:.2f}% | Train Fscore: {train_fscore * 100:.2f}%"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f} |  Val Prec: {valid_prec * 100:.2f}% | Val Rec: {valid_rec * 100:.2f}% | Val Fscore: {valid_fscore * 100:.2f}%"
            )

        return best_valid_loss

    def predict(self, iterator):
        # On the validation dataset we don't want training so we need to set the model on evaluation mode
        self.model.eval()
        # Also tell Pytorch to not propagate any error backwards in the model or calculate gradients
        # This is needed when you only want to make predictions and use your model in inference mode!
        predictions = []
        labels = []
        with torch.no_grad():
            # The remaining part is the same with the difference of not using the optimizer to backpropagation
            for batch in iterator:
                text_vecs = batch[0]
                labels = labels + batch[1].tolist()

                prediction = self.model(text_vecs).reshape(1,-1).squeeze().round().tolist()
                # prediction = torch.argmax(self.model(text_vecs), dim=1).tolist()
                predictions = predictions + prediction

        return pd.DataFrame(zip(predictions, labels), columns=['predictions', 'labels'])


class TrainerSklearn:
    def __init__(
            self,
            dataset: NeuroToxicDatasetSklearn,
            model: RandomForestClassifier,
    ):
        self.dataset = dataset
        self.model = model

    def calculate_performance(self, preds, y):
        """
        Returns precision, recall, fscore per batch
        """
        precision, recall, fscore, support = precision_recall_fscore_support(preds, y)

        return precision[1], recall[1], fscore[1]

    def train(self, X_train, y_train):
        # We will calculate loss and accuracy epoch-wise based on average batch accuracy
        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0

        # You always need to set your model to training mode
        # If you don't set your model to training mode the error won't propagate back to the weights
        self.model.train(X_train, y_train)

        # This runs the forward function on your model (you don't need to call it directly)
        predictions = self.model.predict(X_train)

        # Calculate the loss and the accuracy on the predictions (the predictions are log probabilities, remember!)

        prec, recall, fscore = self.calculate_performance(predictions, y_train)

        # Propagate the error back on the model (this means changing the initial weights in your model)
        # Calculate gradients on parameters that requries grad

        # We add batch-wise loss to the epoch-wise loss
        # epoch_loss = loss.item()
        # We also do the same with the scores
        epoch_prec = prec.item()
        epoch_recall = recall.item()
        epoch_fscore = fscore.item()
        return (
            epoch_loss,
            epoch_prec,
            epoch_recall,
            epoch_fscore,
        )

    # The evaluation is done on the validation dataset
    def evaluate(self, X_val, y_val):
        epoch_loss = 0
        epoch_prec = 0
        epoch_recall = 0
        epoch_fscore = 0

        predictions = self.model.predict(X_val)
        # loss = self.criterion(predictions, labels)

        prec, recall, fscore = self.calculate_performance(predictions, y_val)

        # epoch_loss = loss.item()
        epoch_prec = prec.item()
        epoch_recall = recall.item()
        epoch_fscore = fscore.item()

        # Return averaged loss on the whole epoch!
        return (
            epoch_loss,
            epoch_prec,
            epoch_recall,
            epoch_fscore,
        )

    def training_loop(self, X_train, y_train, X_val, y_val):
        best_valid_loss = float("inf")
        start_time = time.time()

        # Train the model on the training set using the dataloader
        train_loss, train_prec, train_rec, train_fscore = self.train(X_train, y_train)
        # And validate your model on the validation set
        valid_loss, valid_prec, valid_rec, valid_fscore = self.evaluate(X_val, y_val)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # If we find a better model, we save the weights so later we may want to reload it

        print(f"Epoch: {1 + 1:02} | Epoch Time: {1}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {train_loss:.3f} | Train Prec: {train_prec * 100:.2f}% | Train Rec: {train_rec * 100:.2f}% | Train Fscore: {train_fscore * 100:.2f}%"
        )
        print(
            f"\t Val. Loss: {valid_loss:.3f} |  Val Prec: {valid_prec * 100:.2f}% | Val Rec: {valid_rec * 100:.2f}% | Val Fscore: {valid_fscore * 100:.2f}%"
        )
        best_valid_loss = valid_loss
        return best_valid_loss

    def predict(self, X_test, y_test):
        # On the validation dataset we don't want training so we need to set the model on evaluation mode
        predictions = []
        labels = y_test
        # The remaining part is the same with the difference of not using the optimizer to backpropagation
        predictions = self.model.predict(X_test)

        return pd.DataFrame(zip(predictions, labels), columns=['predictions', 'labels'])


class RandomForestModel:
    def __init__(self):
        # Initialize CountVectorizer for text vectorization
        self.vectorizer = CountVectorizer()
        # Initialize RandomForestClassifier
        self.model = RandomForestClassifier()

    def train(self, X, y):
        # Vectorize training data
        X_vec = self.vectorizer.fit_transform(X)
        # Train the RandomForestClassifier
        self.model.fit(X_vec, y)

    def predict(self, X):
        # Vectorize input data
        X_vec = self.vectorizer.transform(X)
        # Make predictions
        predictions = self.model.predict(X_vec)
        return predictions

    def evaluate(self, X, y):
        # Vectorize input data
        X_vec = self.vectorizer.transform(X)
        # Make predictions
        predictions = self.model.predict(X_vec)
        # Evaluate precision, recall, and F-score
        precision, recall, fscore, _ = precision_recall_fscore_support(y, predictions, average='binary')
        return precision, recall, fscore

    def save_model(self, filename):
        # No need to save the RandomForestModel as training is performed every run
        pass

    def load_model(self, filename):
        # No need to load the RandomForestModel as training is performed every run
        pass
