import stanza
import pickle
import argparse
import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nlptoxicity.utils import decode_binary_comments, read_docs_from_json, \
    remove_duplicate_comments, write_docs_to_conllu, read_docs_from_conllu, convert_stanza_doc_to_list_of_words_cleaned
from random import shuffle
from math import floor, ceil


def do_data_preparation(data_path: str, conllu_path: str):
    stanza.download('de')
    df = read_docs_from_json(data_path)
    cleaned_df = remove_duplicate_comments(df)
    cleaned_df = decode_binary_comments(cleaned_df)
    nlp = stanza.Pipeline('de', processors='tokenize,mwt,lemma,pos,sentiment')
    stanza_docs = [nlp(comment) for comment in cleaned_df['Comment']]
    write_docs_to_conllu(stanza_docs, conllu_path, cleaned_df)


def create_train_test_val(len, train_ratio, test_ratio):
    assert train_ratio + test_ratio < 1, "train_ratio and test_ratio have to sum to less than 1."
    indices = list(range(len))
    shuffle(indices)
    train_lim = floor(len * train_ratio)
    train_indices = indices[:train_lim]
    test_lim = train_lim + ceil(len * test_ratio)
    test_indices = indices[train_lim:test_lim]
    val_indices = indices[test_lim:]
    return train_indices, test_indices, val_indices


def do_further_preparation(conllu_path: str):
    nltk.download("stopwords")
    stopwords = nltk.corpus.stopwords.words('german')
    doclist, infolist = read_docs_from_conllu(conllu_path)
    labels_all = np.array([x["label"] for x in infolist])

    # using dicts instead of simple lists to preserve indices in case of dropped documents due to pruning
    corpus_lists = {}
    corpus_full = {}
    all_wordcounts = {}
    singleton_candidates = set()
    ordinal_pattern = "\d+\."  # matches ordinal numbers like "1.", "77." etc that the pipeline did not identify as numerical
    for n, doc in enumerate(doclist):
        wordlist = convert_stanza_doc_to_list_of_words_cleaned(doc)  # removes punctuation and numericals
        cleaned_wordlist = []
        for word in wordlist:
            if word not in stopwords and not re.match(ordinal_pattern, word):
                cleaned_wordlist.append(word)
                try:
                    all_wordcounts[word] += 1
                    singleton_candidates.discard(
                        word)  # discard doesn't raise an error if the element is not present, unlike remove
                except KeyError:
                    # 99% sure just trying for the key is quicker than checking if it exists first
                    all_wordcounts[word] = 1
                    singleton_candidates.add(word)
        if len(cleaned_wordlist) > 0:
            corpus_lists[n] = cleaned_wordlist
            corpus_full[n] = " ".join(cleaned_wordlist)

    corpus_desingletonified = {}
    for n, corpus_element in corpus_lists.items():
        new_element = [x for x in corpus_element if x not in singleton_candidates]
        if len(new_element) > 0:
            corpus_desingletonified[n] = " ".join(new_element)

    vectorizer_full = CountVectorizer().fit(corpus_full.values())
    matrix_full = vectorizer_full.transform(corpus_full.values())
    vectorizer_pruned = CountVectorizer().fit(corpus_desingletonified.values())
    matrix_pruned = vectorizer_pruned.transform(corpus_desingletonified.values())

    train_full, test_full, val_full = create_train_test_val(len(corpus_full), 0.6, 0.2)
    train_pruned, test_pruned, val_pruned = create_train_test_val(len(corpus_desingletonified), 0.6, 0.2)

    splits = {
        "full": {"train": train_full, "test": test_full, "val": val_full},
        "pruned": {"train": train_pruned, "test": test_pruned, "val": val_pruned}
    }

    pickle.dump(splits, open("../data/splits.pkl", "wb"))

    dataset_full = {
        "train": {"features": matrix_full[train_full], "labels": labels_all[train_full]},
        "test": {"features": matrix_full[test_full], "labels": labels_all[test_full]},
        "val": {"features": matrix_full[val_full], "labels": labels_all[val_full]}
    }
    pickle.dump(dataset_full, open("../data/dataset_full.pkl", "wb"))

    dataset_pruned = {
        "train": {"features": matrix_pruned[train_pruned], "labels": labels_all[train_pruned]},
        "test": {"features": matrix_pruned[test_pruned], "labels": labels_all[test_pruned]},
        "val": {"features": matrix_pruned[val_pruned], "labels": labels_all[val_pruned]}
    }
    pickle.dump(dataset_pruned, open("../data/dataset_pruned.pkl", "wb"))


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-d", "--data", type=str, required=False, default='../data/data_all.json', help='Path to data')
    parser.add_argument("-c", "--conllu", type=str, required=False, default='../data/output_full.conllu',
                        help="Path to save conllu data")
    return parser.parse_args()


if "__main__" == __name__:
    args = get_args()
    do_data_preparation(args.data, args.conllu)
    do_further_preparation(args.conllu)
