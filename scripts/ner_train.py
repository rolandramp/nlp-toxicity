import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from nlptoxicity.utils import create_folder_with_timestamp_and_random, get_deepset_gbert_base_mode_ner2, \
    get_transformer_model_tokenizer

LEARNING_RATE = 5e-3
EPOCHS = 10
BATCH_SIZE = 8


def align_label(texts, labels, tokenizer):
    """
    Method to align each token (word part) to its corresponding label

    """
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=100, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            # -100 is a special token to mark parts that should not be included in the error calculation
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


class DataSequence(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        lb = [i.split() for i in data['ner_label'].values.tolist()]
        txt = data['Comment'].values.tolist()
        self.texts = [tokenizer(str(i),
                                padding='max_length', max_length=100, truncation=True, return_tensors="pt") for i in
                      txt]
        self.labels = [align_label(i, j, tokenizer) for i, j in zip(txt, lb)]

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


def train_loop(model, df_train, df_val, tokenizer):
    train_dataset = DataSequence(df_train, tokenizer)
    val_dataset = DataSequence(df_val, tokenizer)

    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=1, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)

    model.to(device)

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(EPOCHS):

        total_acc_train = 0
        total_loss_train = 0
        total_precicion_train = 0
        total_recall_train = 0
        total_fscore_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][train_label[i] != -100]
                label_clean = train_label[i][train_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                precision, recall, fscore, support = precision_recall_fscore_support(predictions.cpu(),
                                                                                     label_clean.cpu(),
                                                                                     zero_division=0.0)

                acc = (predictions == label_clean).float().mean()
                total_acc_train += acc
                total_loss_train += loss.item()
                total_precicion_train += precision[1] if len(precision) == 2 else 0
                total_recall_train += recall[1] if len(recall) == 2 else 0
                total_fscore_train += fscore[1] if len(fscore) == 2 else 0

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0
        total_precicion_val = 0
        total_recall_val = 0
        total_fscore_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][val_label[i] != -100]
                label_clean = val_label[i][val_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                acc = (predictions == label_clean).float().mean()
                precision, recall, fscore, support = precision_recall_fscore_support(predictions.cpu(),
                                                                                     label_clean.cpu(),
                                                                                     zero_division=0.0)
                total_acc_val += acc
                total_loss_val += loss.item()
                total_precicion_val += precision[1] if len(precision) == 2 else 0
                total_recall_val += recall[1] if len(recall) == 2 else 0
                total_fscore_val += fscore[1] if len(fscore) == 2 else 0

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(
            f'Epochs train: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Precision  {total_precicion_train / len(df_train): .3f} | Recall: {total_recall_train / len(df_train): .3f} | F1Score: {total_fscore_train / len(df_train): .3f}')
        print(
            f'Epochs val: {epoch_num + 1} | Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f} | Precision  {total_precicion_val / len(df_val): .3f} | Recall: {total_recall_val / len(df_val): .3f} | F1Score: {total_fscore_val / len(df_val): .3f}')


def evaluate(model, df_test, tokenizer):
    test_dataset = DataSequence(df_test, tokenizer)
    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    total_acc_test = 0.0
    total_precicion_test = 0
    total_recall_test = 0
    total_fscore_test = 0

    for test_data, test_label in test_dataloader:

        test_label = test_label.to(device)
        mask = test_data['attention_mask'].squeeze(1).to(device)
        input_id = test_data['input_ids'].squeeze(1).to(device)

        loss, logits = model(input_id, mask, test_label)

        for i in range(logits.shape[0]):
            logits_clean = logits[i][test_label[i] != -100]
            label_clean = test_label[i][test_label[i] != -100]

            predictions = logits_clean.argmax(dim=1)
            precision, recall, fscore, support = precision_recall_fscore_support(predictions.cpu(), label_clean.cpu(),
                                                                                 zero_division=0.0)
            acc = (predictions == label_clean).float().mean()
            total_acc_test += acc
            total_precicion_test += precision[1] if len(precision) == 2 else 0
            total_recall_test += recall[1] if len(recall) == 2 else 0
            total_fscore_test += fscore[1] if len(fscore) == 2 else 0

    val_accuracy = total_acc_test / len(df_test)
    print(
        f'Test Accuracy: {total_acc_test / len(df_test): .3f} | Test Precision : {total_precicion_test / len(df_test): .3f} | Test Recall: {total_recall_test / len(df_test): .3f} | Test F1Score: {total_fscore_test / len(df_test): .3f}')


def align_word_ids(texts):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=100, truncation=True)
    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(1)
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(1 if True else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids


def evaluate_one_text(model, sentence, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = model.to(device)

    text = tokenizer(sentence, padding='max_length', max_length=100, truncation=True, return_tensors="pt")

    mask = text['attention_mask'].to(device)
    input_id = text['input_ids'].to(device)
    label_ids = torch.Tensor(align_word_ids(sentence)).unsqueeze(0).to(device)

    logits = model(input_id, mask, None)
    logits_clean = logits[0][label_ids != -100]

    predictions = logits_clean.argmax(dim=1).tolist()
    prediction_label = [ids_to_labels[i] for i in predictions]
    sentence = sentence.split()

    print(f'sentence: {sentence} ner prediction {prediction_label}')


if "__main__" == __name__:

    model_save_path = create_folder_with_timestamp_and_random('../runs/', 'GBERTNER2')

    data = pd.read_csv('../data/ner_prep.csv')
    data['ner_label'] = data['ner_label'].str.replace('[', '').str.replace(']', '').str.replace("'", '').str.replace(
        ',', '')
    data = data[data["ner_label"].str.contains("Vul")]

    labels = [i.split() for i in data['ner_label'].values.tolist()]

    # Check how many labels are there in the dataset
    unique_labels = set()

    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]

    # Map each label into its id representation and vice versa
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

    df_train, df_val, df_test = np.split(data.sample(frac=1, random_state=42),
                                         [int(.8 * len(data)), int(.9 * len(data))])
    tokenizer = get_transformer_model_tokenizer()
    model = get_deepset_gbert_base_mode_ner2(freeze_first_n_layers=6)
    train_loop(model, df_train, df_val, tokenizer)

    evaluate(model, df_test, tokenizer)

    test_sentences = pd.read_csv('../data/ner_sample_sentences.csv')['Comment'].values.tolist()
    test_ner_labels = pd.read_csv('../data/ner_sample_sentences.csv')['ner_label'].values.tolist()

    for i, sentence in enumerate(test_sentences):
        print(f'Sentence: {sentence} true ner labels: {test_ner_labels[i]}')
        evaluate_one_text(model, sentence, tokenizer)
