import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_fscore_support
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback
from transformers import Trainer, TrainingArguments

from nlptoxicity import logger


class GBertTrainer:

    def __init__(self, model, tokenizer, train_dataset, val_dataset, test_dataset, num_epochs=2,
                 output_path="runs/default"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_epochs = num_epochs
        self.output_path = output_path
        self.trainer = self.create_trainer()

    def create_trainer(self) -> Trainer:
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_path, "model"),
            logging_dir=os.path.join(self.output_path, "logs"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=float(5e-5),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=200,
            load_best_model_at_end=True
        )
        return Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics_bin,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    @staticmethod
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        return {
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average='macro'),
            "precision": precision_score(labels, preds, average='macro'),
            "recall": recall_score(labels, preds, average='macro'),
        }

    @staticmethod
    def compute_metrics_bin(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
                }

    def do_training(self):
        train_result = self.trainer.train()
        # self.trainer.log_metrics(self, "train", metrics=train_result.metrics)
        # self.trainer.save_metrics(self, "train", metrics=train_result.metrics)

    def do_evaluation(self, val_dataset):
        eval_result = self.trainer.evaluate(eval_dataset=val_dataset)
        # self.trainer.log_metrics(self,"test", metrics=eval_result)
        # self.trainer.save_metrics(self,"test", metrics=eval_result)
        if not os.path.exists(os.path.join(self.output_path, "logs")):
            os.mkdir(os.path.join(self.output_path, "logs"))
        with open(os.path.join(self.output_path, "logs", "eval_results.txt"), "w") as writer:
            logger.info("Writing evaluation to file ")
            for key, value in sorted(eval_result.items()):
                writer.write(f"{key} = {value}\n")

    def do_predictions(self, df_test: pd.DataFrame) -> pd.DataFrame:
        pred, _, _ = self.trainer.predict(test_dataset=self.test_dataset)
        predictions = np.argmax(pred, axis=1)
        return_df = df_test.copy()
        return_df['prediction'] = predictions
        return return_df

    def save_model(self):
        self.trainer.save_model(os.path.join(self.output_path, "model"))


class GBertNerTrainer:

    def __init__(self, model, tokenizer, train_dataset, val_dataset, test_dataset, num_epochs=2,
                 output_path="runs/default"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.num_epochs = num_epochs
        self.output_path = output_path
        self.trainer = self.create_trainer()

    def create_trainer(self) -> Trainer:
        training_args = TrainingArguments(
            output_dir=os.path.join(self.output_path, "model"),
            logging_dir=os.path.join(self.output_path, "logs"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            learning_rate=float(5e-5),
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=200,
            load_best_model_at_end=True
        )
        return Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=self.compute_metrics_bin,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

    @staticmethod
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        preds = np.argmax(logits, axis=-1)
        return {
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, average='macro'),
            "precision": precision_score(labels, preds, average='macro'),
            "recall": recall_score(labels, preds, average='macro'),
        }

    @staticmethod
    def compute_metrics_bin(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc,
                "f1": f1,
                "precision": precision,
                "recall": recall
                }

    def do_training(self):
        train_result = self.trainer.train()
        # self.trainer.log_metrics(self, "train", metrics=train_result.metrics)
        # self.trainer.save_metrics(self, "train", metrics=train_result.metrics)

    def do_evaluation(self, val_dataset):
        eval_result = self.trainer.evaluate(eval_dataset=val_dataset)
        # self.trainer.log_metrics(self,"test", metrics=eval_result)
        # self.trainer.save_metrics(self,"test", metrics=eval_result)
        if not os.path.exists(os.path.join(self.output_path, "logs")):
            os.mkdir(os.path.join(self.output_path, "logs"))
        with open(os.path.join(self.output_path, "logs", "eval_results.txt"), "w") as writer:
            logger.info("Writing evaluation to file ")
            for key, value in sorted(eval_result.items()):
                writer.write(f"{key} = {value}\n")

    def do_predictions(self, df_test: pd.DataFrame) -> pd.DataFrame:
        pred, _, _ = self.trainer.predict(test_dataset=self.test_dataset)
        predictions = np.argmax(pred, axis=1)
        return_df = df_test.copy()
        return_df['prediction'] = predictions
        return return_df

    def save_model(self):
        self.trainer.save_model(os.path.join(self.output_path, "model"))


class DataSequence(torch.utils.data.Dataset):
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


class GBertNerTrainer2:

    def __init__(self, model, tokenizer, output_path, num_epochs, train_data, test_data, val_data, labels_to_ids,
                 batch_size=8):
        self.model = model
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.learning_rate = 5e-3
        self.epochs = num_epochs
        self.batch_size = batch_size
        self.label_to_ids = labels_to_ids
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

    def do_training(self):
        train_dataset = DataSequence(self.train_data, self.tokenizer, self.label_to_ids)
        val_dataset = DataSequence(self.val_data, self.tokenizer, self.label_to_ids)

        train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=self.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, num_workers=1, batch_size=self.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizer = SGD(self.model.parameters(), lr=self.learning_rate)

        self.model.to(device)

        best_acc = 0
        best_loss = 1000

        for epoch_num in range(self.epochs):

            total_acc_train = 0
            total_loss_train = 0

            self.model.train()

            for train_data, train_label in tqdm(train_dataloader):

                train_label = train_label.to(device)
                mask = train_data['attention_mask'].squeeze(1).to(device)
                input_id = train_data['input_ids'].squeeze(1).to(device)

                optimizer.zero_grad()
                loss, logits = self.model(input_id, mask, train_label)

                for i in range(logits.shape[0]):
                    logits_clean = logits[i][train_label[i] != -100]
                    label_clean = train_label[i][train_label[i] != -100]

                    predictions = logits_clean.argmax(dim=1)
                    acc = (predictions == label_clean).float().mean()
                    total_acc_train += acc
                    total_loss_train += loss.item()

                loss.backward()
                optimizer.step()

            self.model.eval()

            total_acc_val = 0
            total_loss_val = 0

            for val_data, val_label in val_dataloader:

                val_label = val_label.to(device)
                mask = val_data['attention_mask'].squeeze(1).to(device)
                input_id = val_data['input_ids'].squeeze(1).to(device)

                loss, logits = self.model(input_id, mask, val_label)

                for i in range(logits.shape[0]):
                    logits_clean = logits[i][val_label[i] != -100]
                    label_clean = val_label[i][val_label[i] != -100]

                    predictions = logits_clean.argmax(dim=1)
                    acc = (predictions == label_clean).float().mean()
                    total_acc_val += acc
                    total_loss_val += loss.item()

            val_accuracy = total_acc_val / len(self.val_data)
            val_loss = total_loss_val / len(self.val_data)

            print(
                f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(self.train_data): .3f} | Accuracy: {total_acc_train / len(self.train_data): .3f} | Val_Loss: {total_loss_val / len(self.val_data): .3f} | Accuracy: {total_acc_val / len(self.val_data): .3f}')

    def do_evaluation(self):
        test_dataset = DataSequence(self.test_data, self.tokenizer, self.label_to_ids)
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

            loss, logits = self.model(input_id, mask, test_label)

            for i in range(logits.shape[0]):
                logits_clean = logits[i][test_label[i] != -100]
                label_clean = test_label[i][test_label[i] != -100]

                predictions = logits_clean.argmax(dim=1)
                precision, recall, fscore, support = precision_recall_fscore_support(predictions.cpu(),
                                                                                     label_clean.cpu(),
                                                                                     zero_division=0.0)
                acc = (predictions == label_clean).float().mean()
                total_acc_test += acc
                total_precicion_test += precision[1] if len(precision) == 2 else 0
                total_recall_test += recall[1] if len(recall) == 2 else 0
                total_fscore_test += fscore[1] if len(fscore) == 2 else 0

        val_accuracy = total_acc_test / len(self.test_data)
        print(
            f'Test Accuracy: {total_acc_test / len(self.test_data): .3f} | Test Precision : {total_precicion_test / len(self.test_data): .3f} | Test Recall: {total_recall_test / len(self.test_data): .3f} | Test F1Score: {total_fscore_test / len(self.test_data): .3f}')

    def save_model(self):
        Trainer(self.model).save_model(os.path.join(self.output_path, "model"))
