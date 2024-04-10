import argparse
import os

import numpy as np

from nlptoxicity import logger
from nlptoxicity.projectdatasets import NeuroToxicBertDataset
from nlptoxicity.trainer import GBertTrainer, GBertNerTrainer2
from nlptoxicity.utils import load_and_split_data_set, get_bert_sequence_classification_model, \
    get_transformer_model_tokenizer, load_conllu_data_set, clean_and_augment_data, \
    create_folder_with_timestamp_and_random, load_vulgarities_entries, prepare_ner_data_conllu, \
    get_deepset_gbert_base_mode_ner2


def train_bert_neuro_toxic_network(data_path, splits_path=None, save=False, output_path=None, num_epochs=15,
                                   freeze_first_n_layers=None, small_training=False, conllu=False,
                                   clean=False, augment=False, training_only=False, mltype='GBERT'):
    if mltype == 'GBERT':
        logger.info("Load GBERT model...")
        model = get_bert_sequence_classification_model(freeze_first_n_layers=freeze_first_n_layers)
        tokenizer = get_transformer_model_tokenizer()
    elif mltype == 'GTOXBERT':
        logger.info("Load GBERT model...")
        model = get_bert_sequence_classification_model(freeze_first_n_layers=freeze_first_n_layers,
                                                       model_name='ankekat1000/toxic-bert-german')
        tokenizer = get_transformer_model_tokenizer(model_name='ankekat1000/toxic-bert-german')
    else:
        raise Exception("Model has to be selected GBERT or GTOXBERT")

    logger.info("Loading data...")
    if conllu:
        df_train, df_test, df_val = load_conllu_data_set(data_path, splits_path)
    else:
        df_train, df_test, df_val = load_and_split_data_set(data_path, splits_path=splits_path)

    if small_training:
        df_train = df_train.sample(100)
        df_val = df_val.sample(50)
        df_test = df_test.sample(50)

    df_train = clean_and_augment_data(df_train, clean, augment)
    dataset_train = NeuroToxicBertDataset(df_train, tokenizer=tokenizer)

    if not training_only:
        df_test = clean_and_augment_data(df_test, clean, False)
        df_val = clean_and_augment_data(df_val, clean, False)

    dataset_test = NeuroToxicBertDataset(df_test, tokenizer=tokenizer)
    dataset_val = NeuroToxicBertDataset(df_val, tokenizer=tokenizer)

    logger.info("Training...")
    gtrainer = GBertTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset_train, val_dataset=dataset_val,
                            test_dataset=dataset_test, num_epochs=num_epochs, output_path=output_path)

    gtrainer.do_training()

    if save:
        gtrainer.save_model()
    logger.info("Evaluating...")
    gtrainer.do_evaluation(dataset_test)


def train_gbert_ner_neuro_toxic_network_new_try(data_path: str, save: bool = False, output_path: str = None,
                                                num_epochs: int = 15, freeze_first_n_layers: int = None):
    label_ids = list(['O', 'Vul'])
    label2id = {label: id for id, label in enumerate(label_ids)}
    id2label = {id: label for label, id in label2id.items()}
    logger.info("Load GBERT NER model...")
    model = get_deepset_gbert_base_mode_ner2(freeze_first_n_layers=freeze_first_n_layers)
    tokenizer = get_transformer_model_tokenizer()
    logger.info("Loading data...")
    data_vulgarity_df = prepare_ner_data_conllu(load_vulgarities_entries(data_path))
    df_train, df_val, df_test = np.split(data_vulgarity_df.sample(frac=1, random_state=42),
                                         [int(.8 * len(data_vulgarity_df)), int(.9 * len(data_vulgarity_df))])
    logger.info("Training...")
    g_bert_ner_trainer = GBertNerTrainer2(model=model,
                                          tokenizer=tokenizer,
                                          num_epochs=num_epochs,
                                          output_path=output_path,
                                          labels_to_ids=label2id,
                                          train_data=df_train,
                                          val_data=df_val,
                                          test_data=df_test)
    g_bert_ner_trainer.do_training()

    if save:
        g_bert_ner_trainer.save_model()
    logger.info("Evaluating...")
    g_bert_ner_trainer.do_evaluation()


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--train-data", type=str, required=False, help="Path to training data")
    parser.add_argument("-s", "--save", default=False, action="store_true", help="Save model")
    parser.add_argument("-co", "--conllu", default=False, action="store_true", help="Use conllu file")
    parser.add_argument("-st", "--small-training", default=False, action="store_true", help="Train on reduced dataset")
    parser.add_argument("-sp", "--save-path", default="..runs/default", type=str, help="Path to save model")
    parser.add_argument("-ml", "--ml-type", type=str,
                        choices=['GBERT', 'GTOXBERT', 'GBERTNER', 'GBCN', 'GBERTNER2'],
                        help="Model to use")
    parser.add_argument("-e", "--epochs", type=int, default=15, help="Number of epochs to train")
    parser.add_argument("-dp", "--data-path", type=str, help="Path to data")
    parser.add_argument("-spp", "--splits-path", type=str, help="Path to splits file", default=None)
    parser.add_argument("-fl", "--freeze_layers", type=int, default=None,
                        help="Freeze first number of layers including embedding layer")
    parser.add_argument("-cl", "--clean", default=False, action="store_true",
                        help="Perform basic cleaning operations on training data")
    parser.add_argument("-a", "--augment", default=False, action="store_true",
                        help="Perform class balancing on training data")
    parser.add_argument("-to", "--training-only", default=False, action="store_true",
                        help="Data processing on training set only")

    return parser.parse_args()


if "__main__" == __name__:
    args = get_args()

    train_data = args.train_data
    model_save = args.save
    model_save_path = args.save_path
    mltype = args.ml_type
    num_epochs = args.epochs
    data_path = args.data_path
    freeze_first_n_layers = args.freeze_layers
    small_training = args.small_training
    splits_path = args.splits_path
    use_conllu = args.conllu
    do_cleaning = args.clean
    do_augment = args.augment
    training_only = args.training_only

    logger.info("Create run folder...")
    model_save_path = create_folder_with_timestamp_and_random(model_save_path, mltype)
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    with open(os.path.join(model_save_path, "train_call_args.txt"), "w") as f:
        f.write(str(args.__dict__))
    if mltype == 'GBERT':
        train_bert_neuro_toxic_network(data_path, splits_path=splits_path, save=model_save,
                                       output_path=model_save_path,
                                       num_epochs=num_epochs, freeze_first_n_layers=freeze_first_n_layers,
                                       small_training=small_training, conllu=use_conllu, clean=do_cleaning,
                                       augment=do_augment, training_only=training_only, mltype=mltype)
    elif mltype == 'GTOXBERT':
        train_bert_neuro_toxic_network(data_path, splits_path=splits_path, save=model_save,
                                       output_path=model_save_path,
                                       num_epochs=num_epochs, freeze_first_n_layers=freeze_first_n_layers,
                                       small_training=small_training, conllu=use_conllu, clean=do_cleaning,
                                       augment=do_augment, training_only=training_only, mltype=mltype)
    elif mltype == 'GBERTNER2':
        train_gbert_ner_neuro_toxic_network_new_try(data_path, save=model_save, output_path=model_save_path,
                                                    num_epochs=num_epochs, freeze_first_n_layers=freeze_first_n_layers)
