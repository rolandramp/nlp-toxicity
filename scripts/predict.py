import argparse
import os.path

import pandas as pd

from nlptoxicity.projectdatasets import NeuroToxicBertDataset
from nlptoxicity.trainer import GBertTrainer
from nlptoxicity.utils import get_trained_sequence_classification_model, get_transformer_model_tokenizer, \
    load_conllu_data_set, load_and_split_data_set, clean_and_augment_data
from nlptoxicity import logger


def predict_gbert_class(data_path, model_path, splits_path=None, kind='full', conllu=None, clean=False,
                        augment=False, mltype="GBERT") -> pd.DataFrame:
    """
    Method to run predictions on the given data set
    :param data_path: path to the data set
    :param model_path: path to the trained model
    :return: a DataFrame with predictions and true labels
    """
    logger.info("Load Trained GBERT model...")
    model = get_trained_sequence_classification_model(model_path)
    if mltype == 'GERT':
        tokenizer = get_transformer_model_tokenizer()
    elif mltype == 'GTOXBERT':
        tokenizer = get_transformer_model_tokenizer(model_name='ankekat1000/toxic-bert-german')
    logger.info("Loading data...")
    if conllu:
        _, df_test, _ = load_conllu_data_set(data_path, splits_path)
    else:
        _, df_test, _ = load_and_split_data_set(data_path, splits_path=splits_path)

    df_test = clean_and_augment_data(df_test, clean, False)
    dataset_test = NeuroToxicBertDataset(df_test, tokenizer=tokenizer)

    logger.info("Predicting...")
    gtrainer = GBertTrainer(model=model, tokenizer=tokenizer, train_dataset=None, val_dataset=None,
                            test_dataset=dataset_test)
    predictions = gtrainer.do_predictions(df_test)
    return predictions


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--test-data", type=str, required=True, help="Path to test data")
    parser.add_argument("-mp", "--model-path", type=str, help="Path of trained model")
    parser.add_argument("-ml", "--ml-type", type=str, choices=['GBERT', 'GTOXBERT'], help="Model to use")
    parser.add_argument("-vs", "--vocabulary-size", required=False, type=int, help="Vocabulary size")
    parser.add_argument("-prp", "--prediction-result-path", type=str, help="Path to store predictions as csv")
    parser.add_argument("-sd", "--splits-path", type=str, help="Path to splits")
    parser.add_argument("-k", "--kind", type=str, choices=['full', 'pruned'], help="Kind if dataset")
    parser.add_argument("-co", "--conllu", type=str, help="Path to CONLLU file")
    parser.add_argument("-uc", "--use-conllu", default=False, action="store_true", help="Use conllu file")
    parser.add_argument("-cl", "--clean", default=False, action="store_true",
                        help="Perform basic cleaning operations on training data")
    parser.add_argument("-a", "--augment", default=False, action="store_true",
                        help="Perform class balancing on training data")

    return parser.parse_args()


def write_predictions(prediction_result_path, predictions_df, ml_type: str):
    if prediction_result_path:
        if not os.path.exists(prediction_result_path):
            os.makedirs(prediction_result_path)
        predictions_df.to_csv(os.path.join(prediction_result_path, f'predictions_{ml_type}.csv'), sep=',')


if "__main__" == __name__:
    args = get_args()

    test_data = args.test_data
    model_path = args.model_path
    mltype = args.ml_type
    input_dim = args.vocabulary_size
    prediction_result_path = args.prediction_result_path
    splits = args.splits_path
    kind = args.kind
    conllu_path = args.conllu
    use_conllu = args.use_conllu
    do_cleaning = args.clean
    do_augment = args.augment

    with open(os.path.join(prediction_result_path, "prediction_call_args.txt"), "w") as f:
        f.write(str(args.__dict__))

    if mltype == 'GBERT':
        df = predict_gbert_class(test_data, model_path=model_path, splits_path=splits,
                                 conllu=use_conllu, clean=do_cleaning,
                                 augment=do_augment, kind=kind, mltype=mltype)
        write_predictions(prediction_result_path, df, mltype)
    elif mltype == 'GTOXBERT':
        df = predict_gbert_class(test_data, model_path=model_path, splits_path=splits,
                                 conllu=use_conllu, clean=do_cleaning,
                                 augment=do_augment, kind=kind, mltype=mltype)
        write_predictions(prediction_result_path, df, mltype)
