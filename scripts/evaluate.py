import argparse

from nlptoxicity.utils import (
    calculate_tp_fp_fn,
)




def print_evaluation(predictions):
    y_true = predictions['labels'].tolist()
    y_pred = predictions['predictions'].tolist()
    tp, fp, fn, precision, recall, fscore = calculate_tp_fp_fn(y_true, y_pred)
    print("Statistics:")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {fscore}")


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-t", "--test-data", type=str, required=True, help="Path to test data")
    parser.add_argument("-mp", "--model-path", type=str, help="Path of trained model")
    parser.add_argument("-ml", "--ml-type", type=str, choices=['NN', 'RF'],
                        help="Model to use")

    return parser.parse_args()


if "__main__" == __name__:
    args = get_args()

    test_data = args.test_data
    model_path = args.model_path
    mltype = args.ml_type

