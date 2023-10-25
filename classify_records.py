import json
import pickle
from argparse import ArgumentParser

from gensim.parsing.porter import PorterStemmer
from tqdm import tqdm


def classify_record(token_stemmer, svm_model, record):
    tokenized_text = token_stemmer.stem_sentence(record['text'])
    record['include'] = int(svm_model.predict([tokenized_text])[0])
    return record


def load_records_to_classify(input_path):
    input_records = []
    with open(input_path, 'r') as file:
        for line in file:
            yield json.loads(line)


def load_svm(model_path):
    with open(model_path, 'rb') as svm_file:
        return pickle.load(svm_file)


def parse_command_line_arguments():
    parser = ArgumentParser()
    parser.add_argument(
        'svm_model_path',
        type=str,
        help='the location of the trained classifier model'
    )
    parser.add_argument(
        'input_path',
        type=str,
        help='the location of the jsonl file containing records to be classified'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help='the location where the classifier results will be saved'
    )
    return parser.parse_args()


def write_results_to_jsonl_file(records, output_path):
    with open(output_path, 'w') as output_file:
        for record in records:
            json.dump(record, output_file)
            output_file.write('\n')


if __name__ == '__main__':
    args = parse_command_line_arguments()

    records_to_classify = load_records_to_classify(args.input_path)
    svm = load_svm(args.svm_model_path)
    stemmer = PorterStemmer()

    classified_records = [classify_record(stemmer, svm, record) for record in tqdm(records_to_classify)]

    write_results_to_jsonl_file(classified_records, args.output_path)
