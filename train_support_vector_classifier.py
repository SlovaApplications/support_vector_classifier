import json
import math
import pickle
import random
from argparse import ArgumentParser

from gensim.parsing.porter import PorterStemmer
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def balance_training_data(records):
    include = [i for i in records if i['include'] == 1]
    random.shuffle(include)
    exclude = [i for i in records if i['include'] == 0]
    random.shuffle(exclude)

    minimum_len = min([len(include), len(exclude)])
    records = include[:minimum_len] + exclude[:minimum_len]
    random.shuffle(records)
    return records


def split_training_test(records, split_val=.9):
    cutoff = math.floor(split_val * len(records))
    training_data = records[:cutoff]
    validation_data = records[cutoff:]
    return training_data, validation_data


def stem_tokens(records):
    p = PorterStemmer()
    output = []
    for r in records:
        r['text'] = p.stem_sentence(r['text'])
        output.append(r)
    return output


def load_and_prepare_data(input_path):
    training_records = []
    with open(input_path, 'r') as file:
        for line in file:
            training_records.append(json.loads(line))
    balanced_training_data = balance_training_data(training_records)
    stemmed_training_data = stem_tokens(balanced_training_data)
    train, test = split_training_test(stemmed_training_data)
    x_train = [i['text'] for i in train]
    y_train = [i['include'] for i in train]
    x_test = [i['text'] for i in test]
    y_test = [i['include'] for i in test]
    return x_train, y_train, x_test, y_test


def parse_command_line_args():
    parser = ArgumentParser()
    parser.add_argument(
        'training_data_input_path',
        type=str,
        help='the location of the jsonl file containing records for training the classifier'
    )
    parser.add_argument(
        'model_output_path',
        type=str,
        help='the location where the trained model will be saved'
    )
    return parser.parse_args()


def save_model(output_path, classifier):
    with open(output_path, 'wb') as output_file:
        pickle.dump(classifier, output_file)


if __name__ == '__main__':
    args = parse_command_line_args()

    training_x_vars, training_y_vars, validation_x_vars, validation_y_vars = load_and_prepare_data(args.training_data_input_path)

    svm_classifier = Pipeline([
        ('vect', CountVectorizer(min_df=2, ngram_range=(1, 3))),
        ('tfidf', TfidfTransformer()),
        ('clf', LinearSVC(dual=False)),
    ])
    svm_classifier.fit(training_x_vars, training_y_vars)
    svm_predicted = svm_classifier.predict(validation_x_vars)
    print(metrics.classification_report(validation_y_vars, svm_predicted))
    save_model(args.model_output_path, svm_classifier)
