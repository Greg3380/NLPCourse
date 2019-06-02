import io
import os
import random
from os.path import join
from pathlib import Path

import fastText
import pandas as pd
import regex
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, DocumentLSTMEmbeddings, FlairEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support


def get_files_to_be_processed(directory_path):
    files_to_be_processed = []
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            files_to_be_processed.append(os.path.join(directory_path, filename))
    return files_to_be_processed


def categorize_bill_and_remove_begining(filename, file_content, amended, not_amended):
    try:
        pattern = r'U\s*S\s*T\s*A\s*W\s*A[\s\d\p{L}\p{P}]+?(?=Rozdział|Art)'
        result = regex.findall(pattern, file_content, regex.MULTILINE | regex.IGNORECASE | regex.DOTALL)
        if (r'o zmianie ustawy' in result[0]):
            amended.append((filename, file_content.replace(result[0], ''), 'amended'))
        else:
            not_amended.append((filename, file_content.replace(result[0], ''), 'not_amended'))
    except:
        print("Error while removing header in bill:" + filename)


def split_documents_into_gropus(bill_list):
    training = bill_list[int(len(bill_list) * 0): int(len(bill_list) * .6)]
    validation = bill_list[int(len(bill_list) * .6): int(len(bill_list) * .8)]
    testing = bill_list[int(len(bill_list) * .8): int(len(bill_list))]
    return training, validation, testing


def random_ten_percent_lines(file):
    file_split = file.split('\n')
    lines_count = len(file_split)
    return "\n".join([random.choice(file_split) for i in range(0, int(lines_count * 0.1))])


def top_n_lines(file, n):
    file_split = file.split('\n')
    return "\n".join(file_split[:n])


def random_ten_lines(file):
    file_split = file.split('\n')
    return "\n".join([random.choice(file_split) for i in range(0, 10)])


def random_one_lines(file):
    file_split = file.split('\n')
    return "\n".join([random.choice(file_split)])


def create_classification():
    grid_search_tune = create_binary_tf_idf_classifier()


def create_binary_tf_idf_classifier() -> GridSearchCV:
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])
    parameters = {
        'tfidf__max_df': ([1 / 100] + list(x / 100 for x in range(5, 101, 5))),
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__class_weight": ['balanced', None]
    }
    return GridSearchCV(pipeline, parameters, n_jobs=4)


def train_svm(training_set, testing_set, filename, varint):
    df_training = pd.DataFrame(
        training_set,
        columns=['file_name', 'bill_content', 'is_amendment'])

    data_x = df_training[['bill_content']].as_matrix()
    data_y = df_training[['is_amendment']].as_matrix()

    train_x = [x[0].strip() for x in data_x.tolist()]
    train_y = [x[0].strip() for x in data_y.tolist()]

    grid_search_tune = create_binary_tf_idf_classifier()
    grid_search_tune.fit(train_x, train_y)

    df_testing = pd.DataFrame(
        testing_set,
        columns=['file_name', 'bill_content', 'is_amendment'])

    data_x = df_testing[['bill_content']].as_matrix()
    data_y = df_testing[['is_amendment']].as_matrix()

    test_x = [x[0].strip() for x in data_x.tolist()]
    test_y = [x[0].strip() for x in data_y.tolist()]

    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)
    report = classification_report(test_y, predictions)

    with open(os.path.join("output", filename), 'w+', encoding="utf8") as fout:
        fout.write('Classification results for variant:\t{}\n'.format(str(varint)))
        fout.write(report)


def train_flair(dir_name):
    corpus = NLPTaskDataFetcher.load_classification_corpus(os.path.join("flair", dir_name), test_file='test.csv',
                                                           dev_file='validation.csv', train_file='train.csv')
    word_embeddings = [WordEmbeddings('pl'), FlairEmbeddings('polish-forward'),
                       FlairEmbeddings('polish-backward')]
    document_embeddings = DocumentLSTMEmbeddings(word_embeddings, hidden_size=512, reproject_words=True,
                                                 reproject_words_dimension=256)
    classifier = TextClassifier(document_embeddings, label_dictionary=corpus.make_label_dictionary(), multi_label=False)
    trainer = ModelTrainer(classifier, corpus)
    trainer.train(os.path.join("flair", dir_name), max_epochs=10)


def convert_set_to_csv(output_filename, output_dir, data_set):
    import csv
    prepared_set = [("__label__" + x[2], x[1].replace("\n", " ")) for x in data_set]
    with open(os.path.join("flair", output_dir, output_filename), 'w+') as out_file:
        writer = csv.writer(out_file)
        writer.writerow(('isAmended', 'billContent'))
        writer.writerows(prepared_set)


# def prepare_document_for_fasttext(bill_set, filename, dir_name):
#     with open(os.path.join("fasttext", dir_name, filename), 'w+', encoding="utf8") as fout:
#         for x in bill_set:
#             fout.write("__label__" + x[2] + " " + x[1].replace("\n", " ") + "\n")


# def prepare_data_for_fasttext(training_set, validation_set, testing_set, dir_name):
#     prepare_document_for_fasttext(training_set, "set.train", dir_name)
#     prepare_document_for_fasttext(validation_set, "set.validation", dir_name)
#     prepare_document_for_fasttext(testing_set, "set.test", dir_name)


def prepare_data_for_flair(training_set, validation_set, testing_set, dir_name):
    convert_set_to_csv("test.csv", dir_name, testing_set)
    convert_set_to_csv("train.csv", dir_name, training_set)
    convert_set_to_csv("validation.csv", dir_name, validation_set)


def prepare_text(text):
    return text.replace('\n', ' ').replace('\r', '').replace("\"", "\"\"")


def csv_row(row):
    label = (int)(row['amended'])
    text = prepare_text(row['text'])
    return '__label__{0}, "{1}"\n'.format(label, text)


def prepare_files(df, path):
    with open(path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            f.write(csv_row(row))


def evaluate(model, test):
    is_amended = [entry[0] for entry in test]
    _, _, f1, _ = precision_recall_fscore_support(
        is_amended,
        fasttext_predict(model, test),
        average='weighted')
    return f1


def teach_fasttext(train_path, validation):
    best_score = None
    best_classifier = None
    metrics = []

    for lr in [i / 10 for i in range(1, 10)]:
        for wordNgrams in [1, 2, 3]:
            model = fastText.train_supervised(
                train_path, lr=lr, wordNgrams=wordNgrams)

            score = evaluate(model, validation)

            if best_score is None or score > best_score:
                best_score = score
                best_classifier = model
            print('Score {0} for lr={1}, wordNgrams={2}'.format(
                score, lr, wordNgrams))

            metrics.append((lr, score, wordNgrams))

    metric_df = pd.DataFrame(metrics, columns=['lr', 'score', 'wordNgrams'])

    return best_classifier, metrics


def extract_label(result):
    if "not_amended" in result[0]:
        return "not_amended"
    return "amended"


def fasttext_predict(model, test):
    labels, _ = model.predict([prepare_text(entry[1]) for entry in test])
    return list(map(extract_label, labels))


def read_from_csv(path):
    import csv
    with open(path, 'r') as f:
        reader = csv.reader(f)
        result = list(reader)
    del result[0]
    print(result)
    return [(extract_label(entry), entry[1]) for entry in result]


def main():
    directory_path = os.path.join("..", "ustawy")
    files = get_files_to_be_processed(directory_path)
    random.shuffle(files)
    amended, not_amended = get_segregated_files(files)
    training_amended, validation_amended, testing_amended = split_documents_into_gropus(amended)
    training_notamended, validation_notamended, testing_notamended = split_documents_into_gropus(not_amended)

    training_set_full = [(entry[0], top_n_lines(entry[1], 50), entry[2]) for entry in
                         training_amended + training_notamended]
    testing_set_full = [(entry[0], top_n_lines(entry[1], 50), entry[2]) for entry in
                        testing_amended + testing_notamended]
    validation_set_full = [(entry[0], top_n_lines(entry[1], 50), entry[2]) for entry in
                           validation_amended + validation_notamended]

    training_set_ten_percent = [(entry[0], random_ten_percent_lines(entry[1]), entry[2]) for entry in
                                training_amended + training_notamended]
    testing_set_ten_percent = [(entry[0], random_ten_percent_lines(entry[1]), entry[2]) for entry in
                               testing_amended + testing_notamended]
    validation_set_ten_percent = [(entry[0], random_ten_percent_lines(entry[1]), entry[2]) for entry in
                                  validation_amended + validation_notamended]

    training_set_ten_lines = [(entry[0], random_ten_lines(entry[1]), entry[2]) for entry in
                              training_amended + training_notamended]
    testing_set_ten_lines = [(entry[0], random_ten_lines(entry[1]), entry[2]) for entry in
                             testing_amended + testing_notamended]
    validation_set_ten_lines = [(entry[0], random_ten_lines(entry[1]), entry[2]) for entry in
                                validation_amended + validation_notamended]

    training_set_one_line = [(entry[0], random_one_lines(entry[1]), entry[2]) for entry in
                             training_amended + training_notamended]
    testing_set_one_line = [(entry[0], random_one_lines(entry[1]), entry[2]) for entry in
                            testing_amended + testing_notamended]
    validation_set_one_line = [(entry[0], random_one_lines(entry[1]), entry[2]) for entry in
                               validation_amended + validation_notamended]

    prepare_data_for_flair(training_set_full, validation_set_full, testing_set_full, "full")
    prepare_data_for_flair(training_set_ten_percent, validation_set_ten_percent, testing_set_ten_percent, "ten_percent")
    prepare_data_for_flair(training_set_ten_lines, validation_set_ten_lines, testing_set_ten_lines, "ten_lines")
    prepare_data_for_flair(training_set_one_line, validation_set_one_line, testing_set_one_line, "one_line")
    train_svm(training_set_full, testing_set_full, "exercise7_svm_1", 1)
    # train_svm(training_set_ten_percent, testing_set_ten_percent, "exercise7_svm_2", 2)
    # train_svm(training_set_ten_lines, testing_set_ten_lines, "exercise7_svm_3", 3)
    # train_svm(training_set_one_line, testing_set_one_line, "exercise7_svm_4", 4)
    train_flair("one_line")
    # train_flair("ten_lines")
    # train_flair("ten_percent")
    # train_flair("full")

    # variants = ["full", "ten_percent", "ten_lines", "one_line"]
    variants = ["one_line"]

    for variant in variants:
        print("--------------" + variant + "--------------")
        validation_list = read_from_csv(os.path.join("flair", variant, "validation.csv"))
        clf, metrics = teach_fasttext("./flair/" + variant + "/train.csv", validation_list)
        fasttext_show_scores(clf, validation_list, variant)


def fasttext_show_scores(clf, test, name):
    is_amended = [entry[0] for entry in test]
    p, r, f1, _ = precision_recall_fscore_support(
        is_amended, fasttext_predict(clf, test), average='weighted')

    with open(os.path.join("output", "ex8_fasttext_" + name), 'w+', encoding="utf8") as fout:
        fout.write('Precision: {0}\n'.format(p))
        fout.write('Recall:    {0}\n'.format(r))
        fout.write('F1 score:  {0}\n'.format(f1))


def get_segregated_files(files):
    amended = []
    not_amended = []
    for file in files:
        with io.open(file, 'r', encoding="utf8") as fin:
            categorize_bill_and_remove_begining(file, fin.read(), amended, not_amended)
    return amended, not_amended


if __name__ == '__main__':
    main()
