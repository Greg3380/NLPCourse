import io
import os
import random
from os.path import join

import pandas as pd
import regex
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


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
        if(r'o zmianie ustawy' in result[0]):
            amended.append((filename, file_content.replace(result[0], ''), 'amended'))
        else:
            not_amended.append((filename, file_content.replace(result[0], ''), 'not_amended'))
    except:
        print("Error while removing header in bill:" + filename)

def split_documents_into_gropus(bill_list):
    training = bill_list[int(len(bill_list) * 0) : int(len(bill_list) * .6)]
    validation = bill_list[int(len(bill_list) * .6) : int(len(bill_list) * .8)]
    testing = bill_list[int(len(bill_list) * .8): int(len(bill_list))]
    return  training, validation, testing

def random_ten_percent_lines(file):
    file_split = file.split('\n')
    lines_count = len(file_split)
    return "\n".join([random.choice(file_split) for i in range(0,int(lines_count * 0.1))])

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


def train_svm(training_amended, testing_amended, training_notamended, testing_notamended, filename, varint):
    df_training = pd.DataFrame(
        training_amended + training_notamended,
        columns=['file_name', 'bill_content', 'is_amendment'])

    data_x = df_training[['bill_content']].as_matrix()
    data_y = df_training[['is_amendment']].as_matrix()

    train_x = [x[0].strip() for x in data_x.tolist()]
    train_y = [x[0].strip() for x in data_y.tolist()]

    grid_search_tune = create_binary_tf_idf_classifier()
    grid_search_tune.fit(train_x, train_y)

    df_testing = pd.DataFrame(
        testing_amended + testing_notamended,
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


def train_flair(training_amended, validation_amended, testing_amended, training_notamended, validation_notamended,
                testing_notamended, filename, variant):
    pass


def prepare_document_for_fasttext(bill_set, filename, dir_name):
    with open(os.path.join("fasttext", dir_name, filename), 'w+', encoding="utf8") as fout:
        for x in bill_set:
            fout.write("__label__" + x[2] + " " + x[1] + "\n")



def prepare_data_for_fasttext(training_set, validation_set, testing_set, dir_name):

    prepare_document_for_fasttext(training_set, "set.train", dir_name)
    prepare_document_for_fasttext(validation_set, "set.validation", dir_name)
    prepare_document_for_fasttext(testing_set, "set.test", dir_name)



def main():
    directory_path = os.path.join("..", "ustawy")
    files = get_files_to_be_processed(directory_path)
    random.shuffle(files)
    amended, not_amended = get_segregated_files(files)
    training_amended, validation_amended, testing_amended = split_documents_into_gropus(amended)
    training_notamended, validation_notamended, testing_notamended = split_documents_into_gropus(not_amended)
    prepare_data_for_fasttext(training_amended + training_notamended, validation_amended + validation_notamended,
                              testing_amended + testing_notamended, "full")
    prepare_data_for_fasttext([(x[0], random_ten_percent_lines(x[1]), x[2]) for x in training_amended + training_notamended],
                              [(x[0], random_ten_percent_lines(x[1]), x[2]) for x in validation_amended + validation_notamended],
                              [(x[0], random_ten_percent_lines(x[1]), x[2]) for x in testing_amended + testing_notamended],
                              "ten_percent")
    prepare_data_for_fasttext([(x[0], random_ten_lines(x[1]), x[2]) for x in training_amended + training_notamended],
                              [(x[0], random_ten_lines(x[1]), x[2]) for x in validation_amended + validation_notamended],
                              [(x[0], random_ten_lines(x[1]), x[2]) for x in testing_amended + testing_notamended],
                              "ten_lines")
    prepare_data_for_fasttext([(x[0], random_one_lines(x[1]), x[2]) for x in training_amended + training_notamended],
                              [(x[0], random_one_lines(x[1]), x[2]) for x in validation_amended + validation_notamended],
                              [(x[0], random_one_lines(x[1]), x[2]) for x in testing_amended + testing_notamended],
                              "one_line")

    train_svm(training_amended, testing_amended, training_notamended, testing_notamended, "exercise7_svm_variant1", 1)
    train_svm([(x[0], random_ten_percent_lines(x[1]), x[2]) for x in training_amended],
              [(x[0], random_ten_percent_lines(x[1]), x[2]) for x in testing_amended],
              [(x[0], random_ten_percent_lines(x[1]), x[2]) for x in training_notamended],
              [(x[0], random_ten_percent_lines(x[1]), x[2]) for x in testing_notamended], "exercise7_svm_variant2", 2)
    train_svm([(x[0], random_ten_lines(x[1]), x[2]) for x in training_amended],
              [(x[0], random_ten_lines(x[1]), x[2]) for x in testing_amended],
              [(x[0], random_ten_lines(x[1]), x[2]) for x in training_notamended],
              [(x[0], random_ten_lines(x[1]), x[2]) for x in testing_notamended], "exercise7_svm_variant3", 3)
    train_svm([(x[0], random_one_lines(x[1]), x[2]) for x in training_amended],
              [(x[0], random_one_lines(x[1]), x[2]) for x in testing_amended],
              [(x[0], random_one_lines(x[1]), x[2]) for x in training_notamended],
              [(x[0], random_one_lines(x[1]), x[2]) for x in testing_notamended], "exercise7_svm_variant4", 4)


    # train_flair(training_amended, validation_amended, testing_amended, training_notamended, validation_notamended, testing_notamended, "exercise7_flair_variant1", 1)

def get_segregated_files(files):
    amended = []
    not_amended = []
    for file in files:
        with io.open(file, 'r', encoding="utf8") as fin:
            categorize_bill_and_remove_begining(file, fin.read(), amended, not_amended)
    return amended, not_amended


if __name__ == '__main__':
    main()