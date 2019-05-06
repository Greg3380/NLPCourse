# -*- coding: utf-8 -*-
import random
from os.path import join

import numpy
from gensim.models import KeyedVectors
from matplotlib import pyplot
from numpy.random import shuffle
from sklearn.manifold import TSNE


def main():
    model = KeyedVectors.load_word2vec_format("/Users/grzegorz/Downloads/skipgram/skip_gram_v100m8.w2v.txt",
                                              binary=False)
    results = find_n_most_similar(model, 3,
                                  ['Sąd Najwyższy', 'Trybunał Konstytucyjny', 'kodeks cywilny', 'kpk', 'sąd rejonowy',
                                   'szkoda', 'wypadek', 'kolizja', 'szkoda majątkowa', 'nieszczęście', 'rozwód'])
    save_data("\n".join(results), 'exercise-7.txt')
    results = find_n_most_similar_to_operation_result(model, 5, [('Sąd Najwyższy', 'kpc', 'konstytucja'),
                                                                 ('pasażer', 'mężczyzna', 'kobieta'),
                                                                 ('samochód', 'droga', 'rzeka')])
    save_data("\n".join(results), 'exercise-8.txt')
    plot_tsne_fitting_results(model, ['szkoda', 'strata', 'uszczerbek', 'szkoda majątkowa', 'uszczerbek na zdrowiu',
                                      'krzywda', 'niesprawiedliwość', 'nieszczęście'], 'exercise-9.png')


def find_n_most_similar(model, top_n, words):
    results = []
    for word in words:
        try:
            most_similar = model.most_similar(positive=convert_to_compatible_with_word2vec(word))
            result = '{} most similar words to word {} are:\n'.format(top_n, word)
            result += '\n'.join(['{} - similarity: {}'.format(result_word, round(similarity, 4))
                                 for result_word, similarity in most_similar])
            results.append(result)
        except:
            results.append("Word {} cannot be found\n".format(word))
    return results


def convert_to_compatible_with_word2vec(word):
    return word.lower().replace(' ', '_') + "::noun"


def find_n_most_similar_to_operation_result(model, top_n, operands_list):
    results = []
    for operands in operands_list:
        try:
            vector = calculate_operation_result(model, *operands)
            most_similar = model.similar_by_vector(vector=vector)
            result = '{} most similar words to result of operation: "{} - {} + {}" are:\n' \
                .format(top_n, *operands)
            result += '\n'.join(['{} - similarity: {}'.format(result_word, round(similarity, 4))
                                 for result_word, similarity in most_similar])
            results.append(result)
        except:
            results.append("Word {} cannot be found\n".format(operands))
    return results


def calculate_operation_result(model, first_word, second_word, third_word):
    return model[convert_to_compatible_with_word2vec(first_word)] \
           - model[convert_to_compatible_with_word2vec(second_word)] \
           + model[convert_to_compatible_with_word2vec(third_word)]


def to_coordinates(model, words):
    return numpy.array([model.get_vector(word) for word in words])


def get_random_points(model, param):
    vocs = list(model.vocab)[:]
    shuffle(vocs)
    return to_coordinates(model, vocs[:param])


def plot_tsne_fitting_results(model, words, file_name):
    words = map(convert_to_compatible_with_word2vec, words)
    vectors = []
    for word in words:
        try:
            vector = numpy.sum([model.get_vector(part) for part in word.split()], axis=0)
            vectors.append(vector)
        except KeyError:
            print("Word {} cannot be found\n".format(word))
    random_vectors = list(map(lambda x: model.get_vector(x), random.sample(list(model.vocab.keys()), 1000 - len(vectors))))
    fitting_results = TSNE(n_components=2).fit_transform(vectors + random_vectors)

    pyplot.figure(figsize=(10, 10))
    pyplot.title('t-SNE visualization of word2vec vectors')
    pyplot.scatter(fitting_results[:len(vectors), 0], fitting_results[:len(vectors), 1], c="r")
    pyplot.scatter(fitting_results[len(vectors):, 0], fitting_results[len(vectors):, 1], c="y")
    pyplot.savefig(join("output", file_name))


def save_data(data, file_name):
    with open(join("output", file_name), 'w') as file:
        file.write(data)


if __name__ == '__main__':
    main()
