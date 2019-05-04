from gc import collect
from os import listdir
from os.path import getsize, basename, join
from typing import List, Tuple

from gensim.models import KeyedVectors
from gensim.models import Phrases, Word2Vec
from gensim.models.phrases import Phraser
from matplotlib import pyplot
from numpy.random import rand
from sklearn.manifold import TSNE


BYTES_IN_MEGABYTE = 1024 * 1024


def main():
    # input_dir, minimum_data_size = parse_arguments()
    # files = get_files_to_be_processed(input_dir)
    # files = select_random_files(files, minimum_data_size)
    # preprocessed_files = preprocess_judgements(files)
    # trigram_files = convert_to_trigram_sentences(preprocessed_files)
    # trigram_files = convert_to_trigram_sentences(listdir(PREPROCESSED_DATA_DIRECTORY_PATH))
    # model = create_model(trigram_files)
    # model = create_model(listdir(TRIGRAMS_DATA_DIRECTORY_PATH))
    # model.save(join(OUTPUT_DIRECTORY_PATH, 'trigrams.model'))
    model = KeyedVectors.load_word2vec_format("C:/Users/Getek/Downloads/skipgram_v100/skipgram/skip_gram_v100m8.w2v.txt", binary=False)
    results = find_n_most_similar(model, 3,
                                  ['Sąd Najwyższy', 'Trybunał Konstytucyjny', 'kodeks cywilny', 'kpk', 'sąd rejonowy',
                                   'szkoda', 'wypadek', 'kolizja', 'szkoda majątkowa', 'nieszczęście', 'rozwód'])
    save_data("output", results, 'exercise-7.txt')
    results = find_n_most_similar_to_operation_result(model, 5, [('Sąd Najwyższy', 'kpc', 'konstytucja'),
                                                                 ('pasażer', 'mężczyzna', 'kobieta'),
                                                                 ('samochód', 'droga', 'rzeka')])
    save_data("output", results, 'exercise-8.txt')
    plot_tsne_fitting_results(model, ['szkoda', 'strata', 'uszczerbek', 'szkoda majątkowa', 'uszczerbek na zdrowiu',
                                      'krzywda', 'niesprawiedliwość', 'nieszczęście'], 'exercise-9.png')


# def preprocess_judgements(files: List[str]) -> List[str]:
#     file_names = []
#     for file in tqdm(files, unit='judgement files'):
#         judgements = extract_judgements_from_file(file)
#         judgements = [replace_redundant_characters(judgement["textContent"]) for judgement in judgements]
#         file_name = basename(file).replace('.json', '.txt')
#         save_data(PREPROCESSED_DATA_DIRECTORY_PATH, judgements, file_name)
#         file_names.append(file_name)
#     return file_names

#
# def select_random_files(files: List[str], minimum_data_size: float) -> List[str]:
#     total_size = get_total_size_in_megabytes(files)
#     division_point = get_division_point(total_size, minimum_data_size)
#     return files[:int(len(files) / division_point)]
#
#
# def get_total_size_in_megabytes(files: List[str]) -> float:
#     return sum([getsize(file) for file in files]) / BYTES_IN_MEGABYTE
#
#
# def get_division_point(total_size: float, minimum_data_size: float) -> int:
#     divider = 1
#     while total_size / (divider + 1) > minimum_data_size:
#         divider += 1
#     return divider + 1 if total_size / divider + 1 >= 0.9 * minimum_data_size else divider
#
#
# def convert_to_trigram_sentences(preprocessed_files_file_names: List[str]) -> List[str]:
#     for file_name in tqdm(preprocessed_files_file_names, unit='preprocessed files'):
#         judgements = read_data(PREPROCESSED_DATA_DIRECTORY_PATH, file_name)
#         sentences_stream = convert_to_sentences_stream(judgements)
#         phrases = Phrases(sentences_stream)
#         bigram = Phraser(phrases)
#         trigram = Phrases(bigram[sentences_stream])
#         trigram_phrases = [trigram[bigram[sentence]] for sentence in sentences_stream]
#         trigram_sentences = [' '.join(trigrams) for trigrams in trigram_phrases]
#         save_data(TRIGRAMS_DATA_DIRECTORY_PATH, trigram_sentences, file_name)
#     return preprocessed_files_file_names
#
#
# def convert_to_sentences_stream(judgements):
#     sentences_stream = []
#     for judgement in judgements:
#         sentences = extract_sentences(judgement)
#         for sentence in sentences:
#             sentences_stream.append(convert_to_stream(sentence))
#     return sentences_stream
#
#
# def convert_to_stream(sentence: str) -> List[str]:
#     return sentence.lower().replace('.', '').replace(',', '').split(' ')
#
#
# def create_model(trigram_files: List[str]):
#     sentences = []
#     for file_name in tqdm(trigram_files, unit='trigram files'):
#         trigram_sentences = read_data(TRIGRAMS_DATA_DIRECTORY_PATH, file_name)
#         for sentence in trigram_sentences:
#             sentences.append(sentence.split(' '))
#         collect()
#     print('Training started')
#     model = Word2Vec(sentences, sg=0, window=5, size=300, min_count=3, workers=4)
#     print('Training finished')
#     return model


def find_n_most_similar(model, top_n: int, words: List[str]) -> List[str]:
    results = []
    for word in words:
        most_similar = model.most_similar(positive=convert_to_compatible_with_word2vec(word))
        result = '{} most similar words to word {} are:\n'.format(top_n, word)
        result += '\n'.join(['{} - similarity: {}'.format(result_word, round(similarity, 4))
                             for result_word, similarity in most_similar])
        results.append(result)
    return results


def convert_to_compatible_with_word2vec(word: str) -> str:
    return word.lower().replace(' ', '_') + '::noun'


def find_n_most_similar_to_operation_result(model: Word2Vec, top_n: int, operands_list: List[Tuple[str, str, str]]) \
        -> List[str]:
    """
    :param operands_list: list of operands, each of them is tuple of 3 words for which we will calculate:
    first word - second + third
    """
    results = []
    for operands in operands_list:
        vector = calculate_operation_result(model, *operands)
        most_similar = model.wv.similar_by_vector(vector=vector, topn=top_n)
        result = '{} most similar words to result of operation: "{} - {} + {}" are:\n' \
            .format(top_n, *operands)
        result += '\n'.join(['{} - similarity: {}'.format(result_word, round(similarity, 4))
                             for result_word, similarity in most_similar])
        results.append(result)
    return results


def calculate_operation_result(model: Word2Vec, first_word: str, second_word: str, third_word: str):
    return model[convert_to_compatible_with_word2vec(first_word)] \
           - model[convert_to_compatible_with_word2vec(second_word)] \
           + model[convert_to_compatible_with_word2vec(third_word)]


def plot_tsne_fitting_results(model: Word2Vec, words: List[str], file_name: str):
    vectors = [model[convert_to_compatible_with_word2vec(word)] for word in words]
    fitting_results = TSNE().fit_transform(vectors)
    x = fitting_results[:, 0]
    y = fitting_results[:, 1]
    colors = rand(len(fitting_results))
    pyplot.figure(figsize=(12, 9), dpi=100)
    pyplot.title('t-SNE visualization of word2vec vectors')
    pyplot.scatter(x, y, s=50, c=colors)
    for i in range(len(fitting_results)):
        pyplot.annotate(words[i], xy=(x[i], y[i]))
    pyplot.grid(True)
    pyplot.savefig(join("output", file_name))
    pyplot.close()


def save_data(data: str, file_name: str) -> None:
    with open(join("output", file_name), 'w') as file:
        file.write(data)


if __name__ == '__main__':
    main()