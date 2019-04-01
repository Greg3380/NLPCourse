import os
from collections import Counter
from collections import OrderedDict
from json import loads, dumps
from os import getcwd
from os.path import join

import requests
from matplotlib import pyplot

HOST = r'http://localhost:9200'
INDEX_URL = HOST + r'/nlp-ex3'
INDEX_DATA_URL = INDEX_URL + r'/_doc'
TERM_VECTOR_DATA_URL = INDEX_DATA_URL + r'/{doc_number}/_termvectors'
COUNT_QUERY_URL = INDEX_DATA_URL + r'/_count'
SEARCH_QUERY_URL = INDEX_DATA_URL + r'/_search'
HEADERS = {'content-type': 'application/json'}


def get_json_as_string(working_dir, file_name):
    return __read_json_file_to_string(join(working_dir, "json", file_name))


def __read_json_file_to_string(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        return file.read().replace('\n', '')


def prepare_data(filename, content):
    content = content.replace("\n", " ")
    data = {"textContent": content, "billTitle": filename}
    return dumps(data) + '\n'


def upload_bills(directory_path):
    for root, dirs, files in os.walk(directory_path):
        i = 1
        for filename in files:
            with open(os.path.join(directory_path, filename), 'r', encoding="utf8") as fin:
                print("Bill name: " + filename + "\n\n")
                content = fin.read()
                data = prepare_data(filename, content)
                res = requests.post(url=INDEX_DATA_URL + '/' + str(i), headers=HEADERS, data=data)
                i += 1
                print(res)


def create_index_with_analyzer():
    data = get_json_as_string(getcwd(), "analyzer.json")
    response = requests.put(url=INDEX_URL, headers=HEADERS, data=data.encode('utf-8'))
    return response.content


def retrieve_terms(bills_count):
    term_to_occurrances_dict = Counter()
    for i in range(1, bills_count + 1):
        data = get_json_as_string(getcwd(), "term_vectors.json")
        response = requests.get(url=TERM_VECTOR_DATA_URL.replace("{doc_number}", str(i)), headers=HEADERS,
                                data=data.encode('utf-8'))
        response_data = response.content.decode('utf-8')
        all_terms = loads(response_data)["term_vectors"]["textContent"]["terms"]
        for key, value in all_terms.items():
            term_to_occurrances_dict[key] += value["term_freq"]
    return term_to_occurrances_dict


def count_all_bills_in_directory(directory_path):
    counter = 0
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            with open(os.path.join(directory_path, filename), 'r', encoding="utf8") as fin:
                counter += 1
    return counter


def filter_frequency_list(frequency_list):
    for key in list(frequency_list):
        if any(char.isdigit() for char in key) or len(key) < 2:
            del frequency_list[key]


def prepare_data_for_plotting(data):
    tmp = data.most_common()
    y = [occurrance[1] for occurrance in tmp]
    x = list(range(1, len(y) + 1))
    return x, y


def create_chart(data):
    x, y = prepare_data_for_plotting(data)
    pyplot.loglog(x, y)
    pyplot.xlabel('Index on the frequency list')
    pyplot.ylabel('Number of word occurrences')
    pyplot.grid(True)
    pyplot.savefig(os.path.join("output", "exercise-4.png"), dpi=150)
    pyplot.close()


def read_dictionary_data(dictionary_path):
    with open(dictionary_path, 'r', encoding="utf8") as file:
        data = {}
        for line in file.readlines():
            word = line.split(';')[1].lower()
            if len(word) > 1:
                bucket = word[:2].lower()
                try:
                    data[bucket].append(word)
                except KeyError:
                    data[bucket] = [word]
    return data


def find_and_save_words_that_are_not_in_dictionary(dictionary_path, results):
    dictionary_data = read_dictionary_data(dictionary_path)
    not_in_dictionary = []
    result_list = [entry for entry in results]
    for word in result_list:
        bucket = word[:2].lower()
        try:
            if word not in dictionary_data[bucket]:
                not_in_dictionary.append(word)
        except KeyError:
            not_in_dictionary.append(word)
    with open(os.path.join("output", "exercise3_5"), 'w+', encoding="utf8") as fout:
        fout.write("All words that do not appear in that dictionary:\n")
        fout.write(str(not_in_dictionary))
    return not_in_dictionary


def find_words_with_highest_rank(words_not_in_dict, frequency_list):
    most_common_words = frequency_list.most_common()
    intersection = list(filter(lambda x: x[0] in words_not_in_dict, most_common_words))
    return intersection[:30], list(filter(lambda x: x[1] == 3, intersection))[:30]


def find_corrections_using_Levenshtein_distance(not_in_dictionary, results,
                                                distance):
    words_with_corrections = OrderedDict()
    for word in not_in_dictionary:
        best_match = None
        best_score = 0
        for candidate, score in results:
            if length_difference_is_not_greater_than_distance(word, candidate, distance) \
                    and Levenshtein_distance(word, candidate) <= distance and word != candidate:
                if best_score < score:
                    best_match = candidate
                    best_score = score
        words_with_corrections[word] = (best_match, best_score)
    return words_with_corrections


def length_difference_is_not_greater_than_distance(word, candidate, distance):
    return abs(len(word) - len(candidate)) <= distance


def Levenshtein_distance(word: str, candidate: str) -> int:
    distances = [[0 for i in range(len(candidate) + 1)] for j in range(len(word) + 1)]
    for i in range(0, len(word) + 1):
        distances[i][0] = i
    for j in range(0, len(candidate) + 1):
        distances[0][j] = j
    for j in range(1, len(candidate) + 1):
        for i in range(1, len(word) + 1):
            if word[i - 1] == candidate[j - 1]:
                cost = 0
            else:
                cost = 1
            deletion_cost = distances[i - 1][j] + 1
            insertion_cost = distances[i][j - 1] + 1
            substitution_cost = distances[i - 1][j - 1] + cost
            distances[i][j] = min(deletion_cost, insertion_cost, substitution_cost)
    return distances[len(word)][len(candidate)]


def main():
    directory_path = os.path.join("..", "ustawy")
    bills_count = count_all_bills_in_directory(directory_path)
    create_index_with_analyzer()
    # upload_bills(directory_path)
    frequency_list = retrieve_terms(bills_count)
    filter_frequency_list(frequency_list)
    create_chart(frequency_list)
    words_not_in_dict = find_and_save_words_that_are_not_in_dictionary(
        os.path.join("polimorfologik-2.1", "polimorfologik-2.1.txt"), frequency_list)
    highest_rank, words_with_certain_occurrances = find_words_with_highest_rank(words_not_in_dict, frequency_list)
    with open(os.path.join("output", "exercise3_6"), 'w+', encoding="utf8") as fout:
        fout.write("30 words with the highest ranks that do not belong to the dictionary\n")
        fout.write(str(highest_rank))
    with open(os.path.join("output", "exercise3_7"), 'w+', encoding="utf8") as fout:
        fout.write("30 words with 3 occurrences that do not belong to the dictionary:\n")
        fout.write(str(words_with_certain_occurrances))

    words_to_find_correction = [entry[0] for entry in words_with_certain_occurrances]

    levenshtein_distances = OrderedDict()
    distance = 1
    while words_to_find_correction:
        distances = find_corrections_using_Levenshtein_distance(words_to_find_correction,
                                                           [(entry, frequency_list[entry]) for entry in frequency_list],distance)

        for key in distances:
            if distances[key][0] is not None:
                levenshtein_distances[key] = (distances[key], distance)
                words_to_find_correction.remove(key)
        distance += 1

    with open(os.path.join("output", "exercise3_8"), 'w+', encoding="utf8") as fout:
        fout.write("Words corrections using Levensthtein distance:\n")
        fout.write(str(levenshtein_distances))

if __name__ == '__main__':
    main()
