from __future__ import division

import io
import math
import os
from collections import Counter
from operator import itemgetter
from re import sub, search, match

import regex as regex
import requests

HOST = r'http://localhost:9200'
HEADERS = {'content-type': 'text/plain', 'charset': 'utf-8'}


def main():
    directory_path = os.path.join("..", "ustawy")
    files = get_files_to_be_processed(directory_path)
    bigram_frequency_list = Counter()
    for file in files:
        words = []
        tokens = []
        result_tokens = extract_and_upload_data(file, HOST, HEADERS)
        for token in result_tokens:
            split = token.split(":")
            words.append(split[0])
            tokens.append(token)
        count_double_words_in_content(tokens, bigram_frequency_list)

    llr_list = create_llr_list(bigram_frequency_list)
    with io.open(os.path.join("output", "exercise5_5"), 'w+') as fout:
        fout.write(u'Top 100 llr bigrams:\n')
        fout.write(unicode(str("\n".join(str(i) for i in llr_list[:100]))))
    filtered_llr = filter_llr(llr_list)
    with io.open(os.path.join("output", "exercise5_6"), 'w+') as fout:
        fout.write(u'Top 50 filtered llr bigrams:\n')
        fout.write(unicode(str("\n".join(str(i) for i in filtered_llr[:50])).encode('UTF-8')))


def tagged_to_tokens(tags):
    pattern = r'(?=^(\p{L}+)).*?\n\s(\p{L}*)\s(\p{L}*)'
    results = list(regex.finditer(pattern, tags, flags=regex.MULTILINE))
    return ['{0}:{1}'.format(r.group(2), r.group(3)) for r in results]


def create_bigram_list(bill_files):
    words_with_number_of_occurrences = dict()
    for file_path in bill_files:
        words = get_words_from_tagged_judgement(file_path)
        for i in range(1, len(words)):
            phrase = words[i - 1:i + 1]
            is_phrase = True
            for word in phrase:
                if not is_not_a_number(word.split(':')[0]) or not is_a_word(word.split(':')[0]):
                    is_phrase = False
                    break
            if is_phrase:
                update_dictionary(words_with_number_of_occurrences, ' '.join(phrase).lower())
    return sort_dictionary(words_with_number_of_occurrences)


def update_dictionary(words_with_number_of_occurrences, word):
    try:
        words_with_number_of_occurrences[word] += 1
    except KeyError:
        words_with_number_of_occurrences[word] = 1


def sort_dictionary(results):
    return sorted(results.items(), key=itemgetter(1), reverse=True)


def create_llr_list(bigram_frequency_list):
    bigram_total_words = sum_occurrences_in_ngram_frequency_list(bigram_frequency_list)
    event_occurrences = count_event_occurrences(bigram_frequency_list)
    llr_list = []
    for phrase in bigram_frequency_list:
        a, b = phrase.split(' ')
        k_11 = bigram_frequency_list[phrase]
        k_12 = event_occurrences[b][1] - bigram_frequency_list[phrase]
        k_21 = event_occurrences[a][0] - bigram_frequency_list[phrase]
        k_22 = bigram_total_words - k_11 - k_12 - k_21
        entropy1 = entropy([k_11, k_12, k_21, k_22])
        entropy2 = entropy([k_11 + k_12, k_21 + k_22])
        entropy3 = entropy([k_11 + k_21, k_12 + k_22])
        llr = 2 * (k_11 + k_12 + k_21 + k_22) * (entropy1 - entropy2
                                                 - entropy3)
        llr_list.append((phrase, llr))
    return sorted(llr_list, key=itemgetter(1), reverse=True)


def sum_occurrences_in_ngram_frequency_list(ngram_frequency_list):
    return sum(ngram_frequency_list[ngram] for ngram in ngram_frequency_list)


def count_event_occurrences(bigram_frequency_list):
    event_occurrences = dict()
    # word -> (number of occurrences as first element, number of occurrences as second element)
    for phrase in bigram_frequency_list:
        a, b = phrase.split(' ')
        try:
            event_occurrences[a] = (event_occurrences[a][0] + bigram_frequency_list[phrase], event_occurrences[a][1])
        except KeyError:
            event_occurrences[a] = (bigram_frequency_list[phrase], 0)
        try:
            event_occurrences[b] = (event_occurrences[b][0], event_occurrences[b][1] + bigram_frequency_list[phrase])
        except KeyError:
            event_occurrences[b] = (0, bigram_frequency_list[phrase])
    return event_occurrences


def nlogn(item, N):
    k_div_N = item / N
    return k_div_N * math.log(k_div_N + int(item == 0), 2)


def entropy(items):
    N = sum(items)
    return sum(nlogn(item, N) for item in items)


def filter_llr(llr_list):
    filtered = []
    for phrase, count in llr_list:
        first_word, second_word = phrase.split(' ')
        if is_noun(first_word) and (is_noun(second_word) or is_adjective(second_word)):
            filtered.append((phrase, count))
    return filtered


def get_files_to_be_processed(directory_path):
    files_to_be_processed = []
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            files_to_be_processed.append(os.path.join(directory_path, filename))
    return files_to_be_processed


def extract_and_upload_data(file_path, url, headers):
    with io.open(file_path, 'r', encoding="utf8") as file:
        print("Bill name: " + file_path + "\n\n")
        content = file.read()
        content = replace_all_redundant_characters(content)
        data = replace_all_redundant_characters(content)
        response = requests.post(url=url, headers=headers, data=data.encode('utf-8'))
        response_data = response.content.decode('utf-8')

        tokens = tagged_to_tokens(response_data)
        return tokens


def replace_all_redundant_characters(content):
    content = content.replace('-\n', ' ')
    content = content.replace('\n', ' ')
    content = content.replace('\t', ' ')
    content = sub("<[^>]*>", "", content)
    return content


def save_tagged_judgement(data, bill_filename):
    filename = '{}.txt'.format(bill_filename)
    with io.open(os.path.join('output', filename), 'w+', encoding="utf8") as file:
        file.write(data)


def get_words_from_tagged_judgement(file_path):
    words = []
    with open(file_path) as file:
        for line in file:
            if ':' in line:
                word = ':'.join(line.strip().replace('\t', ':').split(':')[:2])
                words.append(word)
    return words


def is_not_a_number(string):
    return search(r'\d', string) is None


def is_a_word(string):
    return match(r'\w+', string) is not None and not string == string.upper()


def is_noun(string):
    return match('subst', __get_flexeme_abbreviation(string)) is not None


def __get_flexeme_abbreviation(string):
    return string.split(':')[1]


def is_adjective(string):
    return match(r'(\badj[apc]*\b)', __get_flexeme_abbreviation(string)) is not None


def count_double_words_in_content(content, bigrams):
    for i in range(1, len(content)):
        phrase = content[i - 1:i + 1]
        is_phrase = True
        for word in phrase:
            if not is_not_a_number(word) or not is_a_word(word):
                is_phrase = False
                break
        if is_phrase:
            bigrams[' '.join(phrase).lower()] += 1


if __name__ == '__main__':
    main()
