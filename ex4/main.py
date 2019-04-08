import os
from collections import Counter
from math import log2
from operator import itemgetter
from re import fullmatch, sub, search


def is_not_a_number(string):
    return search('\d', string) is None


def is_a_word(string):
    return fullmatch('\w+', string) is not None and not string == string.upper()


def replace_all_redundant_characters(content):
    content = content.replace('-\n', ' ')
    content = content.replace('\n', ' ')
    content = content.replace('\t', ' ')
    content = sub("<[^>]*>", "", content)
    content = sub('[,.;:\-–−()\[\]"„…”/]', "", content)
    return content


def sum_occurrences_in_ngram_frequency_list(ngram_frequency_list):
    return sum(ngram_frequency_list[phrase] for phrase in ngram_frequency_list)


def convert_ngram_list_to_dict(ngram_frequency_list):
    return {ngram: ngram_frequency_list[ngram] for ngram in ngram_frequency_list}


def create_pmi_list(unigram_frequency_list, bigram_frequency_list):
    unigram_dict = convert_ngram_list_to_dict(unigram_frequency_list)
    unigram_total_words = sum_occurrences_in_ngram_frequency_list(unigram_frequency_list)
    bigram_total_words = sum_occurrences_in_ngram_frequency_list(bigram_frequency_list)
    pmi_list = []
    for phrase in bigram_frequency_list:
        x, y = phrase.split(' ')
        px = unigram_dict[x] / unigram_total_words
        py = unigram_dict[y] / unigram_total_words
        pxy = bigram_frequency_list[phrase] / bigram_total_words
        pmi = log2(pxy / (px * py))
        pmi_list.append((phrase, pmi))
    return sorted(pmi_list, key=itemgetter(1), reverse=True)


def nlogn(item: int, N: int):
    k_div_N = item / N
    return k_div_N * log2(k_div_N + int(item == 0))


def entropy(items):
    N = sum(items)
    return sum(nlogn(item, N) for item in items)


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
        llr = 2 * (k_11 + k_12 + k_21 + k_22) * (entropy([k_11, k_12, k_21, k_22]) - entropy([k_11 + k_12, k_21 + k_22])
                                                 - entropy([k_11 + k_21, k_12 + k_22]))
        llr_list.append((phrase, llr))
    return sorted(llr_list, key=itemgetter(1), reverse=True)


def compute_unigrams_and_bigrams(directory_path):
    unigrams = Counter()
    bigrams = Counter()
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            with open(os.path.join(directory_path, filename), 'r', encoding="utf8") as fin:
                print("Bill name: " + filename + "\n\n")
                content = fin.read()
                content = replace_all_redundant_characters(content)
                content = content.split()
                count_single_words_in_content(content, unigrams)
                count_double_words_in_content(content, bigrams)
    with open(os.path.join("output", "exercise4_1"), 'w+', encoding="utf8") as fout:
        fout.write("Top 1000 bigram counts:\n")
        common = bigrams.most_common()
        fout.write(str("\n".join(str(i) for i in common[:1000])))
    return unigrams, bigrams


def main():
    directory_path = os.path.join("..", "ustawy")
    unigrams, bigrams = compute_unigrams_and_bigrams(directory_path)
    pmi_list = create_pmi_list(unigrams, bigrams)
    with open(os.path.join("output", "exercise4_3"), 'w+', encoding="utf8") as fout:
        fout.write("Top 30 pmi bigrams:\n")
        fout.write(str("\n".join(str(i) for i in pmi_list[:30])))

    llr_list = create_llr_list(bigrams)
    with open(os.path.join("output", "exercise4_4"), 'w+', encoding="utf8") as fout:
        fout.write("Top 30 llr bigrams:\n")
        fout.write(str("\n".join(str(i) for i in llr_list[:30])))


def count_single_words_in_content(content, unigrams):
    for word in content:
        if is_not_a_number(word) and is_a_word(word):
            unigrams[word.lower()] += 1


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
