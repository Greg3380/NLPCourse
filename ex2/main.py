import os
from os import getcwd
from os.path import join
from json import loads, dumps
import requests

HOST = r'http://localhost:9200'
INDEX_URL = HOST + r'/nlp-ex2'
INDEX_DATA_URL = INDEX_URL + r'/bills'
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


def search_for_word_count():
    data = get_json_as_string(getcwd(), "legislative_acts_count.json")
    response = requests.post(url=COUNT_QUERY_URL, headers=HEADERS, data=data.encode('utf-8'))
    counter = get_count_from_response(response)
    with open(os.path.join("output", "exercise2_6"), 'w+', encoding="utf8") as fout:
        fout.write("Occurrances of word 'ustawa ':\n")
        fout.write(str(counter))


def search_for_phrase():
    data = get_json_as_string(getcwd(), "search_for_phrase.json")
    response = requests.post(url=COUNT_QUERY_URL, headers=HEADERS, data=data.encode('utf-8'))
    counter = get_count_from_response(response)
    with open(os.path.join("output", "exercise2_7"), 'w+', encoding="utf8") as fout:
        fout.write("Occurrances of phrase 'kodeks postępowania cywilnego ':\n")
        fout.write(str(counter))


def search_for_phrase_with_additional_words():
    data = get_json_as_string(getcwd(), "search_for_phrase_with_additional_words.json")
    response = requests.post(url=COUNT_QUERY_URL, headers=HEADERS, data=data.encode('utf-8'))
    counter = get_count_from_response(response)
    with open(os.path.join("output", "exercise2_8"), 'w+', encoding="utf8") as fout:
        fout.write("Occurrances of phrase 'wchodzi w życie  ':\n")
        fout.write(str(counter))


def search_for_most_relevant_docs():
    data = get_json_as_string(getcwd(), "search_for_most_relevant_documents.json")
    response = requests.get(url=SEARCH_QUERY_URL, headers=HEADERS, data=data.encode('utf-8'))
    response_data = response.content.decode('utf-8')
    result = loads(response_data)["hits"]["hits"]
    score_and_bill_titles = []
    for entry in result:
        score_and_bill_titles.append((entry["_score"], entry["_source"]["billTitle"]))
    with open(os.path.join("output", "exercise2_9"), 'w+', encoding="utf8") as fout:
        fout.write("Most relevant documents for phrase 'konstytucja' ':\n")
        for entry in score_and_bill_titles:
            fout.write("Bill title: " + entry[1] + " , Score: " + str(entry[0]) + "\n")


def upload_bills(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            with open(os.path.join(directory_path, filename), 'r', encoding="utf8") as fin:
                print("Bill name: " + filename + "\n\n")
                content = fin.read()
                data = prepare_data(filename, content)
                res = requests.post(url=INDEX_DATA_URL, headers=HEADERS, data=data)
                print(res)


def create_index_with_analyzer():
    data = get_json_as_string(getcwd(), "analyzer.json")
    response = requests.put(url=INDEX_URL, headers=HEADERS, data=data.encode('utf-8'))
    return response.content


def get_count_from_response(response) -> int:
    response_data = response.content.decode('utf-8')
    return int(loads(response_data)["count"])


def search_for_most_relevant_docs_highlights():
    data = get_json_as_string(getcwd(), "most_relevant_docs_highlights.json")
    response = requests.get(url=SEARCH_QUERY_URL, headers=HEADERS, data=data.encode('utf-8'))
    response_data = response.content.decode('utf-8')
    result = loads(response_data)["hits"]["hits"]
    fragmenets_and_bill_titles = []
    for entry in result:
        fragmenets_and_bill_titles.append((entry["highlight"]["textContent"], entry["_source"]["billTitle"]))

    with open(os.path.join("output", "exercise2_10"), 'w+', encoding="utf8") as fout:
        fout.write("Fragments of bills with phrase 'konstytucja' ':\n")
        for entry in fragmenets_and_bill_titles:
            fout.write("Bill title: " + entry[1] + " , Fragments:  \n" + "\n".join(entry[0]) + "\n\n")


def main():
    directory_path = os.path.join("..", "ustawy")
    create_index_with_analyzer()
    # upload_bills(directory_path)
    search_for_word_count()
    search_for_phrase()
    search_for_phrase_with_additional_words()
    search_for_most_relevant_docs()
    search_for_most_relevant_docs_highlights()


if __name__ == '__main__':
    main()
