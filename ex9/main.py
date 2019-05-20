import os
import random
from _elementtree import Element
from json import loads
from os.path import join
from re import sub
from time import sleep, time
from typing import List, Dict, Tuple
from xml.etree import ElementTree as etree

from matplotlib import pyplot
from requests import post, get
ENCODING = r'utf-8'
API_PATH = r'http://ws.clarin-pl.eu/nlprest2/base'
HEADERS = {'content-type': 'application/json', 'charset': 'utf-8'}
START_TASK_PATH = API_PATH + r'/startTask'
GET_STATUS_PATH = API_PATH + r'/getStatus/{}'
GET_FILE_PATH = API_PATH + r'/download'


def read_bills(directory_path, bills):
    result = []
    for bill in bills:
        with open(os.path.join(directory_path, bill), 'r', encoding="utf8") as fin:
            result.append(fin.read())
    return result

def main():
    directory_path = os.path.join("..", "ustawy")
    files = get_files_to_be_processed(directory_path)
    bills = select_n_bills(files, 1)
    # save_data(bills, 'data.txt')
    bill_contents = read_bills(directory_path, bills)
    task_ids = create_ner_tasks(bill_contents)
    task_ids = list(filter(lambda x: x[0] == 200, task_ids))
    task_ids = [x[1] for x in task_ids]
    # save_data([x[1] for x in task_ids], 'tasks.txt')
    task_map = create_task_map(task_ids)
    task_map = wait_for_tasks_to_complete(task_map)
    task_map = {task_id: get_result(task_map[task_id]) for task_id in task_map.keys()}
    # save_task_results(task_map)
    task_results = [task_map[id] for id in task_map]
    category_map = create_category_map(task_results)
    super_category_map = create_super_category_map(category_map)
    # plot_cardinality(calculate_category_cardinality(category_map), 'exercise-6-I.png')
    # plot_cardinality(calculate_category_cardinality(super_category_map), 'exercise-6-II.png')
    top_in_categories = find_top_k_in_categories(category_map, 50)
    top_in_categories = convert_top_in_categories_to_list(top_in_categories, 50)
    # save_data(top_in_categories, 'exercise-7.txt')
    top_in_super_categories = find_top_k_in_categories(super_category_map, 10)
    top_in_super_categories = convert_top_in_super_categories_to_list(top_in_super_categories)
    # save_data(top_in_super_categories, 'exercise-8.txt')

def save_task_results(task_map):
    for task_id in task_map.keys():
        with open(os.path.join("task", "task_" + str(task_id)), 'w+') as file:
            file.write(task_map[task_id])


def get_files_to_be_processed(directory_path):
    files_to_be_processed = []
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            files_to_be_processed.append(os.path.join(directory_path, filename))
    return files_to_be_processed



def select_n_bills(file_names, n):
    return [random.choice(file_names) for i in range(0, n)]


def create_ner_tasks(bills):
    return [create_ner_task(bill) for bill in bills]

def replace_all_redundant_characters(content):
    content = content.replace('-\n', ' ')
    content = content.replace('\n', ' ')
    content = content.replace('\t', ' ')
    content = sub("<[^>]*>", "", content)
    content = sub('[,.;:\-–−()\[\]"„…”/]', "", content)
    return content

def create_ner_task(bill):
    bill = replace_all_redundant_characters(bill)
    data = get_json_as_string("json", 'start_task.json').replace('query_data', bill).encode(ENCODING)
    response = post(url=START_TASK_PATH, headers=HEADERS, data=data)
    return (response.status_code, response.content.decode(ENCODING))


def get_json_as_string(working_dir, file_name):
    return __read_json_file_to_string(join(working_dir, file_name))


def __read_json_file_to_string(file_path):
    with open(file_path, 'r', encoding="utf8") as file:
        return file.read().replace('\n', '')

def create_task_map(task_ids):
    tasks_with_status = {task_id: get_task_status(task_id) for task_id in task_ids}
    return tasks_with_status


def get_task_status(task_id):
    response = get(url=GET_STATUS_PATH.format(task_id))
    return loads(response.content.decode(ENCODING))


def wait_for_tasks_to_complete(task_map: Dict) -> Dict:
    print('Waiting for tasks to complete...')
    not_completed = [task_id for task_id in task_map.keys() if not is_completed(task_map, task_id)]
    start = time()
    while len(not_completed) > 0:
        print('Tasks left: ' + str(len(not_completed)))
        now_completed = []
        for task_id in not_completed:
            task_map[task_id] = get_task_status(task_id)
            if is_completed(task_map, task_id):
                now_completed.append(task_id)
            sleep(1)
        for completed in now_completed:
            not_completed.remove(completed)
    print('Waiting took: {}s'.format(round(time() - start, 2)))
    return task_map


def is_completed(task_map, task_id):
    return task_map[task_id]['status'] == 'DONE'


def get_result(status):
    file_path = status['value'][0]['fileID']
    response = get(url=GET_FILE_PATH + file_path)
    return response.content.decode(ENCODING)


def create_category_map(task_results):
    category_map = {}
    for xml in task_results:
        chunk_list = etree.fromstring(xml)
        for chunk in chunk_list:
            for sentence in chunk:
                parse_sentence(sentence, category_map)
    return category_map


def parse_sentence(sentence, category_map):
    tokens = [token for token in sentence]
    for i in range(0, len(tokens)):
        token = tokens[i]
        if is_token(token):
            for ann_tag in get_ann_tags(token):
                if int(ann_tag.text) > 0:
                    phrase_words = [get_token_text(token)]
                    j = 1
                    next_token = get_next_token_or_none(i, j, tokens)
                    if next_token is None:
                        update_category_map(category_map, ann_tag.attrib['chan'], get_token_text(token))
                    while is_token(next_token):
                        corresponding_ann_tag = get_corresponding_ann_tag(next_token, ann_tag)
                        if corresponding_ann_tag is not None:
                            phrase_words.append(get_token_text(next_token))
                            next_token.remove(corresponding_ann_tag)
                        else:
                            update_category_map(category_map, ann_tag.attrib['chan'], ' '.join(phrase_words))
                            break
                        j += 1
                        next_token = get_next_token_or_none(i, j, tokens)
                        if next_token is None:
                            update_category_map(category_map, ann_tag.attrib['chan'], ' '.join(phrase_words))


def is_token(token):
    return token is not None and token.tag == 'tok'


def get_ann_tags(token):
    return [child for child in token if child.tag == 'ann']


def get_token_text(token):
    return token[0].text


def get_next_token_or_none(i, j, tokens):
    return tokens[i + j] if i + j < len(tokens) else None


def update_category_map(category_map, category, phrase):
    if category not in category_map:
        category_map[category] = {}
    if phrase not in category_map[category]:
        category_map[category][phrase] = 1
    else:
        category_map[category][phrase] += 1


def get_corresponding_ann_tag(next_token, ann_tag):
    next_token_ann_tags = get_ann_tags(next_token)
    for tag in next_token_ann_tags:
        if tag.attrib['chan'] == ann_tag.attrib['chan'] and tag.text == ann_tag.text:
            return tag
    return None


def create_super_category_map(category_map):
    super_category_map = {}
    for category in category_map.keys():
        super_category = '_'.join(category.split('_')[:2])
        if super_category not in super_category_map:
            super_category_map[super_category] = {}
        for word in category_map[category].keys():
            if word not in super_category_map[super_category]:
                super_category_map[super_category][word] = category_map[category][word]
            else:
                super_category_map[super_category][word] += category_map[category][word]
    return super_category_map


def calculate_category_cardinality(category_map):
    category_cardinality = {}
    for category in category_map:
        cardinality = 0
        for word in category_map[category].keys():
            cardinality += category_map[category][word]
        category_cardinality[category] = cardinality
    return category_cardinality


def plot_cardinality(cardinality_map, filename):
    output_file = join("output", filename)
    categories = [category for category in cardinality_map.keys()]
    cardinalities = cardinality_map.values()
    pyplot.figure(figsize=(12, 9), dpi=100, tight_layout=True)
    pyplot.xticks(rotation='vertical')
    pyplot.yscale('log')
    pyplot.bar(categories, cardinalities)
    pyplot.tight_layout()
    pyplot.title('Cardinality of categories')
    pyplot.xlabel('Categories')
    pyplot.ylabel('Cardinality [log]')
    pyplot.grid(True)
    pyplot.savefig(output_file)
    pyplot.close()


def find_top_k_in_categories(category_map, k):
    category_top_words_map = {}
    for category in category_map.keys():
        words_in_category = []
        for word in category_map[category].keys():
            words_in_category.append((word, category_map[category][word]))
        words_in_category = sorted(words_in_category, key=lambda pair: pair[1], reverse=True)
        category_top_words_map[category] = words_in_category[:k] if len(words_in_category) > k else words_in_category
    return category_top_words_map


def convert_top_in_categories_to_list(top_in_categories, k):
    top_words = []
    for category, items in top_in_categories.items():
        top_words.extend((word, occurrences, category) for word, occurrences in items)
    top_words = sorted(top_words, key=lambda word: word[1], reverse=True)
    top_words = top_words[:k] if len(top_words) > k else top_words
    return ['{}\t{}\t{}'.format(word, occurrences, category) for word, occurrences, category in top_words]


def convert_top_in_super_categories_to_list(top_in_super_categories):
    words = []
    for category, items in top_in_super_categories.items():
        words.append(category + ':' + "\n")
        words.extend(['{}\t{}'.format(word, occurrences) for word, occurrences in items])
    return words

def save_data(data, file_name):
    with open(join("output", file_name), 'w+') as file:
        file.write("\n".join(data))

if __name__ == '__main__':
    main()