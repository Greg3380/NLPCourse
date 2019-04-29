import os
import sys
from math import inf, log
from os import getcwd
from os.path import join, devnull

from matplotlib import pylab
from networkx import draw_networkx_edge_labels,draw_networkx, circular_layout

from ex6.pywnxml.WNQuery import WNQuery
from typing import List, Tuple, Dict
import requests

from networkx import DiGraph, compose, dag_longest_path_length, shortest_path_length, NetworkXNoPath


def main():
    # # ex 1-3
    # meanings = search_for_meanings("szkoda")
    # for meaning in meanings:
    #     symonyms = search_for_symonyms(int(meaning['id']))
    #     print(symonyms)
    # print(meanings)
    # # ex4
    wn = WNQuery("C:/Users/Getek/Downloads/plwordnet_3_0/plwordnet_3_0/plwordnet-3.0-visdisc.xml")

    result = find_word(wn, 'szkoda', 'n')
    save_data(result, 'exercise-1-3.txt')
    graph, edge_labels = find_transitive_closure(wn, 'wypadek drogowy', 1, 'n', 'hypernym')
    save_graph(graph, edge_labels, 'exercise-4.png')
    result = find_relation(wn, 'wypadek', 1, 'n', 'hyponym', 1)
    save_data(result, 'exercise-5.txt')
    result2 = find_relation(wn, 'wypadek', 1, 'n', 'hyponym', 2)
    save_data(result2, 'exercise-6.txt')
    words_1 = [('szkoda', 2, 'n'), ('strata', 1, 'n'), ('uszczerbek', 1, 'n'), ('szkoda majątkowa', 1, 'n'),
               ('uszczerbek na zdrowiu', 1, 'n'), ('krzywda', 1, 'n'), ('niesprawiedliwość', 1, 'n'),
               ('nieszczęście', 2, 'n')]
    graph, edge_labels = create_relations_graph(wn, words_1)
    save_graph(graph, edge_labels, 'exercise-7-I.png')
    words_2 = [('wypadek', 1, 'n'), ('wypadek komunikacyjny', 1, 'n'), ('kolizja', 2, 'n'), ('zderzenie', 2, 'n'),
               ('kolizja drogowa', 1, 'n'),
               # ('bezkolizyjny', 2, 'j'), # todo this word is in j pos - WNQuery does not support j pos.
               ('katastrofa budowlana', 1, 'n'), ('wypadek drogowy', 1, 'n')]
    graph, edge_labels = create_relations_graph(wn, words_2)
    save_graph(graph, edge_labels, 'exercise-7-II.png')

    wn = WNQuery("C:/Users/Getek/Downloads/plwordnet_3_0/plwordnet_3_0/plwordnet-3.0-visdisc.xml") # somehow if WNQuery is not re-loaded similarity will be -inf
    max_distance = find_max_distance(wn, 'n', 'hypernym')
    result = measure_similarity(wn, ('szkoda', 2, 'n'), ('wypadek', 1, 'n'), 'hypernym', max_distance)
    save_data(result, 'exercise-8-I.txt')
    result = measure_similarity(wn, ('kolizja', 2, 'n'), ('szkoda majątkowa', 1, 'n'), 'hypernym', max_distance)
    save_data(result, 'exercise-8-II.txt')
    result = measure_similarity(wn, ('nieszczęście', 2, 'n'), ('katastrofa budowlana', 1, 'n'), 'hypernym',
                                max_distance)
    save_data(result, 'exercise-8-III.txt')

def save_data(data: str, file_name: str) -> None:
    with open(join("output", file_name), 'w') as file:
        file.write(data)



def save_graph(data: DiGraph, edge_labels: Dict[Tuple[str, str], str], file_name: str) -> None:
    pylab.figure(figsize=(16, 12), dpi=72)
    pylab.axis('off')
    pos = circular_layout(data)
    node_sizes = [250 * len(node) for node in data.nodes._nodes.keys()]
    draw_networkx(data, pos, node_size=node_sizes, node_color='c')
    draw_networkx_edge_labels(data, pos, edge_labels=edge_labels)
    pylab.savefig(join("output", file_name))


def find_word(wn, word, word_type):
    results = wn.lookUpLiteral(word, word_type)
    output = ''
    if len(results) == 0:
        return 'Word {} with type {} has not been found.'.format(word, word_type)
    output += 'Queried word: {}\n'.format(word)
    output += 'Word type: {}\n'.format(word_type)
    output += 'Results:\n'
    for result in results:
        output += '\nDomain: {}\n'.format(result.domain)
        output += 'Definition: {}\n'.format(result.definition)
        output += 'Synonyms: '
        if len(result.synonyms) == 0:
            output += 'No synonyms have been found.'
        else:
            output += ', '.join([synonym.literal for synonym in result.synonyms])
        output += '\n'
    return output


def find_transitive_closure(wn, word, sense_level, word_type, relation_name):
    synset = wn.lookUpSense(word, sense_level, word_type)
    graph = DiGraph()
    edge_labels = {}
    u = get_vertex_label((word, sense_level, word_type))
    graph.add_node(u)
    for synset_id, relation in synset.ilrs:
        if relation == relation_name:
            synset_in_relation = wn.lookUpID(synset_id, synset_id[-1:])
            for word_in_relation in synset_in_relation.synonyms:
                word_in_relation_as_tuple = (word_in_relation.literal, int(word_in_relation.sense),
                                             synset_in_relation.pos)
                v = get_vertex_label(word_in_relation_as_tuple)
                graph.add_node(v)
                graph.add_edge(v, u)
                edge_labels[(u, v)] = relation_name
                sub_graph, sub_graph_edge_labels = find_transitive_closure(wn, *word_in_relation_as_tuple,
                                                                           relation_name)
                graph = compose(graph, sub_graph)
                edge_labels = {**edge_labels, **sub_graph_edge_labels}
    return graph, edge_labels


def get_vertex_label(word):
    return '_'.join([str(part) for part in word][:2])


def find_relation(wn, word, sense_level, word_type, relation_name, relation_depth_level):
    word_id = find_word_id(wn, word, sense_level, word_type)
    output = 'Words that are in relation {} (relation level: {}) with word {}:\n'.format(relation_name,
                                                                                         relation_depth_level, word)
    words = find_words_in_relation(wn, word_id, relation_name, relation_depth_level)
    output += ', '.join(words)
    return output


def find_word_id(wn, word, sense_level, word_type):
    found_word = wn.lookUpSense(word, sense_level, word_type)
    return found_word.wnid


def find_words_in_relation(wn, word_id, relation_name, relation_depth_level):
    words_in_relation = wn.lookUpRelation(word_id, word_id[-1:], relation_name)
    words = []
    for synset_id in words_in_relation:
        if relation_depth_level == 1:
            synset = wn.lookUpID(synset_id, synset_id[-1:])
            words.extend([synonym.literal for synonym in synset.synonyms])
        else:
            words.extend(find_words_in_relation(wn, synset_id, relation_name, relation_depth_level - 1))
    return words


def create_relations_graph(wn, words):
    word_synset_pairs, synset_ids = [], []
    graph = DiGraph()
    for word in words:
        synset = wn.lookUpSense(*word)
        word_synset_pairs.append((word, synset))
        synset_ids.append(synset.wnid)
        graph.add_node(get_vertex_label(word))
    edge_labels = {}
    for word, synset in word_synset_pairs:
        add_synonym_relations(synset, word, words, edge_labels, graph)
        find_and_add_relations(synset, synset_ids, wn, words, word, graph, edge_labels)
    return graph, edge_labels


def add_synonym_relations(synset, word, words, edge_labels, graph):
    for synonym in synset.synonyms:
        if synonym.literal in [word[0] for word in words] and synonym.literal != word[0]:
            u = get_vertex_label(word)
            v = get_vertex_label((synonym.literal, int(synonym.sense), word[2]))
            graph.add_edge(u, v)
            edge_labels[(u, v)] = 'synonym'
            graph.add_edge(v, u)
            edge_labels[(v, u)] = 'synonym'


def find_and_add_relations(synset, synset_ids, wn, words, word, graph, edge_labels):
    synset.ilrs = [relation for relation in synset.ilrs if relation[0] in synset_ids]
    for id_of_synset_in_relation, relation_name in synset.ilrs:
        words_in_relation = get_words_in_synset_that_match(wn, id_of_synset_in_relation, words)
        for word_in_relation in words_in_relation:
            u = get_vertex_label(word)
            v = get_vertex_label(word_in_relation)
            graph.add_edge(u, v)
            edge_labels[(u, v)] = relation_name


def get_words_in_synset_that_match(wn, id_of_synset, words):
    synset = wn.lookUpID(id_of_synset, id_of_synset[-1:])
    words_in_synset = []
    for synonym in synset.synonyms:
        for word in words:
            if word[0] == synonym.literal and str(word[1]) == synonym.sense:
                words_in_synset.append(word)
    return words_in_synset


def find_max_distance(wn, pos, relation):
    graph = DiGraph()
    for synset_id, synset in wn.dat(pos).items():
        graph.add_node(synset_id)
        for id_of_synset_in_relation, relation_name in synset.ilrs:
            if relation_name == relation:
                graph.add_node(id_of_synset_in_relation)
                graph.add_edge(id_of_synset_in_relation, synset.wnid)
    return dag_longest_path_length(graph)


def measure_similarity(wn, word1, word2, relation_name, max_distance):
    graph1 = find_transitive_closure(wn, *word1, relation_name)[0]
    graph2 = find_transitive_closure(wn, *word2, relation_name)[0]
    graph = compose(graph1.to_undirected(), graph2.to_undirected())
    u = get_vertex_label(word1)
    v = get_vertex_label(word2)
    try:
        distance = shortest_path_length(graph, u, v)
    except NetworkXNoPath:
        distance = inf
    similarity = (-1.0) * log(distance / (2.0 * max_distance))
    return 'Leacock-Chodorow similiarity measure between {} and {} is {}. (distance: {}, max_distance: {})' \
        .format(u, v, similarity, distance, max_distance)


if __name__ == '__main__':
    main()
