import os
from collections import Counter

import regex


def build_bill_name(bill):
    return "ust. " + bill


def build_article_name(article, bill):
    return "art. " + article + " ust. " + bill


class Bill(object):
    def __init__(self, filename, references):
        self.filename = filename
        self.references = references
        self.total_references = 0
        self.bill_dict = None
        self.article_dict = None

    def group_and_count_references(self):
        bill_dict = Counter()
        article_dict = Counter()
        for article, bill in self.references:
            if not article:
                name = build_bill_name(bill)
                bill_dict[name] += 1
            else:
                name = build_article_name(article, bill)
                article_dict[name] += 1
        bill_dict.most_common()
        article_dict.most_common()
        self.total_references = sum(bill_dict.values()) + sum(article_dict.values())
        self.bill_dict = bill_dict
        self.article_dict = article_dict


def find_occurrences(bill):
    pattern = r'(?:art(?:i| |\.)+(\d+))?(?:\.| |i|w|z)+ust(?:i| |\.)+(\d+)'
    res = regex.findall(pattern, bill, regex.MULTILINE | regex.DOTALL)
    return res


directory_path = os.path.join("..", "ustawy")

for root, dirs, files in os.walk(directory_path):
    bill_list = []
    for filename in files:
        with open(os.path.join(directory_path, filename), 'r', encoding="utf8") as fin:
            print("Bill name: " + filename + "\n")
            res = find_occurrences(fin.read())
            bill = Bill(filename, res)
            bill.group_and_count_references()
            bill_list.append(bill)
    bill_list.sort(key=lambda x: x.total_references, reverse=True)

    with open(os.path.join("..","output","exercise1_2"), 'w+', encoding="utf8") as fout:
        for bill in bill_list:
            fout.write("Bill filename: " + bill.filename + "\n")
            fout.write("Total references: " + str(bill.total_references) + "\n\n")
            fout.write("References in format art. <article_number> ust. <bill_number>" + "\n")
            for art_ref, occurrences in bill.article_dict.most_common():
                fout.write(art_ref + " Count: " + str(occurrences) + "\n")
            fout.write("References in format ust. <number_of_bill>" + "\n")
            for bill_ref, occurrences in bill.bill_dict.most_common():
                fout.write(bill_ref + " Count: " + str(occurrences) + "\n")

