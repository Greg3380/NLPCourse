import os
from collections import Counter

import regex


class Date(object):
    def __init__(self, day, month, year):
        self.day = day
        self.month = month
        self.year = year


class Regulation(object):
    def __init__(self, title, regulation_year, regulation_number, regulation_poz):
        self.title = title
        self.regulation_year = regulation_year
        self.regulation_number = regulation_number
        self.regulation_poz = regulation_poz

    def set_title(self, title):
        self.title = title


def build_reference_name(reference):
    return str(reference.regulation_year) + " position: " +  reference.regulation_poz + " journal number: " + str(reference.regulation_number)

def split(txt, seps):
    if len(seps) == 0:
        return txt
    default_sep = seps[0]

    # we skip seps[0] because that's the default seperator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]


def extract_number(string):
    res = regex.findall(r'\d+', string)
    return int(res[0])


def match_old_fashioned_case(string, date):
    references = split_string_by_year(string, date)
    result = []
    for ref in references:
        num_and_pos = find_jounal_number_and_position(ref[1])
        result += [(ref[0],) + entry for entry in num_and_pos]
    return result


def split_string_by_year(string, date):
    pattern = r"\d\d\d\d\s*r.*?"
    res = regex.split(pattern, string)
    dates = regex.findall(pattern, string, regex.IGNORECASE | regex.MULTILINE | regex.DOTALL)
    dates = list(map(extract_number, dates))
    if len(dates) != len(res):
        dates = [date.year] + dates
    return list(zip(dates, res))


def find_jounal_number_and_position(string):
    pattern = r"(nr[\s|.]+\d+)"
    journal_numbers = regex.findall(pattern, string, regex.IGNORECASE | regex.MULTILINE | regex.DOTALL)
    positions = split(string, journal_numbers)
    if len(journal_numbers) < len(positions):
        positions = positions[1:]

    result = []
    for pos, journal_number in zip(positions, journal_numbers):
        result += find_positions(pos, journal_number)

    return result


def find_positions(string, number):
    journal_number = extract_number(number)
    position_pattern = r'(\d+)'
    positions = regex.findall(position_pattern, string, regex.IGNORECASE | regex.MULTILINE | regex.DOTALL)
    result = [tuple([journal_number, pos]) for pos in positions]
    return result


def match_modern_case(string):
    modern_pattern = r'(?<=(\d\d\d\d)\s*r.*?)(?<=poz)*?(\d+)'  # ten pattern jest ok (positive lookbehind)
    results = regex.findall(modern_pattern, string,
                            regex.IGNORECASE | regex.MULTILINE | regex.DOTALL)  # TODO case gdy nie ma roku w referencji
    return [(res[0],) + (None,) + (res[1],) for res in results]


def match_number_and_position_in_journal(string, date, title):
    if regex.search('nr', string, regex.IGNORECASE):
        references = match_old_fashioned_case(string, date)
    else:
        references = match_modern_case(string)
    res = [Regulation('', ref[0], ref[1], ref[2]) for ref in references]
    if len(res) != 0:
        res[0].set_title(title)
    return res


def match_date(string):
    pattern = r'(\d+)\s+(\p{L}+)\s+(\d+)'
    res = regex.findall(pattern, string, regex.IGNORECASE | regex.MULTILINE | regex.DOTALL)
    if len(res) == 0:
        return Date(-1, -1, -1)
    return Date(extract_number(res[0][0]), res[0][1], extract_number(res[0][2]))


def find_all_external_references(bill):
    pattern = r'ustaw(?:a|y|ie|ę|ą|o||om|ami|ach)((?:.){1,200})(\([^()]*?(?:poz)[^()]*\))'
    bills = regex.findall(pattern, bill, regex.IGNORECASE | regex.MULTILINE | regex.DOTALL)
    result = []
    for bill in bills:

        if not regex.findall("poz", bill[1], regex.IGNORECASE | regex.MULTILINE | regex.DOTALL):
            continue
        title = bill[0]
        date = match_date(title)
        result += match_number_and_position_in_journal(bill[1], date, title)
    return result


path = os.path.join("..", "ustawy", "1995_2.txt")
# with open(path, 'r', encoding="utf8") as fin:
#     find_all_external_references(fin.read())

directory_path = os.path.join("..", "ustawy")

for root, dirs, files in os.walk(directory_path):
    result = []
    for filename in files:
        with open(os.path.join(directory_path, filename), 'r', encoding="utf8") as fin:
            print("Bill name: " + filename + "\n\n")
            references = find_all_external_references(fin.read())
            if len(references) != 0:
                result += references

    reference_dict = Counter()
    for ref in result:
        reference_dict[build_reference_name(ref)] += 1
    with open(os.path.join("..", "output", "exercise1_1"), 'w+', encoding="utf8") as fout:
        for ref, occurrences in reference_dict.most_common():
            fout.write(ref + " Count: " + str(occurrences) + "\n")
