import os

import regex

directory_path = os.path.join("..", "ustawy")


def find_occurrences(bill):
    pattern = r'(\bustaw(?:a|y|ie|ę|ą|o||om|ami|ach))'
    res = regex.findall(pattern, bill, regex.IGNORECASE | regex.MULTILINE | regex.DOTALL)
    return len(res)


for root, dirs, files in os.walk(directory_path):
    counter = 0
    for filename in files:
        with open(os.path.join(directory_path, filename), 'r', encoding="utf8") as fin:
            print("Bill name: " + filename + "\n")
            occurrences = find_occurrences(fin.read())
            counter += occurrences
    print("Number of occurrences:")
    print(counter)
    with open(os.path.join("..", "output", "exercise1_3"), 'w+', encoding="utf8") as fout:
            fout.write("Number of occurrences:\n")
            fout.write(str(counter))