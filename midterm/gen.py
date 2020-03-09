from itertools import combinations
import random


def read(file_path) -> list:
    file_iter = open(file_path, 'r')
    items = list()
    for line in file_iter:
        line = line.strip()
        items.append(line)

    return items


def get_random_count():
    return random.randint(2, 5)


def generate_itemset(input_file):
    rv = list()
    out_list = list()
    for r in range(0, 20):
        item_permutations = [out_list.append(i) for i in combinations(input_file, get_random_count())]
        total_permutations = len(item_permutations)
        ic = random.randint(0, total_permutations)
        item_set = out_list[ic]
        rv.append(sorted(item_set))

    return rv


def generate_db_file(input_file, output_file):
    file1 = generate_itemset(input_file)
    with open(output_file, "w") as outfile:
        all_buffer = ""
        for item in file1:
            buffer = ""
            for i in item:
                buffer += i + ","

            buffer.strip().rstrip(',')
            all_buffer += buffer + '\n'

        outfile.writelines(all_buffer)


random.seed(2020)
input_file = read('./data/item.csv')

generate_db_file(input_file, './data/db_file1.csv')
generate_db_file(input_file, './data/db_file2.csv')
generate_db_file(input_file, './data/db_file3.csv')
generate_db_file(input_file, './data/db_file4.csv')
generate_db_file(input_file, './data/db_file5.csv')