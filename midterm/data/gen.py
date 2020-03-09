#import os
from itertools import permutations, combinations
import random


def read(file_path) -> list:
    file_iter = open(file_path, 'r')
    items = list()
    for line in file_iter:
        line = line.strip()
        # remove whitespace around items
        # record = tuple(sorted(trimmed))
        items.append(line)

    return items


def get_random_count():
    return random.randint(2, 6)


def generate_itemset():
    for r in range(0, 20):
        item_permutations = [out_list.append(i) for i in combinations(intput_file, get_random_count())]
        total_permutations = len(item_permutations)
        ic = random.randint(0, total_permutations)
        item_set = out_list[ic]
        print(sorted(item_set))


random.seed(3)


intput_file = read('item.csv')
out_list = list()
generate_itemset()



#item_permutations = [out_list.append(i) for i in combinations(intput_file, get_random_count())]
#total_permutations = len(item_permutations)
