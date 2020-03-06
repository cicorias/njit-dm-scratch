from typing import List, Tuple
#from itertools import combinations #, product, permutations
#from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
from collections import namedtuple

import pandas as pd

import logging

logging.basicConfig()
logger = logging.getLogger("apriori")


myRecord =  namedtuple('myRecord', ['full_key', 'antecedent', 'support1', 'result', 'support2', 'support_full_key', 'confidence'])


def create_associations(data: pd.DataFrame):
    n2 = generate_associations(data)
    pc = generate_combo_itemsets(n2)
    return pc

#maybe make a data frame outbound..
def generate_associations(data: pd.DataFrame, threshold = 0.10):
    #TODO: use threshold to filter below
    rv = list()
    for r in data.iterrows():
        # current row ID
        idx = r[0]
        item = data.iloc[idx]['itemsets']
        ##maybe make this a dict with key, support, and current itemset, associated itemsets
        #all_other is everything BUT the current key.
        all_other = [ k for k,v in data.iterrows() if k != idx]
        ## THIS current item support.
        support = data.iloc[idx]['support']
        
        temp = [ (item, support, data.iloc[y]['itemsets'], data.iloc[y]['support']) 
                for y in all_other 
                if not set(item).issubset(data.iloc[y]['itemsets']) ]

        #TODO: need another column that is the "combination" of the concatenation of the two tuples - antecedent and consequent (P(A|B))
        rv.extend(temp)

    return pd.DataFrame(rv, columns= ['antecedent', 'support1', 'result', 'support2'])



def generate_combo_itemsets(data: pd.DataFrame):
    possible_combos = [ ( i[1]['antecedent'] + i[1]['result'], i[1]['antecedent'], i[1]['support1'], i[1]['result'], i[1]['support2'] ) 
    for i in data.iterrows()
    #elimiate where item on boths sides 
    if not bool(set(i[1]['result']) & set(i[1]['antecedent'])) ]

    return pd.DataFrame(possible_combos, columns= ['fullkey', 'antecedent', 'support1', 'result', 'support2'])
 



def get_support_for_key(trans_list: pd.DataFrame, antecedent_key: Tuple):
    rv = 0
    srted_key = tuple(sorted(antecedent_key))
    matches = trans_list[trans_list['antecedent'] == srted_key].head(1)
    if len(matches) ==  1:
        rv =  matches['support1'].array[0]
    
    return rv


def calculate_confidence(data: pd.DataFrame, confidence_level = 0.0):
    rv = list()
    for r in data.itertuples():
        full_key = r.fullkey
        ant = r.antecedent
        support1 = r.support1
        res = r.result
        support2 = r.support2
        support_full_key = get_support_for_key(data, full_key)
        if support1 != 0:
            confidence = support_full_key / support1
        else:
            confidence = -1
            
        item_rv = myRecord(full_key, ant, support1, res, support2, support_full_key, confidence )
        
        rv.append( item_rv )
        
 
    rv_df = pd.DataFrame(rv)
    return rv_df[rv_df.confidence > confidence_level] 
