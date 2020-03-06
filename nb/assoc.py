from typing import List, Tuple
from itertools import combinations, product, permutations
from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger("apriori")


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

    return rv

def generate_basket_column(data: pd.DataFrame):
    pass