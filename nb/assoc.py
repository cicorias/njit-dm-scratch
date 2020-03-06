from typing import List, Tuple
from itertools import combinations, product, permutations
from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger("apriori")


#maybe make a data frame outbound..
def generate_associations(data: pd.DataFrame, threshold = 0.10):
    rv = list()
    for r in data.iterrows():
        # current row ID
        idx = r[0]
        item = data.iloc[idx]['itemsets']
        #print("at {0}  for {1}".format(idx, item))
        ##maybe make this a dict with key, support, and current itemset, associated itemsets
        all_other = [ k for k,v in data.iterrows() if k != idx]
        ##print("at {0}  for {1}".format(idx, item))
        ##print(all_other) # all other itemsets current belongs to...
        
        ## THIS current item support.
        support = data.iloc[idx]['support']
        
        ###rv = [ y for y in all_other ]
        ###rv = [ (item, support, output.iloc[y]['itemsets'], output.iloc[y]['itemsets']['support']) for y in all_other ]
        temp = [ (item, support, data.iloc[y]['itemsets'], data.iloc[y]['support']) 
                for y in all_other 
                if not set(item).issubset(data.iloc[y]['itemsets']) ]


        #todo: need another column that is the "combination" of the concatenation of the two tuples - antecedent and consequent (P(A|B))
        

        #print("type: {0}".format(type(temp)))
        rv.extend(temp)
        

        
    return rv

