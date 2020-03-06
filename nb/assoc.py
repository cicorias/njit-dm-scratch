from typing import List, Tuple
from collections import namedtuple

import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger("apriori")


assocation_record = namedtuple('assocation_record', ['full_key', 'predecessor', 
            'support1', 'result', 'support2', 'support_full_key', 'confidence'])


def create_associations(data: pd.DataFrame) -> pd.DataFrame:
    n2 = generate_associations(data)
    pc = generate_combo_itemsets(n2)
    return pc

def generate_associations(data: pd.DataFrame) -> pd.DataFrame:
    rv = list()
    #TODO: refactor itertuples
    for r in data.iterrows():
        # current row ID
        idx = r[0]
        item = data.iloc[idx]['itemsets']
        #all_other is everything BUT the current key.
        all_other = [ k for k,v in data.iterrows() if k != idx]
        ## THIS current item support.
        support = data.iloc[idx]['support']
        
        temp = [ (item, support, data.iloc[y]['itemsets'], data.iloc[y]['support']) 
                for y in all_other 
                if not set(item).issubset(data.iloc[y]['itemsets']) ]

        rv.extend(temp)

    return pd.DataFrame(rv, columns= ['predecessor', 'support1', 'result', 'support2'])



def generate_combo_itemsets(data: pd.DataFrame) -> pd.DataFrame:
    possible_combos = [ ( i[1]['predecessor'] + i[1]['result'], 
                    i[1]['predecessor'], i[1]['support1'], i[1]['result'], i[1]['support2'] ) 
                    #TODO: refactor itertuples
                    for i in data.iterrows()
                    #elimiate where item on boths sides 
                    if not bool(set(i[1]['result']) & set(i[1]['predecessor'])) ]

    return pd.DataFrame(possible_combos, columns= ['fullkey', 'predecessor', 'support1', 'result', 'support2'])
 

def get_support_for_key(data: pd.DataFrame, predecessor_key: Tuple) -> float:
    rv = 0
    srted_key = tuple(sorted(predecessor_key))
    matches = data[data['predecessor'] == srted_key].head(1)
    if len(matches) ==  1:
        rv =  matches['support1'].array[0]
    
    return rv


def calculate_confidence(data: pd.DataFrame, confidence_level: float = 0.0) -> pd.DataFrame:
    rv = list()
    for r in data.itertuples():
        full_key = r.fullkey
        ant = r.predecessor
        support1 = r.support1
        res = r.result
        support2 = r.support2
        support_full_key = get_support_for_key(data, full_key)
        if support1 != 0:
            confidence = support_full_key / support1
        else:
            confidence = -1
            
        item_rv = assocation_record(full_key, ant, support1, res, support2, support_full_key, confidence )
        
        rv.append( item_rv )
        
 
    rv_df = pd.DataFrame(rv)
    return rv_df[rv_df.confidence > confidence_level] 
