from typing import List, Tuple
from itertools import combinations, product, permutations
from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
import pandas as pd
import logging


from apriori import CollapsedCsvFileReader, Apriori
from assoc import generate_associations

logging.basicConfig()
logger = logging.getLogger('apriori')
logger.setLevel(logging.INFO)

#fr = CollapsedCsvFileReader('./data/test4.csv')
fr = CollapsedCsvFileReader('./data/test-dm-bookch6.csv')

t2 = fr.read()

#logger.info("tran count: {0}".format(t2))
g = Apriori(t2)
#output = g.generate_levels(support_level=0.60, drop_below_support=False)
output = g.generate_levels(support_level=0.22, drop_below_support=True)


print(output)


nnn = generate_associations(output)

old_rows = pd.get_option("display.max_rows")
pd.set_option("display.max_rows", 200)


df = pd.DataFrame(nnn)
print(df)

df.to_csv('./out/out.csv')

pd.set_option("display.max_rows", old_rows)
