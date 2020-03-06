import logging

from apriori import Apriori, CollapsedCsvFileReader
from assoc import (calculate_confidence, generate_associations,
                   get_support_for_key, generate_combo_itemsets, 
                   create_associations,  myRecord)

logging.basicConfig()
logger = logging.getLogger('apriori')
logger.setLevel(logging.INFO)


#fr = CollapsedCsvFileReader('./data/test4.csv')
fr = CollapsedCsvFileReader('./data/test-dm-bookch6.csv')

t2 = fr.read()

g = Apriori(t2)

output = g.generate_levels(support_level=0.22, drop_below_support=True)

pc = create_associations(output)

f = calculate_confidence(pc, confidence_level = 0.0)

print(f)