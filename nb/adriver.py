import logging

from apriori import Apriori, CollapsedCsvFileReader
from assoc import (calculate_confidence, generate_associations,
                   get_support_for_key, generate_combo_itemsets, 
                   create_associations,  assocation_record)

logging.basicConfig()
logger = logging.getLogger('apriori')
logger.setLevel(logging.INFO)


#TODO: verify
#fr = CollapsedCsvFileReader('./data/test4.csv')
# OK
# fr = CollapsedCsvFileReader('./data/test-dm-bookch6.csv')
fr = CollapsedCsvFileReader('./data/test3.csv')

t2 = fr.read()

g = Apriori(t2)

support = 0.22
confidence = 0.0

output = g.generate_levels(support_level = support, drop_below_support=True)

pc = create_associations(output)

f = calculate_confidence(pc, confidence_level = confidence)

print(f)

print(len(f))