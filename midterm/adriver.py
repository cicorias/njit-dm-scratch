import logging

from py_apriori.apriori import Apriori, CollapsedCsvFileReader
from py_apriori.assoc import (calculate_confidence, generate_associations,
                   get_support_for_key, generate_combo_itemsets, 
                   create_associations,  assocation_record)

logging.basicConfig()
logger = logging.getLogger('apriori')
logger.setLevel(logging.WARN)


#TODO: verify
#fr = CollapsedCsvFileReader('./data/test4.csv')
# OK
fr = CollapsedCsvFileReader('./data/test-dm-bookch6.csv')

# TBD
#fr = CollapsedCsvFileReader('./data/test3.csv')
fr = CollapsedCsvFileReader('./data/test_data.csv')


# first read in a file that creates a list
t2 = fr.read()

# instantiate the Apriori class with the reader
g = Apriori(t2)

# setup for output
support = 0.22
confidence = 0.0

#just generate the levels and filter as needed
output = g.generate_levels(support_level = support, drop_below_support=True)

# create the associations
#TODO: mabye encapsulate this step.
pc = create_associations(output)
#print(pc)


# generate the confidence levels.
f = calculate_confidence(pc, confidence_level = confidence)

print(f)

print(len(f))