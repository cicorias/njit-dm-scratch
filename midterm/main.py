import logging

from py_apriori.apriori import Apriori, CollapsedCsvFileReader
from py_apriori.assoc import (calculate_confidence, generate_associations,
                   get_support_for_key, generate_combo_itemsets, 
                   create_associations,  assocation_record)

logging.basicConfig()
logger = logging.getLogger('apriori')
logger.setLevel(logging.WARN)

