from typing import List, Tuple
from itertools import combinations
from abc import ABC, abstractmethod
import pandas as pd
import logging

logging.basicConfig()
logger = logging.getLogger("apriori")


class Apriori(object):
    # following https://en.wikipedia.org/wiki/Apriori_algorithm#Examples
    def __init__(self, transactions: List):
        self.__verify__(transactions)

        self._transactions = transactions

    def __str__(self):
        return str(tuple(self))

    def __verify__(self, transactions):
        if transactions is None:
            raise ValueError("Transaction itemset is none")

        if not isinstance(transactions, List):
            raise ValueError("Transaction itemset is not a List")

        if len(transactions) == 0:
            raise ValueError("Transaction is empty")

        if len(transactions) > 0 and not isinstance(transactions[0], Tuple):
            raise ValueError("Transaction lement is not a Tuple")

    @property
    def transactions(self) -> List:
        return self._transactions

    @transactions.setter
    def transactions(self, value: List):

        self.__verify__(value)
        self._transactions = value

    def generate_levels(self, support_level: float = 0.20, drop_below_support: bool = True) -> pd.DataFrame:
        k = 1
        full_set = list()  # this contains a dataframe for each level.
        while True:
            logger.info("k = {0}".format(k))
            item_levels = self.__generate_combinination_levels(self._transactions, k)
            sl = self.__gen_support_level(self._transactions, item_levels,
                                          support=support_level, drop=drop_below_support)

            # logger.debug("transactions at level {k} are {n}".format(k = k, n = (len(sl))))
            k += 1

            if len(sl) == 0 or k == 100:
                break

            df = pd.DataFrame.from_dict(sl, orient='index', columns=['count'])
            df.index.name = 'itemsets'
            df.reset_index()
            full_set.append(df)

        rv = self.__append_colums(full_set)
        return rv

    def __append_colums(self, data: List, tran_list=None) -> pd.DataFrame:
        if tran_list is None:
            tran_list = self.transactions

        tran_count = len(tran_list)

        rows_list = []
        for r in data:
            # logger.debug('type of r is: {0}'.format(type(r)))
            # logger.debug('len of r is: {0}'.format(len(r)))
            # logger.debug('r is: {0}'.format(r))
            for index, row in r.iterrows():
                # d = { 'count' : r['count'], 'support': r['count']/tran_count}
                d = {'itemsets': index, 'count': row['count'], 'support': row['count']/tran_count}
                # logger.debug("THE DICTd: {0}".format(d))
                rows_list.append(d)

        df = pd.DataFrame(rows_list)

        return df

    def __generate_combinination_levels(self, tran_list, level):
        """generate keys that are used for subset checking"""
        """on each transaction"""
        results = list()
        for t in tran_list:
            logger.debug("gen_com_levell: t: {0}  and level: {1}".format(t, level))
            [results.append(i) for i in combinations(t, level)]

        rv = sorted(set(results))
        logger.debug("combo levels: {0}".format(rv))
        return rv

    def __gen_support_level(self, tran_list, items_keys, support=0.20, drop=True):
        """for each key which can be a set find in transactions"""
        """how many contain the combination"""
        logger.info('Using support level of {0}'.format(support))
        logger.info('drop below support? {0}'.format(drop))
        tran_count = len(tran_list)
        base_level = tran_count * support
        logger.debug('base level count: {0}'.format(base_level))
        itemSet = dict()

        for key in items_keys:
            for t in tran_list:
                if set(key).issubset(t):
                    # logger.debug('is subset: {0}'.format(t))
                    if (key) in itemSet:
                        itemSet[key] += 1
                    else:
                        itemSet[key] = 1

        if drop:
            return {key: value for (key, value) in itemSet.items() if value >= base_level}
        else:
            return {key: value for (key, value) in itemSet.items()}


class FileReader(ABC):
    def __init__(self, file_path):
        self.file_path = file_path

    @abstractmethod
    def read(self) -> list:
        pass


class CollapsedCsvFileReader(FileReader):
    """the file format is lines, with individual transactinos"""
    """separated by commma - thus calling this collapsed"""
    """file format as it is non-traditional"""

    def read(self) -> list:
        file_iter = open(self.file_path, 'r')
        raw_transactions = list()
        for line in file_iter:
            line = line.strip().rstrip(',')
            # remove whitespace around items
            trimmed = [i.strip() for i in line.split(',')]
            record = tuple(sorted(trimmed))
            raw_transactions.append(record)

        return raw_transactions
