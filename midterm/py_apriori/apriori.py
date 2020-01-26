from typing import List, Tuple

class Apriori:
# following https://en.wikipedia.org/wiki/Apriori_algorithm#Examples
    def __init__(self, transactions: List):
        self.__verify__(transactions)

        self._itemset_transactions = transactions

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
    def transactions(self):
        return self._itemset_transactions
    

    @transactions.setter
    def transactions(self, value):

        self.__verify__(value)
        self._itemset_transactions = value



