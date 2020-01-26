from typing import Tuple

class Apriori:
# following https://en.wikipedia.org/wiki/Apriori_algorithm#Examples
    def __init__(self, transactions: Tuple):
        if transactions is None:
            raise ValueError("Transaction itemset is none")

        if not isinstance(transactions, Tuple):
            raise ValueError("Transaction itemset is not a Tuple")
        
        self._itemset_transactions = transactions

    def __str__(self):
        return str(tuple(self))


    @property
    def transactions(self):
        return self._itemset_transactions
    

    @transactions.setter
    def transactions(self, value):
        if not value:
            raise ValueError("Cannot set transactions to None")

        if not isinstance(value, Tuple):
            raise ValueError("Cannot set transactions - must be a Tuple")        

        self._itemset_transactions = value



