

https://www.geeksforgeeks.org/python-check-if-a-list-is-contained-in-another-list/

https://stackoverflow.com/questions/3931541/how-to-check-if-all-of-the-following-items-are-in-a-list

https://thispointer.com/python-check-if-a-list-contains-all-the-elements-of-another-list/

https://www.digitalvidya.com/blog/apriori-algorithms-in-data-mining/



## Apriori algorithm – The Theory
Three significant components comprise the apriori algorithm. They are as follows.
- Support
- Confidence
- Lift

This example will make things easy to understand.

As mentioned earlier, you need a big database. Let us suppose you have 2000 customer transactions in a supermarket. You have to find the Support, Confidence, and Lift for two items, say bread and jam. It is because people frequently bundle these two items together.

Out of the 2000 transactions, 200 contain jam whereas 300 contain bread. These 300 transactions include a 100 that includes bread as well as jam. Using this data, we shall find out the support, confidence, and lift.

### Support
Support is the default popularity of any item. You calculate the Support as a quotient of the division of the number of transactions containing that item by the total number of transactions. Hence, in our example,

```
Support (Jam) = (Transactions involving jam) / (Total Transactions)

        = 200/2000 = 10%
```

### Confidence
In our example, Confidence is the likelihood that customers bought both bread and jam. Dividing the number of transactions that include both bread and jam by the total number of transactions will give the Confidence figure.

```
Confidence = (Transactions involving both bread and jam) / (Total Transactions involving jam)

        = 100/200 = 50%
```

It implies that 50% of customers who bought jam bought bread as well.


### Lift
According to our example, Lift is the increase in the ratio of the sale of bread when you sell jam. The mathematical formula of Lift is as follows.

```
Lift = (Confidence (Jam͢͢ – Bread)) / (Support (Jam))

      = 50 / 10 = 5
```

It says that the likelihood of a customer buying both jam and bread together is 5 times more than the chance of purchasing jam alone. If the Lift value is less than 1, it entails that the customers are unlikely to buy both the items together. Greater the value, the better is the combination.

![](https://image.slidesharecdn.com/apriorialgorithm-140619035225-phpapp02/95/apriori-algorithm-8-638.jpg?cb=1403150201)

How does the Apriori Algorithm in Data Mining work?
We shall explain this algorithm with a simple example.

Consider a supermarket scenario where the itemset is I = {Onion, Burger, Potato, Milk, Beer}. The database consists of six transactions where 1 represents the presence of the item and 0 the absence.

![](https://www.digitalvidya.com/wp-content/uploads/2018/11/01.jpg)

#### The Apriori Algorithm makes the following assumptions.
All subsets of a frequent itemset should be frequent.
In the same way, the subsets of an infrequent itemset should be infrequent.
Set a threshold support level. In our case, we shall fix it at 50%

#### Step 1
Create a frequency table of all the items that occur in all the transactions. Now, prune the frequency table to include only those items having a threshold support level over 50%. We arrive at this frequency table.

![](https://www.digitalvidya.com/wp-content/uploads/2018/11/02.jpg)


This table signifies the items frequently bought by the customers.

#### Step 2
Make pairs of items such as OP, OB, OM, PB, PM, BM. This frequency table is what you arrive at.

![](https://www.digitalvidya.com/wp-content/uploads/2018/11/02.jpg)


#### Step 3
Apply the same threshold support of 50% and consider the items that exceed 50% (in this case 3 and above).

Thus, you are left with OP, OB, PB, and PM

#### Step 4
Look for a set of three items that the customers buy together. Thus we get this combination.

OP and OB gives OPB
PB and PM gives PBM
#### Step 5

![](https://www.digitalvidya.com/wp-content/uploads/2018/11/04.jpg)

Determine the frequency of these two itemsets. You get this frequency table.

If you apply the threshold assumption, you can deduce that the set of three items frequently purchased by the customers is OPB.

We have taken a simple example to explain the apriori algorithm in data mining. In reality, you have hundreds and thousands of such combinations.  