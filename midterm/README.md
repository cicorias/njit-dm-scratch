# Py_Apriori Module

## Setup
First download the TAR file to a location that you will use. I recomend a scratch directory that you can remove all when done.

In the text below, my extracted directory is: `/c/temp/py-apriori-0.1.0`

### Setup your Python environment
General recomendation is to use a Python Virtual Environment. With Pythhon `3.5+` execute the following:


#### Create virtual env and activate
```
# from /c/temp/
python -m venv env  # this creates directory env
. ./env/bin/activate

```

### Unpack the tar file

From the directory where the virtual env and the tar file is:

```
tar -xvf py-apriori-0.1.0.tar.gz
cd py-apriori-0.1.0.tar.gz

```

### Run setup

Once you've extracted the tar file, and changed to the tar output directory, now run `setup.py install` under python to add all dependencies

```
# from /c/temp/py-apriori-0.1.0

python setup.py install

```

At this point the program is ready to run, and a Test data file is present in `./data/`

## Running

The program makes use of argument parsing and all arguments can be seen by running the following:

```
# from /c/temp/py-apriori-0.1.0

python main.py -h

# ---
usage: main.py [-h] -i FILE [-c CONFIDENCE_LEVEL] [-s SUPPORT_LEVEL] [-n]
               [-o FILE]

implementation of the Apriori Algorithm

optional arguments:
  -h, --help            show this help message and exit
  -i FILE, --input FILE
                        input transaction file collapsed CSV format
  -c CONFIDENCE_LEVEL, --confidence CONFIDENCE_LEVEL
                        confidence level for association generation see https://en.wikipedia.org/wiki/Association_rule_learning#Confidence
  -s SUPPORT_LEVEL, --support SUPPORT_LEVEL
                        support level for support generation see https://en.wikipedia.org/wiki/Association_rule_learning#Support
  -n, --no-drop         DO NOT drop transactions below support level
  -o FILE, --output FILE
                        output file
```

### Sample run

The extracted TAR file has a sample input file in `./data` -- to run:

```
# from /c/temp/py-apriori-0.1.0
python main.py -i data/data.csv
```

#### Sample Run Output

```
cicorias@cicoria-msi:/c/temp/py-apriori-0.1.0$ python main.py -i data/data.csv
For this run we are using the following

        Support: 0.2
        Confidence: 0.8
        Drop Trans: True
        File:       /c/temp/py-apriori-0.1.0/data/data.csv


=== SUPPORT LEVELS ===

        itemsets  count   support
0          (I1,)      6  0.666667
1          (I2,)      7  0.777778
2          (I3,)      6  0.666667
3          (I4,)      2  0.222222
4          (I5,)      2  0.222222
5       (I1, I2)      4  0.444444
6       (I1, I3)      4  0.444444
7       (I1, I5)      2  0.222222
8       (I2, I3)      4  0.444444
9       (I2, I4)      2  0.222222
10      (I2, I5)      2  0.222222
11  (I1, I2, I3)      2  0.222222
12  (I1, I2, I5)      2  0.222222


=== ASSOCIATION AND CONFIDENCE LEVELS ===

        full_key predecessor  support1    result  support2  support_full_key  confidence
23      (I4, I2)       (I4,)  0.222222     (I2,)  0.777778          0.222222         1.0
33      (I5, I1)       (I5,)  0.222222     (I1,)  0.666667          0.222222         1.0
34      (I5, I2)       (I5,)  0.222222     (I2,)  0.777778          0.222222         1.0
37  (I5, I1, I2)       (I5,)  0.222222  (I1, I2)  0.444444          0.222222         1.0
50  (I1, I5, I2)    (I1, I5)  0.222222     (I2,)  0.777778          0.222222         1.0
64  (I2, I5, I1)    (I2, I5)  0.222222     (I1,)  0.666667          0.222222         1.0

```

### Sample Run Data File

```csv
I1, I2, I5
I2, I4
I2, I3
I1, I2, I4
I1, I3
I2, I3
I1, I3
I1, I2, I3, I5
I1, I2, I3

```
## Data Format

The data file is in a simple format that I call **Collapses CSV** as each line has multiple transaction items separated by a comma. So, it's not exactly a CSV file, but close.



# References:
This is a basic implementation of the Apriori Algorithm[1]

> [Google Scholar](https://scholar.google.com/scholar?q=R.C.%20Agarwal%2C%20C.C.%20Aggarwal%2C%20and%20V.V.V.%20Prasad.%20Depth%20first%20generation%20of%20long%20patterns.%20In%20Proc.%20of%20the%206th%20ACM%20SIGKDD%20Int.%20Conf.%20on%20Knowledge%20Discovery%20and%20Data%20Mining%2C%20pages%20108%E2%80%93118%2C%20Boston%2C%20MA%2C%20USA%2C%202000.) - Agrawal, Rakesh, Tomasz Imieliński, and Arun Swami. "Mining association rules between sets of items in large databases." Proceedings of the 1993 ACM SIGMOD international conference on Management of data. 1993.


