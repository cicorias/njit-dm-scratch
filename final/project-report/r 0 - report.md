# TODO:

- DT work
  - complete the k fold analysis or the ROC/AUC curve
    - https://stackoverflow.com/questions/35097003/cross-validation-decision-trees-in-sklearn
  - explain why # of levels chosen
  - kill the reload scripts
  - save all files as py
    - add snipped to check if in script or notebook
    - if in script print to files.
  

# 1 - Introduction
The goal, repeated here, is to be able to predict a "good wine" based upon its physical and chemical properties.

> NOTE: In speaking with the Instructor -- **Yasser Abduallah** -- that given the thin, essentially lack of Python implementation of **C4.5** or **ID3** that instead the **CART** algorithm would be used. 

# 2 - Methodology and Approach

For this project we are using two non-parametric classifiers to analyze and learn from the data. Both Support Vector Machines and Decision Trees adapt to the data and learn the features or parameters as it evolves in training. This is contrasted to other models, such as Linear or Logistic that have a set number of parameters before the training begins. While we look at each of the features in the Wine data set as "parameters", from the non-parametric perspective, they are not fixed, but learned from the data, thus is sticks to the data. This can cause over-fitting with lots of data or features. In Decision Tree analysis, for example, we would limit the number of "decisions" or levels as going through each and every possible outcome can cause over-fitting to the training data.


## 2.1 - Classification

Classification uses the data to determine for each of the observations what class or outcome it might or should be. The classes or labels can often be binary (just two possible outcomes) or multi-class where there are more than 2 possible outcomes.  For example in binary - Male or Female, Rain or No-Rain, etc.

In this report we are using a data set that starts with a **quality** class that ranks wine from 0-9, with 9 being the highest quality, and 0 the lowest. The ranking is done by humans thus subjective to each of the taste testers abilities. However, for this analysis, the data is converted to a binary outcome, which is explained later, but intended to simplify the prediction of if the wine is **good** or **not-good**.  The reasons for this is simplification, reduce training time, and as seen with the data, the quality label itself is not a great distribution (normal) and has groupings of data itself clustered in a low range and a high range. The transformation of the data is discussed later. 

## 2.2 - Support Vector Machine
This report uses an implementation of Support Vector Machines (SVM) from [Scikit learn](https://scikit-learn.org/stable/modules/svm.htmlhttps://scikit-learn.org/stable/modules/svm.html). This is a well used and one of the most popular Python frameworks for Data Mining, Machine Learning, Statistical processing. It is kept up-to-date and aligns with latest versions of Python including Python 3.8.

### 2.2.1 - Linear vs non-Linear Data
In using Support Vector Machines with a linear classified the data must be linear separable. That means each of the feature to feature mappings must present the actual class that is clearly separated by a linear plane. While SVM with Radial Bias Function (RBF) can accommodate non-linear data we will look towards taking our Data and modifying it for easier analysis.

In this report, RBF is used, as shown later due to the non-linear separability of the data. RBF is also explained in greater detail in that section.

## 2.3 - Decision Tree
Along with SVM, a Decision Tree study is also done.  

Initially the goal was to use the [C4.5](https://en.wikipedia.org/wiki/C4.5_algorithm) or even  [ID3](https://en.wikipedia.org/wiki/ID3_algorithm) which are both relatively well known from [Ross Quinlan](https://en.wikipedia.org/wiki/Ross_Quinlan) and date back to 1995 and 1965 respectively. While the author of those two algorithms has created new versions, such as C5.0, but they have been commercialized. Additionally, the C4.5 algorithm has been well implemented in [Weka as J48](https://en.wikipedia.org/wiki/Weka_(machine_learning)).

Unfortunately, while there are many attempted efforts in Open Source for a Python version of C4.5 or ID3, of all the modules researched none were found to be stable enough for use in this report. Many were just simple steps of the algorithm, not a library for re-use.

> NOTE: In speaking with the Instructor -- **Yasser Abduallah** -- that given the thin, essentially lack of Python implementation of **C4.5** or **ID3** that instead the **CART** algorithm would be used. 

### 2.3.1 - CART Decision Tree
The Classification and Regression Trees (CART) algorithm is implemented in [Scikit](https://scikit-learn.org/stable/modules/tree.html) learn as the only Decision Tree algorithm. CART can work with features that are either continuous or categorical. Where as C4.5 and ID3 both require categorical features. C4.5 can handle continuous features, but underneath it creates bins or thresholds -- essentially categories to accommodate.

The CART implementation can work with both binary or multi-class output as well. For features it uses Regression and minimization of mean squared error (MSE) and mean absolute error (MSA).  For Categorical features uses base classification with either **entropy** or **gini** for the impurity measures among the features for each decision tree node.  The germ **gini** is named for the statistician [Corrado Gini](https://en.wikipedia.org/wiki/Corrado_Gini) that devised the measure for gauging economic inequality.

At splitting time, information gain is used based upon the outcome of the Regression or the Classification impurity results for the features.

### 2.3.2 - Python ID3/C4.5 Implementations Reviewed and Rejected

Below are some of the Python code bases I took a look at for a stable ID3 or C4.5 implementation. As stated before, most didn't essentially work and root cause of failure varied from changes in underlying dependencies, bad algorithms in that it only worked with the supplied test data and not general data, some still required Python 2.7 which for this report Python 3.7+ is used, and some even had hard coding.

* Source code "almost" works but have bugs remaining that have not figured out yet; not very robust: [Tree algorithms: ID3, C4.5, C5.0 and CART - Data Driven Investor - Medium](https://medium.com/datadriveninvestor/tree-algorithms-id3-c4-5-c5-0-and-cart-413387342164)
* ChefBoost - has bugs now, can't predict and intermittent issues with Rules are invalid Python (this emits a Rules.;py file --- I have it almost working when I reclassified as categories vs. continous. It defaults to Regression if the features are numerical. [chefboost/chefboost at master · serengil/chefboost](https://github.com/serengil/chefboost/tree/master/chefboost)
  * [serengil/chefboost: Lightweight Decision Trees Framework supporting Gradient Boosting (GBDT, GBRT, GBM), Random Forest and Adaboost w/categorical features support for Python](https://github.com/serengil/chefboost)
* Poorly written and Python 2.7 - [ryanmadden/decision-tree: C4.5 Decision Tree python implementation with validation, pruning, and attribute multi-splitting](https://github.com/ryanmadden/decision-tree)
* Bugs and written for Python 2.7 - [NFG: barisesmer/C4.5: A python implementation of C4.5 algorithm by R. Quinlan](https://github.com/barisesmer/C4.5)


#### 3.3.2.1 - Experimentation Notebooks

For completeness, the Jupyter notebooks used in the evaluation and experimentation with these libraries is packed along with the rest of the project in the archive file.

# 3 - Analysis and Modeling
# 3 - Analysis and Modeling
First a discussion of the data utilized for this report. The data set is provided in two different `csv` files. 

The source of the file is from [UC Irvine Machine Learning Repository - Wine Quality Data Set](https://archive.ics.uci.edu/ml/datasets/wine+quality).

The two files and the full description are contained in **Appendix A** but the relevant parts needed for this initial EDA are detailed below.

Note that the **output** variable of **quality** is based upon **sensor** data. That means a human has interpreted the Wine and classified the level of quality. Soon we will see those levels of quality - or the **labels** that each wine can be assigned based upon that human classification.

The details of the files again are:
- two data files - red wine, white wine
- number of instances (records/observations) - red wine - 1599; white wine - 4898. 
- number of Attributes: 11 + output attribute
- attribute information
    - Input variables (based on physicochemical tests):
       - fixed acidity
       - volatile acidity
       - citric acid
       - residual sugar
       - chlorides
       - free sulfur dioxide
       - total sulfur dioxide
       - density
       - pH
       - sulphates (UK Eng) - sulfates
       - alcohol
   - Output variable (based on sensory data): 
       - quality (score between 0 and 10)

 - missing attribute values: **None


Wine Quality Data Set
The Wine 

## 3.1 - Data EDA
>TODO: from Notebook output

## 3.2 - Model Generation
>TODO: Notebook 3.2 SVM
>TODL: Notebook 3.2 DT

# 4 - Results

# 5 - Conclusions and Further Work



# Appendices

## Appendix A - Wine Data Set
The following is from the `winequality.names` file 

```
Citation Request:
  This dataset is public available for research. The details are described in [Cortez et al., 2009]. 
  Please include this citation if you plan to use this database:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  Available at: [@Elsevier] http://dx.doi.org/10.1016/j.dss.2009.05.016
                [Pre-press (pdf)] http://www3.dsi.uminho.pt/pcortez/winequality09.pdf
                [bib] http://www3.dsi.uminho.pt/pcortez/dss09.bib

1. Title: Wine Quality 

2. Sources
   Created by: Paulo Cortez (Univ. Minho), Antonio Cerdeira, Fernando Almeida, Telmo Matos and Jose Reis (CVRVV) @ 2009
   
3. Past Usage:

  P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis. 
  Modeling wine preferences by data mining from physicochemical properties.
  In Decision Support Systems, Elsevier, 47(4):547-553. ISSN: 0167-9236.

  In the above reference, two datasets were created, using red and white wine samples.
  The inputs include objective tests (e.g. PH values) and the output is based on sensory data
  (median of at least 3 evaluations made by wine experts). Each expert graded the wine quality 
  between 0 (very bad) and 10 (very excellent). Several data mining methods were applied to model
  these datasets under a regression approach. The support vector machine model achieved the
  best results. Several metrics were computed: MAD, confusion matrix for a fixed error tolerance (T),
  etc. Also, we plot the relative importances of the input variables (as measured by a sensitivity
  analysis procedure).
 
4. Relevant Information:

   The two datasets are related to red and white variants of the Portuguese "Vinho Verde" wine.
   For more details, consult: http://www.vinhoverde.pt/en/ or the reference [Cortez et al., 2009].
   Due to privacy and logistic issues, only physicochemical (inputs) and sensory (the output) variables 
   are available (e.g. there is no data about grape types, wine brand, wine selling price, etc.).

   These datasets can be viewed as classification or regression tasks.
   The classes are ordered and not balanced (e.g. there are munch more normal wines than
   excellent or poor ones). Outlier detection algorithms could be used to detect the few excellent
   or poor wines. Also, we are not sure if all input variables are relevant. So
   it could be interesting to test feature selection methods. 

5. Number of Instances: red wine - 1599; white wine - 4898. 

6. Number of Attributes: 11 + output attribute
  
   Note: several of the attributes may be correlated, thus it makes sense to apply some sort of
   feature selection.

7. Attribute information:

   For more information, read [Cortez et al., 2009].

   Input variables (based on physicochemical tests):
   1 - fixed acidity
   2 - volatile acidity
   3 - citric acid
   4 - residual sugar
   5 - chlorides
   6 - free sulfur dioxide
   7 - total sulfur dioxide
   8 - density
   9 - pH
   10 - sulphates
   11 - alcohol
   Output variable (based on sensory data): 
   12 - quality (score between 0 and 10)

8. Missing Attribute Values: None


```