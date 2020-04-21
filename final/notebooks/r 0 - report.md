# 1 - Introduction
The goal, repeated here, is to be able to predict a "good wine" based upon its physical and chemical properties.

# 2 - Methodology and Approach
## 2.1 - Classification
- binary vs multi-class problem

## 2.2 - Support Vector Machine
In using Support Vector Machines with a linear classified (which we will try first) the data must be linear separable. That means each of the feature to feature mappings must present the actual class that is clearly separated by a linear plane. While SVM with Radial Bias Function (RBF) can accommodate non-linear data we will look towards taking our Data and modifying it for easier analysis.

## 2.3 - Decision Tree
# 3 - Analysis and Modeling
## 3.1 - Data EDA
- Heat map / Correlation matrix via SNS - generate heatmap for each of the DF's - White, Red, All together
- Spread of labels
    - binary vs multi-class and the data
```    
np.unique(df.loc[:, 'quality'])   && plt.hist(df.loc[:, 'quality'])
```
- Spread of features
- testing for linear separable data
```
http://www.tarekatwan.com/index.php/2017/12/methods-for-testing-linear-separability-in-python/
    - sns.pairwise plot
```    
## 3.2 - Data Properties and Wrangling
- lots of ideas here:
  - https://subscription.packtpub.com/book/data/9781789615326/10/ch10lvl1sec75/hyperparameter-tuning-with-grid-search

## 3.3 - Model Generation
- https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html


## other things
- catch and articulate early convergence issues: https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.ConvergenceWarning.html


generate SVM models for each of the DF's predicting quality - White, Red, All
normalize / regularization of the data
PCA or refinement

https://scikit-learn.org/stable/modules/svm.html#multi-class-classification

Generate the scores to: https://scikit-learn.org/stable/modules/svm.html#scores-probabilities

Hyperparameter Tuning with Scikit https://subscription.packtpub.com/book/data/9781789615326/10/ch10lvl1sec75/hyperparameter-tuning-with-grid-search

https://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html#sphx-glr-auto-examples-svm-plot-svm-nonlinear-py

Try different kerne's
https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html#sphx-glr-auto-examples-svm-plot-iris-svc-py

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