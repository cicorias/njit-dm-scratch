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

## 3.1 - Data Exploratory Data Analysis (EDA)
Prior to jumping into modeling and running the algorithms, it is best to get a handle on the data. EDA is generally is visualizing, running statistics to understand the various features, data types within each feature and even between features and labels.

For this there are few areas that I'll document here that impacted the EDA and modeling. 
- multi-class vs. binary classification
- feature and label correlation
