#!/usr/bin/env python
# coding: utf-8

# ## 3.1 - Data Exploratory Data Analysis (EDA)
# Prior to jumping into modeling and running the algorithms, it is best to get a handle on the data. EDA is generally is visualizing, running statistics to understand the various features, data types within each feature and even between features and labels.
# 
# For this there are few areas that I'll document here that impacted the EDA and modeling. For each of the concerns, the sections that follow discuss some of the decisions made and approach for prepping the data for final modeling.
# - 3.1.1 - multi-class vs. binary classifcation and
#     - feature and label correlation
# - 3.1.2 - normalization and scaling feature data
# - 3.1.3 - linear separability of data
# 
# 
# Support Vector Machines are sensitive to data (features) that have not been scaled consistently and without customized kernels such as Radial Bias Function (RBF) cannot converge with non-linear data.  Decision Trees are not sensitive to feature scaling differences. In addition, Decision Trees are non-linear classifiers and have no sensitivity to the linear separability of the data. For the scaling and standardization needs you may decide to scale or normalize the data for visualization needs but the underlying alrorithm relies on value in the Entropy calculations that results in values between `0 and 1`, along with the information gain based upon conditional probabilities that is also positive.
# 
# 

# In[2]:


# First lets pull and load up Pandas data frames as follows:
# df_red - just the red wine
# df_white - just the white wine
# df_all -- both red and white in the same data frame.

# NOTE: if the file already exists, we skip the downoad from an internet
#       location. The force flag allows overriding that behavior.

from utils.helpers import *
df_red, df_white, df_all = pull_and_load_data(force = False)


# In[3]:


in_script()


# ### 3.1.1 - Class Analysis - multi-class vs. binary classifcation
# For this analysis I only use the combined dataset of red and white wine. You will see shortly as to why this may not matter **much** - I emphasize that this is something for further analsys in the future. However, my own experimentation showed little impact of the difference between white or red with only a few points difference in accuracy of the model. So, in short, you'll see the `df_all` data set - the combined dataset - as the primary corpus for this report.
# 
# #### 3.1.1.1 - Categories of Labels
# First we look at the category of labels and do a quick analysis on the data. The first visualization is a pairwise correlation for feature + labels. This generates a correlation matrix that can be shown using a Seaborn Heatmap that emphasizes stronger correlations, either negative or positive, for combination of feature or label. Again this capability is built into Pandas using the [DataFrame.corr()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html) function and Seaborn's Heatmap function [seaborn.heatmap()](https://seaborn.pydata.org/generated/seaborn.heatmap.html?highlight=heatmap#seaborn.heatmap).

# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns

figsize = 18, 18
plt.figure(figsize = figsize)
rv = df_all.corr()
sns.heatmap(rv, annot = True, annot_kws = {"size": 14}, cmap = "YlGnBu") # Blues, Blues_r YlGnBu
plt.show()


# #### 3.1.1.1 - Feature importance
# From the heatmap above, and the summary of the pairwise correlation below, we can see taht **color**, while not zero, has some impact on the quality - the value of `-0.12` in the overall order below, which is absolute value of each **quality vs feature** is in the middle. 
# 
# Since I'm not interested in determnining strictly either White or Red predictability, the future data sets will remove that column in order to just focus on the remaining features and the label.

# In[23]:


# this gives us a sigle series of the quality vs features absolute
# value sorted descending. The higher the number (abs) the greater the correlation.
ordered_corr = df_all.corr().loc['quality'].abs().sort_values(ascending=False)[1:]

ordered_corr.plot.bar()
plt.show()

emit_md('> Ordered Correlation result: {}')
print(ordered_corr)


# #### 3.1.1.2 - Dropping Color
# Now take a look at the same red & white wine data without color using the same heatmap and sort of correlation.
# 

# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns

# for is I will drop the color column that was added initially.
# we probably don't need the color given the correlation to
# quality was relatively 'low'

df_no_color = get_df_no_color(df_all)

figsize = 18, 18
plt.figure(figsize = figsize)
rv = df_no_color.corr()
sns.heatmap(rv, annot = True, annot_kws = {"size": 14}, label = "foo", cmap = "YlGnBu") # Blues, Blues_r YlGnBu
plt.show()


# In[25]:


# this gives us a sigle series of the quality vs features absolute
# value sorted descending. The higher the number (abs) the greater the correlation.
ordered_corr = df_no_color.corr().loc['quality'].abs().sort_values(ascending=False)[1:]

ordered_corr.plot.bar()
plt.show()
emit_md('> Ordered Correlation result: {}')
print(ordered_corr)


# #### 3.1.1.2 - Collapsing Classes
# So far, while there's has been quite a bit of analysis on impact of all the features, the focus initially was on the removing the **color** feature from our combined data set of red and white wines. Recall that the goal for **THIS** report is to see if we can predict how good a wine is based upon the chemical characteristics.
# 
# For the class data, the metadata provided indicates that the **quality** column can range from 0 - 10. Now we take a look at the quality data.
# 
# First, let's take a quick look at what values actually appear and the frequency of each.

# In[26]:


# here we can use a helper function that can return
# two distinct data frames - but since only interested in 
# labels for now, we can ignore the features using the python '_' syntax

_, df_labels = get_features_and_labels(df_all)

emit_md('> unique set of labels: `{}`'.format(np.unique(df_labels)))
emit_md('> bin count for 0-9: `{}`'.format(np.bincount(df_labels)))

#print('unique set of labels:', np.unique(df_labels))
#print('bin count for 0-9: ', np.bincount(df_labels))


# 
# First, we can see that the only values that exist for the **quality** label are from `3-9` - so, there are no `0, 1,2, or 10'`s as indicated in the metadata for the data set from the provider. 
# 
# This means we do not need to interpret if a `0` or `10` are the same. 
# 
# If we take a visual histogram plot of the frequency for each, we can see that the distribution is not normal and there is a clear break or cutoff within the labels.

# In[27]:


df_labels.plot.hist()
plt.show()


# 
# 
# Using this information we now take a guess at what is meant by a **good** wine. At this point I'm making a decision to use the label of `5` as the cutoff for **good** vs **not-good** for the outcome of the sensory analysis by humans. Clearly this can be challenged as the idea of **good** vs **not-good** may have changed each review of the wines by each human. For example one human may feel that their scale of `8` or higher is **good** where another may feel that `3` or higher is **good**.
# 
# But for this analysis since we are splitting the data approximately in half (actually more on the **good** side vs **not bood**) we are being in some ways *generous*. But again, thats subjective interpretation of this scale as recorded in the dataset.
# 
# #### 3.1.1.3 - Converting Multi-class to Binary (two class)
# 
# This is easy enough to accomplish

# In[28]:


_, df_labels = get_features_and_labels(df_all, binary=True)

rv1 = 'unique set of labels: {}'.format(np.unique(df_labels))
rv2 = 'bin count for 0-9: {}'.format(np.bincount(df_labels))


# In[29]:


emit_md('> RESULT: ' + rv1)
emit_md('> RESULT: ' + rv2)


# In[30]:



bins = range(0, 3)
df_labels.plot.hist(bins = bins, width = 0.5)
bins_labels(bins, shift = 0.25)
plt.show()


# ### 3.1.2 - Feature normalization with scaling
# 
# #### 3.1.2.1 - Training and Modeling performace
# 
# In order to improve performance of certian machine learning algorithms scaling data so that it is both consistent and within reasonable ranges can help. As articulated in [Role of Scaling in Data Classification Using SVM. Minaxi Arora, Lekha Bhambhu](http://ijarcsse.com/Before_August_2017/docs/papers/Volume_4/10_October2014/V4I10-0254.pdf)
# 
# Support Vector Machines in LIBSVM, which is what SciKit wraps, leverages a convex function (Gradient Descent or Stochastic Gradient Descent) during optimization. As pointed out in the paper, large variances in feature data can cause issues with the calculations even lengthy even perpetual lack of convergence. Convergence as it relates to Support Vector Machines is when the algorigthm has identified the hyperplane based upon the features that can separate the classes.  Later I'll cover linear separability and illustrate what that means with regards to Support Vector Machines. However, at this point be aware that the data from the Wine data set is not linear-separable. Thus the kernel to be used is [Radial Bias Function (RBF)](https://en.wikipedia.org/wiki/Radial_basis_function_kernel). This algoritm is simpler and requires only two parameters for tuning (hyperparameters) making our experimentation simpler.
# 
# From [Mixaxi](http://ijarcsse.com/Before_August_2017/docs/papers/Volume_4/10_October2014/V4I10-0254.pdf):
# ```
# Main purpose of scaling data before
# processing is to avoid attributes in greater numeric ranges. Other purpose is to avoid some types of numerical difficulties
# during calculation. Large attribute values might cause numerical problems. Generally, scaling is performed in the range
# of [-1, +1] or [0, 1].
# ```
# 
# For our scaling needs Scikit has [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) which happens to be  used scaler
# 
# As articulated also on the documentation for StandardScaler
# 
# ```
# If a feature has a variance that is orders of magnitude larger that others, it might dominate the objective function and make the estimator unable to learn from other features correctly as expected.
# ```
# 
# There are other preprocessing approaches and classes within SciKit, such as statistical normalization. That takes the data and converts it to Z scores so the mean is always 0, and the data is somewhat normally distributed. 
# 
# 
# #### 3.1.2.2 - Feature normalization with scaling
# At this point we will take a look at each of the features and their statistical measures.
# 
# First, extract the features and labels, and apply the binary classification transform - which again, just adjusts the Quality class from `0-9`, to `1 or 0`, when `x > 5`. Using the following where column `11` is the **quality** class.
# 
# ```
# np.where(df_all.iloc[:, 11].to_numpy() > 5, 1, 0))
# ```
# 
# 

# In[31]:


df_features, df_labels = get_features_and_labels(df_all, binary=True)


# 
# Next, using Pandas built in statistical tool [Describe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html) we generate the descriptive statistics for each of the features.
# 
# 

# In[32]:


df_features.describe()


# 
# Now, generate a boxplot for each of the features:
# 
# 

# In[33]:


figsize = 10, 10
plt.figure(figsize = figsize)
sns.boxplot(data = df_features)
plt.show()


# 
# 
# The first thing is clear is that several of the features are not within the same numerical ranges as the others.
# 
# Next, we can drop just those two to see the other feataures that may better align.
# 
# 

# In[34]:


df_features_drop1 = df_features.drop(axis = 1, columns = ['free_sulfur_dioxide','total_sulfur_dioxide'])
df_features_drop1


# In[35]:


figsize = 10, 10
plt.figure(figsize = figsize)
sns.boxplot(data = df_features_drop1)
plt.show()


# 
# 
# Again, we see that the remaining features are also not all within the same numerical ranges. 
# 
# As mentioned before, lack of standardization or normalization of feature data can inhibit convergence speed and cause potentially perpetual non-convergence.  Since the Support Vector Machine algorithm attempts to maximize the distance on the plane or vector separating data, if the data of one feature is at a greater scale then the algorithm focuses on that feature first, when in fact the feature with the larger scale may have greater influence as while its scale is low, it ends up being ignored in convergence steps.
# 
# 
# Finally, let's plot the features individually just to see their distribuation within their own scale.
# 

# In[36]:


def box_plot_features(df):
    n = len(df_features.columns)
    figsize = 15,15
    fig, axes = plt.subplots(nrows=4, ncols=3, 
                             figsize = figsize, 
                             sharex = True, 
                             sharey = False)

    ic = 0
    for r in range(0, 4, 1):
        for c in range(0, 3, 1):
            if ic < n:
                dt = df.iloc[:,ic]
                sns.boxplot(data = dt.values, ax = axes[r,c]).set_title(dt.name)
                #print(dt.head)
                ic += 1


    figsize = 15, 15
    plt.figure(figsize = figsize)
    plt.show()


box_plot_features(df_features)


# 
# As shown in the boxplots, for many of the features outside the quartile range plot there are many outliers. Normalization won't help as much. For the standardization we will choose the [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) which standardizes the data resulting in a mean of `0` and standard deviation of `1`. This is applied to all the features.
# 
# During modeling experimentation, the StandardScaler is used during the pipeline processing and applied to the features (not labels) before training or testing the the model.
# 
# 
# 

# ### 3.1.3 - Linear separability of data
# 
# Lastly, linear separability of data is important, actually required for algorithms such as Perceptron, ADALine, Linear Regression.  The data in the Wine set is NOT linear separable.
# 
# This first plot is using the original quality values of` 0-9` which won't be usable in a 2D or 3D perspective at all.
# 

# In[37]:


figsize = 15, 15
plt.figure(figsize = figsize)
#sns.pairplot(df_features)
sns.set(style="ticks", color_codes=True)
df_no_color = get_df_no_color(df_all)
sns.pairplot(df_no_color, hue = 'quality')
plt.show()


# 
# 
# This second plot uses the conversion of quality to a binary classification as articulated before.
# 
# 

# In[38]:


figsize = 10, 10
plt.figure(figsize = figsize)
sns.set(style="ticks", color_codes=True)
df_no_color = get_df_no_color(df_all, binary = True)
sns.pairplot(df_no_color, hue = 'quality')
plt.show()


# 
# From the series of scatter plots while there is some level of grouping, there are no areas where there is a clear linear separable plane in the 2D view.  This is where in Support Vector Machies the Radial Bias Function (RBF) can be applied to take non-linear data and move it to a higher plane/dimension via transformation in the RBF Kernel (function).
# 
# As an example, the following scatter plots show a simple linear separable data set on the left (A), and a non-linear separable data set on the right (B).
# From: [Methods for Testing Linear Separability in Python, Tarek Atwan](http://www.tarekatwan.com/index.php/2017/12/methods-for-testing-linear-separability-in-python/)
# 
# Through Kernel functions this data is transformed to higher dimensions where the data can be separated, albeit not always able to visualize so easily.
# 
# ![](./images/linear_sep.png)
# 
# 
# Further more, from [Machine Learning Notebook](https://sites.google.com/site/machinelearningnotebook2/classification/binary-classification/kernels-and-non-linear-svm) the image below demonsrates non-linear data in 2D on the left, through transformation to a higher dimension, it becomes separable by a plane (or vector).
# 
# 
# ![](./images/linear_sep2.png)
# 
# 
# It is these transformations that the RBF Kernel in Scikit and LIBSVM ultimately provide as part of the optimization process.
# 
