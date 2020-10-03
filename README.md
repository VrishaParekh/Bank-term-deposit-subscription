# Bank Term deposit subscription classification  
This project is a classification task to classify whether a customer will subscribe to a term deposit or not based on the dataset bank-additional-full.csv

This project consists of two files which are as follows:

- **Code Script file** with the data visualization and step by step working
- **Function file** which is the automated code script

# Table of Contents 

- **1. Introduction**

- **2. Exploring the data**

- **3. Exploratory data analysis**

 - 3.1.  Describing the features
 - 3.2. Descriptive features
 
     - 3.2.1.Box plots
     - 3.2.2. Distribution plots
     - 3.2.3. Correlation matrix
     - 3.2.4. Outlier Detection
     
 - 3.3. Feature Engineering
     - 3.3.1. Feature transformation 
     - 3.3.2. Dropping highly correlated feature
     - 3.3.3. Creating a new feature
     - 3.3.4. Replacing outplut variable with new data
     - 3.3.5. Creating dummy variables
     
- **4. Model Selection**

 - 4.1. Data preparation
 - 4.2. Creating summary metrics
 - 4.3.Fitting the baseline model
 
     - 4.3.1. Checking the model performance
     - 4.3.2. Creating a confusion matrix
     
 - 4.4.Fit the model with improvised data
 
     - 4.4.1. Using sampling techniques to balance the imbalanced classes
     
     - 4.4.2.  Applying Recursive Feature Elimination(RFE) with cross-fold evaluation
     
- **5. Describing important features**

- **6. Results**

- **7. Actionable Recommendations**
     
 
# 2. Introduction

**About the project**- The data is related to the direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed. The bank runs a marketing campaign to bring customers on board with the term deposits.

Moreover, this dataset is based on the "Bank Marketing" UCI dataset (please check the description at http://archive.ics.uci.edu/ml/datasets/Bank+Marketing). The data is enriched by the addition of five new social and economic features/attributes (national wide indicators from a ~10M population country), published by the Banco de Portugal and publicly available at https://www.bportugal.pt/estatisticasweb.

The binary classification goal is to predict if the client will subscribe to a bank term deposit (variable y).

Number of Instances: 41188 for bank-additional-full.csv

Number of Attributes: 20 + output attribute.

**Feature characteristics**

**Input variables:**

#### bank client data:
1. age (numeric)
2. job : type of job (categorical)
3. marital : marital status (categorical)
4. education (categorical)
5. default: has credit in default? (categorical)
6. housing: has housing loan? (categorical)
7. loan: has personal loan? (categorical)

#### related to the last contact of the current campaign:
8. contact: contact communication type (categorical)
9. month: last contact month of year (categorical)
10. day_of_week: last contact day of the week (categorical)
11. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

#### other attributes:
12. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14. previous: number of contacts performed before this campaign and for this client (numeric)
15. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

#### social and economic context attributes
16. emp.var.rate: employment variation rate - quarterly indicator (numeric)
17. cons.price.idx: consumer price index - monthly indicator (numeric)
18. cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19. euribor3m: euribor 3 month rate - daily indicator (numeric)
20. nr.employed: number of employees - quarterly indicator (numeric)

#### Output variable (desired target):
21. y - has the client subscribed a term deposit? (binary: 'yes','no')

**Aim of the project**- This project is aimed at classifying whether the customer will subscribe a bank term deposit (variable y).


# 3. Launch

## 3.1. Technologies 

**1.Python**- Python programming language is utilized as a standalone base to solve the purpose of the project.

**2.Jupyter IDE**- Jupyter is utilized as a platform to write code scripts in python and do the data pre-processing, visualization, and Machine Learning tasks.

**Pandas**- Pandas library in python is put to use for data manipulation and analysis.

**3.NumPy**- NumPy library of python is used for statistical purposes and also for making use of multidimensional arrays.

**4.Matplotlib**-This library of python is used for data visualization.

**5.seaborn**- This library of python is used for data visualization.

**6.scikit-learn**- The sklearn library is used for building Machine learning models like Linear regression and RandomForest Regressor. It also gives the functionality of splitting the data into train and test sets and is useful for calculating the statistical scores like RMSE, accuracy score, and Mean absolute error. Lastly, its functionality also extends to assisting in finding multicollinearity and doing cross-fold validation.

## 3.2. Launch

### 3.2.1. Setup

To run the code script please follow the steps below.

1.Install Anaconda(64-bit graphical installer version, if not previously installed), it can be installed via https://www.anaconda.com/products/individual

2.Launch Jupyter Notebook(version 6.0.3) from the anaconda navigator page or it can be directly launched from the terminal if Anaconda is previously installed(in case of MacBook).

3.Download the code script from this GitHub repository and save it in your working directory of Jupyter notebook. Alternatively, the code script can be uploaded directly in the jupyter notebook.

4.The required dataset 'bank-additional-full.csv' also needs to be uploaded in the current working directory of jupyter.

5.Install the libraries pandas, NumPy,scikit-learn,matplotlib, seaborn, and statsmodels if not previously installed.

6.Import the list of libraries below.

### 3.2.2.Detailed list of libraries

All the libraries mentioned below are a prerequisite to run the project.

Importing Libraries

- import pandas as pd
- import numpy as np
- import matplotlib.pyplot as plt
- %matplotlib inline
- import seaborn as sns
- from numpy import mean
- from numpy import std
- from scipy.stats import norm
- from sklearn import preprocessing
- from sklearn.preprocessing import StandardScaler
- from sklearn.model_selection import train_test_split
- from sklearn.linear_model import LogisticRegression
- from sklearn.feature_selection import RFE
- from sklearn.tree import DecisionTreeClassifier
- from sklearn.ensemble import RandomForestClassifier
- from sklearn.metrics import classification_report,confusion_matrix,recall_score,precision_score,accuracy_score
- from sklearn.pipeline import Pipeline
- from sklearn.model_selection import RepeatedStratifiedKFold
- from sklearn.model_selection import cross_val_score
- from warnings import simplefilter
- simplefilter(action='ignore', category=FutureWarning)

### 3.2.3. Importing the dataset

The dataset 'bank-additional-full.csv' needs to be imported using the command (daydata=pd.read_csv('bank-additional-full.csv')).This file format has to be in .csv format.


# 4. Project Overview

**Exploring the data**- In this section descriptive statistics have been performed to understand the nature of the data and to analyze its shape and if it consists of any missing values.

**Exploratory data analysis(EDA)**- EDA is performed via distribution plots, box plots, and correlation matrix to understand the distribution of the data and to check if there exist any outliers and the correlation of variables with each other.

**Modelling**- Data is then split into train and test sets with 75% into the train set and 25% into the test set. After this, the data is scaled and a baseline Logistic regression, Decision tree, Random Forest are fitted to compare to the improvised models later. Observing that the data is highly imbalanced with 88% negative class records(0) and only 11% of positive class records(1), a mix of Oversampling and under-sampling techniques was utilized to balance them out. Moreover, Recursive feature elimination(RFE) was utilized followed by 10-fold cross-validation to improve the efficiency of the models and get better models in comparison to the baseline models. Lastly, extending the functionality of RFE, important features for the model were categorized.

### **Results**

1. For our business problem, we wanted to predict the term subscription based on a campaign so it is vital to have a high recall to have higher revenue. For that, it is important to choose a model with better recall and less false negatives, hence we choose the Logistic Model to better classify the customer term deposit prediction.
 
2. The improvised logistic model results in 70.29% better recall than the baseline logistic model.


3. All the model's classification report that we have got still indicate the maximum classification of the majority class(0). This tells us that there is more scope for model optimization or the use of different classification unsupervised learning models.


4. For every false positive the bank will be spending money and investing in other revenue-generating sources. On the other hand, the bank will lose its customer trust and the possibility of losing the customer when the resources would also have been utilized.


5. For every false negative the bank has the possibility of losing out a potential customer.


6. For the marketing campaign, its very important for the bank to get in more customers to subscribe for the term deposit so that the bank can get more revenue, if in this case, the bank starts to lose out on customers in other words the bank would not want customer churn as a result of the output of the marketing campaign. So, it's important to have less False Negatives.

### **Actionable Recommendations**

##### Observing the results and the exploratory data analysis, the most important features which the bank should focus on to attract more customers to buy term deposit are:

- Duration
- age
- campaign
- Euribo3
- nr.employed


1. Duration being one of the most influential factors,i.e. the higher the call duration the higher the chances of a sale. So the bank should focus on enhancing the quality of calls by building a rapport with the customers, decreasing wait time, checking in with the customers, and most importantly take feedback from the customers.


2. Age feature demonstrates that the majority term deposit purchasing capacity lies within the age group of 25-58 yrs adults. So, the bank should target this age group more and allocate more resources in getting in the customers from this particular age group.


3. Campaign feature is important as it indicates the number of calls made during the current campaign. The customers do not like to get bothered with too many calls so a sweet spot lies within 1-5 calls, again depending upon the interest of the customer. So the bank should focus on training the sales team so that they can know the interested and non-interested customers based on the behavior,voice modulations, tone, and pitch of the customer.


4. Euribo3 is indicative of the trend that the higher interest rates attract more customers. So there are two things which the bank can pursue which are as follows:
   -Target the age group which is liable to get higher interest rates (4.5-5) particularly.
   -Increase the marketing campaign when the interest rates are higher, which can help in bringing more clients on board with the term deposits.
   
   
5. nr.employed trend indicates that more number of employees leads to more number of customers, which makes sense because if there are more employees, more leads can be targeted, proper followups and check-ins can be done. On the other hand, customer satisfaction could be achieved by creating a dedicated after-sales team. So, the bank should focus on hiring more people.


# Example of use

1. This classification model can be put into application for predicting whether the customer will subscribe for a term deposit
2. It can similarly be used for other similar bank campaigns example- loan prediction.


# Project Status

This project is developed.


# References

1.UCI Machine Learning Repository.Retrieved from https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

2.Building a logistic regression in python. Retrieved from https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8

3.Feature selection with sklearn and pandas.Retrieved from https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b

4.Code methodologies adopted from the DSDJ-Kyle McKiou team.


