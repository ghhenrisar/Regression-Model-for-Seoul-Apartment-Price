# Performance of Two Regression Models of the Seoul Apartment Price Dataset

![photo by Quang Nguyen Vinh](https://github.com/ghhenrisar/Regression-Model-for-Seoul-Apartment-Price/raw/e56198bf6776f544d8398fa6c67feb872cc06ab7/images/pexels-photo-21624423.jpg)*photo by Quang Nguyen Vinh*

There are some models for prediction of house price nowadays. Regression is often utilized for this purporse. It is a part of statistical tools to find the relationship between independent variable (X) and dependent variable (y). Regression is a part of machine learning because it is a supervised learning to predict value.
As I hope this project is as simplest as possible, scores of the models will be the goal of this project to get a brief understanding of the performance of the two models below.

# 1. Linear Regression
The technique to find the linear relationship between X and y is called Linear Regression. The following libraries will be imported. <span style="color:blue">%matplotlib inline</span> is used to show the graph here. Dataset **SeoulRealEstate.csv** which is used to demonstrate this regression technique below is provided by [Kaggle](https://www.kaggle.com).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df = pd.read_csv('seoul - SeoulRealEstate.csv')
df.head()
```

After passing the SeoulRealEstate dataset to DataFrame df, we can take a look on the data first.

### Column Names
- **lat**: Latitude  
- **lng**: Longitude  
- **households**: Number of households in residence  
- **buildDate**: Date the apartment was built  
- **score**: Total evaluation, maximum 5 stars  
- **m2**: The area of a house(m^2)  
- **p**: Number of floors  
- **minsales, maxsales, avg_sales**: Descriptive statistics of sales price

### Data Preprocessing

```python
df.info()
df.isnull().sum()
```

The data obtained sometimes is quite raw with null values which is an issue for data processing. Cleaning of data is required and dropna function is called to delete the rows with null values.

```python
df['avg_sales'].replace('', np.nan, inplace=True)
df.dropna(subset=['avg_sales'], inplace=True)
df.info()
df.isnull().sum()
```

Let's preview the relationship between the data with the sales price(avg_sales). Seaborn function jointplot is useful here.

```python
sns.jointplot(x=df.households, y=df.avg_sales, kind='reg')
sns.jointplot(x=df.m2, y=df.avg_sales, kind='reg')
sns.jointplot(x=df.buildDate, y=df.avg_sales, kind='reg')
sns.jointplot(x=df.p, y=df.avg_sales, kind='reg')
sns.jointplot(x=df.lat, y=df.lng, height=4)
```

We will split the dataset into X and y for Linear Regression.

```python
y = df.avg_sales
X = df.drop(['id', 'avg_sales', 'min_sales', 'max_sales'], axis=1)
```

Some algorithm and regression require data scaling for good results. MinMax Scaler can transform the data from 0 to 1 without changing the shape of the data distribution.

```python
from sklearn.preprocessing import MinMaxScaler
minMaxScaler = MinMaxScaler().fit(X)
X_normalized = minMaxScaler.transform(X)
df_normalized = pd.DataFrame(X_normalized, columns=X.columns)
df_normalized

sns.boxplot(data=df_normalized)
```

Then train_test_split function is called to split the dataset for training and evaluation of the model.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=2)
```

### Training
x_train and y_train are the data for training purpose.

```python
from sklearn.linear_model import LinearRegression

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
```

### Evaluation

`lr_model.score(x_test, y_test)`

Finially We will see the performance of Linear Regression by using the x_test and y_test data. We got score of 0.417 which is poor. Then, how to get a better result? Well, let's try another technique.

# 2. Gradient Boosting Regressor

According to [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
> Gradient Boosting builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.
 
```python
from sklearn import ensemble

gbr_model = ensemble.GradientBoostingRegressor(n_estimators=400, max_depth=5, min_samples_split=7, learning_rate=0.1, loss='ls')
```

### Parameters of GradientBoostingRegressor function
- **n_estimators**: the number of boosting stages
- **max_depth**: the depth of the tree node
- **min_samples_split**: the number of sample to be split for learning
- **learning_rate** the rate of learning
- **loss**: the loss function and 'ls' means least squares regression

```python
gbr_model.fit(x_train, y_train)
gbr_model.score(x_test, y_test)
```

### Conclusion
From the results above, Gradient Boosting did much better than Linear Regression. You can try other parameters to see if there is any improvement of the result.  
Thank you for your reading.

