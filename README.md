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
```


```python
df = pd.read_csv('seoul - SeoulRealEstate.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>lat</th>
      <th>lng</th>
      <th>households</th>
      <th>buildDate</th>
      <th>score</th>
      <th>m2</th>
      <th>p</th>
      <th>min_sales</th>
      <th>max_sales</th>
      <th>avg_sales</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2766</td>
      <td>37.681604</td>
      <td>127.056592</td>
      <td>492</td>
      <td>200006</td>
      <td>4.3</td>
      <td>139</td>
      <td>42</td>
      <td>60100.0</td>
      <td>62000.0</td>
      <td>61000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5860</td>
      <td>37.679290</td>
      <td>127.057021</td>
      <td>468</td>
      <td>200105</td>
      <td>4.1</td>
      <td>105</td>
      <td>32</td>
      <td>48600.0</td>
      <td>52200.0</td>
      <td>51000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15564</td>
      <td>37.676882</td>
      <td>127.058075</td>
      <td>57</td>
      <td>200502</td>
      <td>4.8</td>
      <td>86</td>
      <td>26</td>
      <td>36000.0</td>
      <td>46000.0</td>
      <td>40500.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3700</td>
      <td>37.675277</td>
      <td>127.060001</td>
      <td>216</td>
      <td>199509</td>
      <td>4.8</td>
      <td>102</td>
      <td>31</td>
      <td>34000.0</td>
      <td>34800.0</td>
      <td>34500.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6204</td>
      <td>37.676381</td>
      <td>127.058361</td>
      <td>165</td>
      <td>200306</td>
      <td>4.8</td>
      <td>91</td>
      <td>28</td>
      <td>27900.0</td>
      <td>50300.0</td>
      <td>40000.0</td>
    </tr>
  </tbody>
</table>
</div>



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

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4021 entries, 0 to 4020
    Data columns (total 11 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   id          4021 non-null   int64  
     1   lat         4021 non-null   float64
     2   lng         4021 non-null   float64
     3   households  4021 non-null   int64  
     4   buildDate   4021 non-null   int64  
     5   score       4021 non-null   float64
     6   m2          4021 non-null   int64  
     7   p           4021 non-null   int64  
     8   min_sales   3931 non-null   float64
     9   max_sales   3931 non-null   float64
     10  avg_sales   3931 non-null   float64
    dtypes: float64(6), int64(5)
    memory usage: 345.7 KB
    




    id             0
    lat            0
    lng            0
    households     0
    buildDate      0
    score          0
    m2             0
    p              0
    min_sales     90
    max_sales     90
    avg_sales     90
    dtype: int64



The data obtained sometimes is quite raw with null values which is an issue for data processing. Cleaning of data is required and dropna function is called to delete the rows with null values.


```python
df['avg_sales'].replace('', np.nan, inplace=True)
df.dropna(subset=['avg_sales'], inplace=True)
df.info()
df.isnull().sum()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 3931 entries, 0 to 4020
    Data columns (total 11 columns):
     #   Column      Non-Null Count  Dtype  
    ---  ------      --------------  -----  
     0   id          3931 non-null   int64  
     1   lat         3931 non-null   float64
     2   lng         3931 non-null   float64
     3   households  3931 non-null   int64  
     4   buildDate   3931 non-null   int64  
     5   score       3931 non-null   float64
     6   m2          3931 non-null   int64  
     7   p           3931 non-null   int64  
     8   min_sales   3931 non-null   float64
     9   max_sales   3931 non-null   float64
     10  avg_sales   3931 non-null   float64
    dtypes: float64(6), int64(5)
    memory usage: 368.5 KB
    




    id            0
    lat           0
    lng           0
    households    0
    buildDate     0
    score         0
    m2            0
    p             0
    min_sales     0
    max_sales     0
    avg_sales     0
    dtype: int64



Let's preview the relationship between the data with the sales price(avg_sales). Seaborn function jointplot is useful here.


```python
sns.jointplot(x=df.households, y=df.avg_sales, kind='reg')
```




    <seaborn.axisgrid.JointGrid at 0x1f212942d00>




    
![png](output_13_1.png)
    



```python
sns.jointplot(x=df.m2, y=df.avg_sales, kind='reg')
```




    <seaborn.axisgrid.JointGrid at 0x1f214549700>




    
![png](output_14_1.png)
    



```python
sns.jointplot(x=df.buildDate, y=df.avg_sales, kind='reg')
```




    <seaborn.axisgrid.JointGrid at 0x1f2148908e0>




    
![png](output_15_1.png)
    



```python
sns.jointplot(x=df.p, y=df.avg_sales, kind='reg')
```




    <seaborn.axisgrid.JointGrid at 0x1f214b240a0>




    
![png](output_16_1.png)
    



```python
sns.jointplot(x=df.lat, y=df.lng, height=4)
```




    <seaborn.axisgrid.JointGrid at 0x1f214e6f460>




    
![png](output_17_1.png)
    


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
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lat</th>
      <th>lng</th>
      <th>households</th>
      <th>buildDate</th>
      <th>score</th>
      <th>m2</th>
      <th>p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.987743</td>
      <td>0.127672</td>
      <td>0.048734</td>
      <td>0.625112</td>
      <td>0.86</td>
      <td>0.471910</td>
      <td>0.469136</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.986488</td>
      <td>0.127910</td>
      <td>0.046203</td>
      <td>0.642793</td>
      <td>0.82</td>
      <td>0.344569</td>
      <td>0.345679</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.985183</td>
      <td>0.128495</td>
      <td>0.002848</td>
      <td>0.713699</td>
      <td>0.96</td>
      <td>0.273408</td>
      <td>0.271605</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.984313</td>
      <td>0.129564</td>
      <td>0.019620</td>
      <td>0.536346</td>
      <td>0.96</td>
      <td>0.333333</td>
      <td>0.333333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.984911</td>
      <td>0.128654</td>
      <td>0.014241</td>
      <td>0.678693</td>
      <td>0.96</td>
      <td>0.292135</td>
      <td>0.296296</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>3926</th>
      <td>0.906735</td>
      <td>0.022327</td>
      <td>0.038186</td>
      <td>0.678693</td>
      <td>0.90</td>
      <td>0.273408</td>
      <td>0.271605</td>
    </tr>
    <tr>
      <th>3927</th>
      <td>0.906486</td>
      <td>0.001650</td>
      <td>0.009494</td>
      <td>0.784962</td>
      <td>0.80</td>
      <td>0.348315</td>
      <td>0.345679</td>
    </tr>
    <tr>
      <th>3928</th>
      <td>0.906411</td>
      <td>0.001105</td>
      <td>0.006540</td>
      <td>0.803001</td>
      <td>0.70</td>
      <td>0.262172</td>
      <td>0.259259</td>
    </tr>
    <tr>
      <th>3929</th>
      <td>0.906077</td>
      <td>0.022471</td>
      <td>0.009388</td>
      <td>0.678693</td>
      <td>0.86</td>
      <td>0.232210</td>
      <td>0.234568</td>
    </tr>
    <tr>
      <th>3930</th>
      <td>0.905642</td>
      <td>0.001871</td>
      <td>0.005063</td>
      <td>0.732809</td>
      <td>0.80</td>
      <td>0.138577</td>
      <td>0.135802</td>
    </tr>
  </tbody>
</table>
<p>3931 rows × 7 columns</p>
</div>




```python
sns.boxplot(data=df_normalized)
```




    <AxesSubplot:>




    
![png](output_22_1.png)
    


Then train_test_split function is called to split the dataset for training and evaluation of the model.


```python
from sklearn.model_selection import train_test_split
```


```python
x_train, x_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=2)
```

### Training
x_train and y_train are the data for training purpose.


```python
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
```




    LinearRegression()



### Evaluation


```python
lr_model.score(x_test, y_test)
```




    0.4172377734948557



Finially We will see the performance of Linear Regression by using the x_test and y_test data. We got score of 0.417 which is poor. Then, how to get a better result? Well, let's try another technique.

# 2. Gradient Boosting Regressor

According to [scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
> Gradient Boosting builds an additive model in a forward stage-wise fashion; it allows for the optimization of arbitrary differentiable loss functions.


```python
from sklearn import ensemble
```


```python
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
```




    GradientBoostingRegressor(max_depth=5, min_samples_split=7, n_estimators=400)




```python
gbr_model.score(x_test, y_test)
```




    0.806625059190917



### Conclusion
From the results above, Gradient Boosting did much better than Linear Regression. You can try other parameters to see if there is any improvement of the result.  
Thank you for your reading.
