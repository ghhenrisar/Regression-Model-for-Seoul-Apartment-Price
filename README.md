# Performance of Two Regression Models of the Seoul Apartment Price Dataset

![photo by Quang Nguyen Vinh](https://github.com/ghhenrisar/Regression-Model-for-Seoul-Apartment-Price/raw/e56198bf6776f544d8398fa6c67feb872cc06ab7/images/pexels-photo-21624423.jpg)*photo by Quang Nguyen Vinh*

There are some models for prediction of house price nowadays. Regression is often utilized for this purporse. It is a part of statistical tools to find the relationship between independent variable (X) and dependent variable (y). Regression is a part of machine learning because it is a supervised learning to predict value.
As I hope this project is as simplest as possible, scores of the models will be the goal of this project to get a brief understanding of the performance of the two models below.

# 1. Linear Regression
The technique to find the linear relationship between X and y is called Linear Regression. The following libraries will be imported. <span style="color:blue">%matplotlib inline</span> is used to show the graph here. Dataset **SeoulRealEstate.csv** which is used to demonstrate this regression technique below is provided by [Kaggle](https://www.kaggle.com).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

