# Predicting home prices with linear, lasso, ridge and random forest regression

This project focuses on predicting the sales price of homes using a variety of regression techniques. This project is a learning exercise I completed as part of Springboard’s data science bootcamp. Below I briefly summarize the project and list some key documents.

## Summary

“What will my home sell for?” is a common question. In this project, I predict the sales prices of homes.

The data comes from a Kaggle competition. The data is a well-known training set called the “Ames Housing" data. It includes about 1,500 observations of the sales price of of residential properties in Ames, Iowa. The data also includes about 80 features describing the home, such as the number of rooms, the number of bathrooms, the year built and the size of the house. 

The task is to predict the sales price of another 1,500 or so homes in Ames. To do this, I clean the data, do exploratory analysis, and select relevant features. I model with a variety of regression techniques, specifically linear, lasso, ridge, and random forest regression. 

Here are some highlights of this project:

-   I select only 12 of the 79 features as many were highly correlated;
-   On this data, a normalized linear regression model outperforms ridge regression, lasso regression, and random forest regression (across different feature combinations), with a RMSE, MAE, and R2 of 0.1509, 0.1095 and 0.8578, respectively; and
-   I predict prices at a 95% confidence level.

## Documents

Here are the key documents and their directories.

-   The project’s report is in the docs directory; and
-   The five Jupiter notebooks focus on: cleaning the source data, exploring the target variable, selecting features, modeling with linear regression, including OLS, ridge, and lasso, and modeling with random forest regression. Generally, each notebook maps to a step in my process, and the numbers in the notebook’s filenames correspond to sections of the report.