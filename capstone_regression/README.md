# Predicting home prices with regression

This project focuses on predicting the sales price of homes using linear and random forest regression. This project is a learning exercise I undertook as part of Springboard’s data science bootcamp.  Below I've listed the key documents as well as the top findings

## Key documents

Here are the key documents and their enclosing directories.

The project's documents directory includes:

*   A final report and 
*   A final presentation.

The code directory includes Jupyter notebooks focused on:

* Cleaning the source data,
* Exploring the target variable--the sales price of homes,
* Exploring features--about 80 variables, such as size, quality, and location of homes,
* Modeling with linear regression, including OLS, Ridge and Lasso, and
* Modeling with random forest regression

The data directory includes the original data set--a collection of prices and home features from about 750 sales in Ames, Iowa during the late 2000s. This directory, however, excludes the interim files produced and consumed by the various notebooks.

## Top findings

Here are some highlights of what I found.  For details, please look at the final report.

*   On this data, the normalized linear regression model outperformed Ridge regression, Lasso regression, and Random forrest regression (across different feature combinations), with a RMSE, MAE, and R2 of 0.1509, 0.1095 and 0.8578, respectively.
*   I was able to predict prices and create prediction intervals at a 95% confidence level.
*   I ended up using only 12 of the 79 features as many were highly correlated.

