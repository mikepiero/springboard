# Forecasting disease from univariate time series using naive, ARMIA, exponential smoothing, additive regression, and LSTM models

This project focuses on forecasting from a univariate time series using a variety of models. The project is a learning exercise I completed as part of Springboard's data science bootcamp. Below I briefly summarize the project and list the key documents.

## Summary

In this project, I forecast cases of dengue fever. Dengue fever is a tropical disease. It is found around the globe and afflicts 400 million people each year. 

The data comes from a data-science competition sponsored by DrivenData. The data includes decades of weekly observations of dengue cases in two Latin American cities. The goal is to make a multi-year forecast of weekly cases for each of these cities.

To do so, I assess, explore, and prepare the data. I forecast using five approaches–naive, ARMIA, exponential smoothing, additive regression, and LSTM. In the process, I work through 35+ variations on these models. 

The best performing model takes an additive approach using log-transformed data. I use Facebook's Prophet forecasting tool to implement this approach.

## Documents

The key documents for this project are:

-   A 20-page report, which is found in the docs directory, and
-   A collection of ten Jupiter notebooks, which are found in the notebooks directory. Generally, these notebooks represent steps in my process, and the numbers in the notebook's filenames correspond to the report's sections.