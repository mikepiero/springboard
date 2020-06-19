# import pandas as pd

def test():
    print('Hello world!')

def test_plus_1(x):
    return x+1

def stationarity_test_via_simple_calcs(df, city):
    """Set up a simple comparison of summary stats of two halves of a time series"""

    # Spit the data frame in two
    half = len(df['total_cases']) // 2  # Use quotent operator
    df_1half, df_2half = df.iloc[:half], df.iloc[half:]

    # Give some context
    print("Here's a quick test for stationarity.")
    print("Most likley, a time series is NOT stationary if . . . ")
    print('')

    # Compare mean
    print(". . . the mean of each half of the data isn't roughly the same.")
    cases_1half_mean, cases_2half_mean = df_1half['total_cases'].mean(), df_2half['total_cases'].mean()
    print('    Are the means of {0} and {1} roughly equal?'.format(cases_1half_mean, cases_2half_mean))
    if ((.95 * cases_1half_mean) <= cases_2half_mean) and ((1.05 * cases_1half_mean) >= cases_2half_mean):
        print('    Yes,  for the {} data, the second half is withing 5% of first.'.format(city))
        result_mean = 'mean_yes'
    else:
        print('    No, for the {} data, the second half is not within 5% of first half.'.format(city))
        result_mean = 'mean_no'
    print('')

    # Compare variance
    print(". . . if the variance of each half of the data isn't roughly the same.")
    cases_1half_var, cases_2half_var = df_1half['total_cases'].var(), df_2half['total_cases'].var()
    print('    Are the variances of {} and {} roughly equal?'.format(cases_1half_var, cases_2half_var))
    if ((.95 * cases_1half_var) <= cases_2half_var) and ((1.05 * cases_1half_var) >= cases_2half_var):
        print('    Yes, for the {} data, the second half is withing 5% of first.'.format(city))
        result_var = 'var_yes'
    else:
        print('    No,  for the {} data, the second half is not within 5% of first half.'.format(city))
        result_var = 'var_no'

    return result_mean, result_var



def stationarity_test_via_seasonal_decompose(df, city):
    """Test for stationarity by graphing trend, seasonality and residuals"""

    from statsmodels.tsa.seasonal import seasonal_decompose

    print('Decomposition of weekly cases into trend, seasonality and residual via a naive, additive model in {}'.format(
        city))
    decompose = seasonal_decompose(df['total_cases'], model='additive', period=52)
    g = decompose.plot()
    # g = plt.show()

    return


def stationarity_test_via_acf_graph(df, city):
    """Graph data frame as ACF"""
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf
    # from statsmodels.graphics.tsaplots import plot_pacf

    # Graph acf
    g = plot_acf(df, lags=52)
    g = plt.xlabel('Lags')
    g = plt.ylabel('Correlation')
    g = plt.title('Autocorrelation with 95% confidence interval - {}'.format(city))
    g = plt.box(on=None)
    g = plt.show()

    return

def stationarity_test_via_pacf_graph(df, city):
    """Graph data frame as PACF"""

    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_pacf

    # Graph acf
    g = plot_pacf(df, lags=52)
    g = plt.xlabel('Lags')
    g = plt.ylabel('Correlation')
    g = plt.title('Autocorrelation with 95% confidence interval - {}'.format(city))
    g = plt.box(on=None)
    g = plt.show()

    return


def stationarity_test_via_adf(df, city):
    """ Use statmodels' adfuller() to calc an ADF test"""

    from statsmodels.tsa.stattools import adfuller

    # Extact the total_cases as a series
    df_cases = df['total_cases'].copy().to_numpy()

    # Calc test statistics
    results = adfuller(df_cases, maxlag=52, autolag=None)  # Set lags to 52

    # Print results
    print('Here are the resutls of testing using ADF in {}'.format(city))
    print('')
    print('ADF test stat: \t{:>20.5f}'.format(results[0]))
    print('P value: \t{:>20.5f}'.format(results[1]))
    print('Lags used: \t{:>20.5f}'.format(results[2]))
    print('Observations: \t{:>20.5f}'.format(results[3]))
    for key, value in results[4].items():
        print('    {}: \t{:>20.5f}'.format(key, value))

    # Print conclusion
    print('')
    print('Applying the test results')
    print('My null is:  there is a unit root, which means not stationary')
    print('At significance level of 5%, my thinking is:')
    if results[1] < .05:
        #     if results[0] < results[4].get('5%'):
        print('\t P-value is less than critical value.')
        print('\t Reject null.  Accept alternative.')
        print('\t No satistically signifcant unit root')
        print('\t That is, this time series is stationary')
    else:
        print('\t P-value is greater than or equal to critical value')
        print('\t Do not reject null.  Do not accept alternative')
        print('\t Cannot conclude there is stat. sig unit root.')
        print('\t That is, this time is no not stationary')

    return


def stationarity_test_via_kpss(df, city):
    """ Use statmodels' adfuller() to calc an ADF test"""

    from statsmodels.tsa.stattools import kpss # Odd

    # Extact the total_cases as a series
    cases_array = df['total_cases'].copy().to_numpy()

    # Calc test statistics
    results = kpss(cases_array, nlags=52)  # Set lags to 52

    print('Here are the restuls of testing using KPSS in {}'.format(city))
    print('')
    print('KPSS stat: \t{:>20.5f}'.format(results[0]))
    print('P value: \t{:>20.5f}'.format(results[1]))
    print('Lags used: \t{:>20.5f}'.format(results[2]))
    for key, value in results[3].items():
        print('    {}: \t{:>20.5f}'.format(key, value))

    # Print conclusion
    print('')
    print('Applying the test results')
    print('My null is:  there is no unit root, which means stationary')
    print('At significance level of 5%, my thinking is:')
    if results[1] < .05:
        #     if results[0] < results[3].get('5%'):
        print('\t P-value is less than critical value.')
        print('\t Reject null.  Accept alternative.')
        print('\t Yes, there is a statistically signifcant unit root')
        print('\t That is, this time series is NOT stationary')
    else:
        print('\t P-value is greater than or equal to critical value')
        print('\t Do not reject null.  Do not accept alternative')
        print('\t Cannot conclude there is no stat. sig. unit root.')
        print('\t That is, this time series is likely stationary.')

    return


# Define function to score RMSE and MAE

def score(approach, variation, summary, city, data, transform, X, y, score_df):
    """Build a dataframe with the RMSE and MAE for given test or training set"""

    import numpy as np
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import mean_squared_error

    # Start a list to add to the score df
    score_ls = [approach, variation, summary, city, data, transform]

    # Score RMSE
    rmse = np.sqrt(mean_squared_error(X, y))
    score_ls.append(rmse)

    # Score MAE
    mae = mean_absolute_error(X, y)
    score_ls.append(mae)

    # Append list to scoring data frame
    score_df.loc[len(score_df)] = score_ls

    return

def graph_actual_and_forecast_from_test(df_original, city):
    """Create line graphs of actual v. forecast when given the right dataframe"""

    # Import
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    import seaborn as sns

    # Set some Seaborn defaults
    sns.set_style('whitegrid')
    sns.set_context('paper')
    sns.set_palette('muted')

    # Clean up data frame
    df = df_original.copy()
    df.reset_index(inplace=True)  # Reset the index for Seaborn

    # Create plot
    g = plt.figure(figsize=(9, 3), dpi=100)
    g = sns.lineplot(data=df, x='week_start_date', y='actual', label='actual')
    g = sns.lineplot(data=df, x='week_start_date', y='forecast', label='forecast')
    g = plt.xlabel('Date')
    g = plt.ylabel('Cases per week')
    g = plt.title('Actual and forecast values of test set for {}'.format(city))
    g = plt.legend(loc=1, prop={'size': 8})
    g = plt.box(on=None)

    return


def graph_actual_and_forecast_from_test_for_log_data(df_original, city):
    """Create line graphs of actual v. forecast when given the right dataframe"""

    # Import
    import matplotlib.pyplot as plt
    import matplotlib.style as style
    import seaborn as sns

    # Set some Seaborn defaults
    sns.set_style('whitegrid')
    sns.set_context('paper')
    sns.set_palette('muted')

    # Clean up data frame
    df = df_original.copy()
    df.reset_index(inplace=True)  # Reset the index for Seaborn

    # Create plot
    g = plt.figure(figsize=(9, 3), dpi=100)
    g = sns.lineplot(data=df, x='week_start_date', y='actual', label='actual')
    g = sns.lineplot(data=df, x='week_start_date', y='forecast', label='forecast')
    g = plt.xlabel('Date')
    g = plt.ylabel('Log(x+1) of cases per week')
    g = plt.title('Actual and forecast values of log(+1) tranformed test set for {}'.format(city))
    g = plt.legend(loc=1, prop={'size': 8})
    g = plt.box(on=None)

    return

def tranform_back_from_log_x_plus_1(df):
    """Returns a new data frames that's transforms both actual and forecast back from log(x+1)"""

    # Import
    import numpy as np

    forc_df = df.copy()
    forc_df['expm1_act'] = np.expm1(forc_df['actual'])
    forc_df['expm1_forc'] = np.expm1(forc_df['forecast'])
    forc_df.drop(['actual', 'forecast'], axis=1, inplace=True)
    forc_df.rename(columns={'expm1_act': 'actual', 'expm1_forc': 'forecast'}, inplace=True)

    return forc_df


def combine_actual_and_forecast_into_single_df(actual, forecast):
    """Returns a single dataframe with an actual and forecast column """

    import pandas as pd

    actual_df = actual.copy()
    actual_df['forecast'] = pd.to_numeric(forecast)
    actual_df.columns = ['actual', 'forecast']

    return actual_df