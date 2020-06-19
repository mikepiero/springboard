import my_func as dfn
import pandas as pd

def test_plus_1():
    assert dfn.test_plus_1(2) == 3

def test_stationarity_test_via_simple_calcs():
    data = [1, 10, 100, 1000]
    temp_df = pd.DataFrame(data)
    temp_df.rename(columns={0: 'total_cases'}, inplace=True)
    assert dfn.stationarity_test_via_simple_calcs(temp_df, 'city') == ('mean_no', 'var_no')
    data = [1, 2, 1, 2]
    temp_df = pd.DataFrame(data)
    temp_df.rename(columns={0: 'total_cases'}, inplace=True)

assert dfn.stationarity_test_via_simple_calcs(temp_df, 'city') == ('mean_yes', 'var_yes')
