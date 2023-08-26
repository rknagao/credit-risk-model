import numpy as np
import pandas as pd


def feat_eng_grave_default(df):

    # Create dummies indicating the worse case of default.
    #df['gravest_default_no_default'] = np.where((df['NumberOfTimes90DaysLate'] == 0) &
    #                                            (df['NumberOfTime60-89DaysPastDueNotWorse'] == 0) &
    #                                            (df['NumberOfTime30-59DaysPastDueNotWorse'] == 0), 1, 0)
    df['gravest_default_def>90'] = np.where(df['NumberOfTimes90DaysLate'] > 0,1, 0)
    df['gravest_default_def60~89'] = np.where((df['NumberOfTimes90DaysLate'] == 0) & (df['NumberOfTime60-89DaysPastDueNotWorse'] > 0),1, 0)
    df['gravest_default_def<30'] = np.where((df['NumberOfTimes90DaysLate'] == 0) &
                                            (df['NumberOfTime60-89DaysPastDueNotWorse'] == 0) &
                                            (df['NumberOfTime30-59DaysPastDueNotWorse'] > 0), 1, 0)

    return df



def feat_eng_non_secure_credit_usage(df):
    # Truncate to 1.
    df['RevolvingUtilizationOfUnsecuredLines_trunc'] = np.where(df['RevolvingUtilizationOfUnsecuredLines'] > 1, 1, df['RevolvingUtilizationOfUnsecuredLines'])
    
    # Create category based on train data.
    #df['RevolvingUtilizationOfUnsecuredLines_trunc_perc'] = np.select(
    #    [
    #        df['RevolvingUtilizationOfUnsecuredLines_trunc'].between(0, 0.029728),
    #        df['RevolvingUtilizationOfUnsecuredLines_trunc'].between(0.029729, 0.154026),
    #        df['RevolvingUtilizationOfUnsecuredLines_trunc'].between(0.154027, 0.558113),
    #        df['RevolvingUtilizationOfUnsecuredLines_trunc'].between(0.558114, 1),
    #    ], ['Q1', 'Q2', 'Q3', 'Q4'], 'error'
    #)
    #try:
    #    assert 'error' in df['RevolvingUtilizationOfUnsecuredLines_trunc_perc']
    #except Exception as e:
    #    print(f"Error on credit usage feature engineering: {e}") 
    #
    ## Dummify.
    #df_nonsecure_credit = pd.get_dummies(data=df[['RevolvingUtilizationOfUnsecuredLines_trunc_perc']], prefix='nonsecure_credit_usage', dtype='int')
    #df_nonsecure_credit = df_nonsecure_credit.drop(columns='nonsecure_credit_usage_Q3')
 
    #df = pd.concat([df, df_nonsecure_credit], axis=1)
    #df = df.drop(columns='RevolvingUtilizationOfUnsecuredLines_trunc_perc')
    
    df['nonsecure_credit_usage_Q1'] = np.where(df['RevolvingUtilizationOfUnsecuredLines_trunc'].between(0, 0.029728), 1, 0)
    df['nonsecure_credit_usage_Q2'] = np.where(df['RevolvingUtilizationOfUnsecuredLines_trunc'].between(0.029729, 0.154026), 1, 0)
    #df['nonsecure_credit_usage_Q3'] = np.where(df['RevolvingUtilizationOfUnsecuredLines_trunc'].between(0.154027, 0.558113), 1, 0)
    df['nonsecure_credit_usage_Q4'] = np.where(df['RevolvingUtilizationOfUnsecuredLines_trunc'].between(0.558114, 1), 1, 0)

    return df



def feat_eng_age(df):

    df['age_decade_20s'] = np.where(df['age'] <= 29, 1, 0)
    df['age_decade_30s'] = np.where(df['age'].between(30, 39), 1, 0)
    df['age_decade_40s'] = np.where(df['age'].between(40, 49), 1, 0)
    # If all other columns are 0, them 50s.
    #df['age_decade_50s'] = np.where(df['age'].between(50, 59), 1, 0)
    df['age_decade_60s'] = np.where(df['age'].between(60, 69), 1, 0)
    df['age_decade_>70s'] = np.where(df['age'] >= 70, 1, 0)

    return df



def feat_eng_past_default_2m(df):
    df['NumberOfTime30-59DaysPastDueNotWorse_dummy'] = np.where(df['NumberOfTime30-59DaysPastDueNotWorse'] == 0, 0, 1)

    return df



def feat_eng_income(df):

    # Identifying income omission.
    df['income_omission'] = np.where(df['MonthlyIncome'].notnull(), 0, 1)

    ## Feature engineering for Income variable.
    #df['MonthlyIncome_quant'] = np.select(
    #    [
    #        df['MonthlyIncome'].between(0, 3400),
    #        df['MonthlyIncome'].between(3401, 5400),
    #        df['MonthlyIncome'].between(5401, 8249),
    #        df['MonthlyIncome'].between(8250, 9999999999)
    #    ], ['Q1', 'Q2', 'Q3', 'Q4'], 'error'
    #)
    #try:
    #    assert 'error' not in df['MonthlyIncome_quant']
    #except Exception as e:
    #    print(f"Error on income's feature engineering: {e}") 
    #
    # Fill missing values with Q3 median.
    df.loc[df['MonthlyIncome'].isnull(), 'MonthlyIncome'] = 6609
    #
    #df_income = pd.get_dummies(data=df[['MonthlyIncome_quant']], prefix='income', dtype='int')
    #df_income = df_income.drop(columns='income_Q3')
#
    #df = pd.concat([df, df_income], axis=1)
    #df = df.drop(columns='MonthlyIncome_quant')

    df['income_Q1'] = np.where(df['MonthlyIncome'].between(0, 3400), 1, 0)
    df['income_Q2'] = np.where(df['MonthlyIncome'].between(3401, 5400), 1, 0)
    #df['income_Q3'] = np.where(df['MonthlyIncome'].between(5401, 8249), 1, 0)
    df['income_Q4'] = np.where(df['MonthlyIncome'].between(8250, 9999999999), 1, 0)

    return df



def feat_eng_past_default_more4m(df):

    # Truncate the number of past defaults.
    #df['number_past_default'] = np.select(
    #    [
    #        df['NumberOfTimes90DaysLate'] == 0,
    #        df['NumberOfTimes90DaysLate'] == 1,
    #        df['NumberOfTimes90DaysLate'] > 0,
    #    ], ['never', 'once', 'more_than_once'], '?'
    #)
#
    ## Dummify.
    #df_default = pd.get_dummies(data=df[['number_past_default']], prefix='number_past_default', dtype='int') 
    #df_default = df_default.drop(columns='number_past_default_never')
#
    #df = pd.concat([df, df_default], axis=1)
    #df = df.drop(columns='number_past_default')

    #df['number_past_default_never'] = np.where(df['NumberOfTimes90DaysLate'] == 0, 1, 0)
    df['number_past_default_once'] = np.where(df['NumberOfTimes90DaysLate'] == 1, 1, 0)
    df['number_past_default_more_than_once'] = np.where(df['NumberOfTimes90DaysLate'] > 0, 1, 0)
         
    return df



def feat_eng_dependents(df):
    # Fill missing values with median.
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
    return df