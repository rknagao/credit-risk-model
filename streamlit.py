import boto3
import pandas as pd
import pickle
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import src.custom_lib as cl

s3 = boto3.client('s3')
bucket = 'rafaelk-credit-risk-model'
key = 'artefacts/model1.pkl'

try:
    response = s3.get_object(Bucket=bucket, Key=key)
    artefact = response['Body'].read()#.decode('utf-8')
    model = pickle.loads(artefact)
    #st.write(' Modelo foi carregado com sucesso!')

except Exception as e:
    print(f"Error reading artifact: {e}")
    #st.write(' Modelo não foi carregado!')

st.write('sanduiche de epa: você abre o pão e fala epa?!')
st.write('by Robert Stern, 2017')
st.write('sanduela de mortandivis somos')

st.markdown('''
# Credit Risk Model            
### How risky is my profile when applying for a loan?            

This model is based on [Kaggle's Give Me Some Credit Challenge](!https://www.kaggle.com/competitions/GiveMeSomeCredit), and its goal is to classify 
different cases of loan applications in term of payment risks. Please try answering the questions below to receive a risk prediction for you financial profile.
            
#### User input:
''')

var1_ammount = st.number_input(
    label='Question 1: how much have you already borrowed? (US$)',
    min_value=0,
    max_value=999999,
    value=1500
    )

var2_credit_limit = st.number_input(
    label='Question 2: how much is your credit limit? (US$)',
    min_value=1,
    max_value=999999,
    value=8000
    )

var3_age = st.number_input(
    label='Question 3: how old are you?',
    min_value=18,
    max_value=120,
    value=35
    )

var4_dependent= st.number_input(
    label='Question 4: how many people financially depend on you (ex: children, elderly)?',
    min_value=0,
    max_value=50,
    value=2
    )

var5_general_loan = st.number_input(
    label='Question 5: how many non-housing related loans (ex: education loan) do you already have?',
    min_value=0,
    max_value=100,
    value=0
    )

var6_mortgage_loan = st.number_input(
    label='Question 6: how many housing related loans (ex: mortgage) do you already have?',
    min_value=0,
    max_value=100,
    value=0
    )

var7_debt_payment = st.number_input(
    label='Question 7: how much are you paying in debt payment monthly? (US$)',
    min_value=0,
    max_value=99999,
    value=0
    )

var8_income= st.number_input(
    label='Question 8: how much is you monthly income? (US$).',
    min_value=1,
    max_value=99999,
    value=4500
    )

var9_late_2m = st.number_input(
    label='Question 9: within the past two years, how many previous debts were paid after 2 months of due date?',
    min_value=0,
    max_value=99999,
    value=0
    )

var10_late_3m = st.number_input(
    label='Question 10: within the past two years, how many previous debts were paid after 3 months of due date?',
    min_value=0,
    max_value=99999,
    value=0
    )

var11_default = st.number_input(
    label='Question 11: within the past two years, how many previous debts were not paid after 3 months of due date?',
    min_value=0,
    max_value=99999,
    value=0
    )


run = st.button("Run prediction")
if run == True:

    df = pd.DataFrame({
        'RevolvingUtilizationOfUnsecuredLines': var1_ammount / var2_credit_limit,
        'age': var3_age,
        'NumberOfTime30-59DaysPastDueNotWorse': var9_late_2m,
        'DebtRatio': var7_debt_payment / var8_income,
        'MonthlyIncome': var8_income,
        'NumberOfOpenCreditLinesAndLoans': var5_general_loan,
        'NumberOfTimes90DaysLate': var11_default,
        'NumberRealEstateLoansOrLines': var6_mortgage_loan,
        'NumberOfTime60-89DaysPastDueNotWorse': var10_late_3m,
        'NumberOfDependents': var4_dependent}
        , index=[0])

    #st.write(df)

    df = cl.feat_eng_non_secure_credit_usage(df)
    df = cl.feat_eng_age(df)
    df = cl.feat_eng_income(df)
    df = cl.feat_eng_past_default_2m(df)
    df = cl.feat_eng_past_default_more4m(df)
    df = cl.feat_eng_grave_default(df)
    df = cl.feat_eng_dependents(df)

    #st.write(df)
    #st.write(df.shape)

    # Reestructure columns order.
    list_order = list(model.feature_names_in_)
    df = df[list_order]

    #list_feat = list(df.columns)
    #list_feat.sort()
    #st.write(list_feat)

    proba = model.predict_proba(df)[:, 1][0]
    
    text_result = None
    if proba < 0.5:
        st.balloons()
        text_result = '''
        ##### Congratulations! ✅
        Your profile is considered low risk (i.e. below 50%).
        '''
    else:
        text_result = '''
        ##### I am sorry ☹️
        Unfortunately, your profile is considered righ risk (i.e. above 50%).
        '''
    st.markdown(f'#### Prediction')
    #st.write(text_result)

    formatted_proba = f'{round(proba * 100, 1)}%'
    col1, col2 = st.columns(2)
    col1.metric(label="Predicted risk:", value=formatted_proba)
    col2.markdown(text_result)
    style_metric_cards()