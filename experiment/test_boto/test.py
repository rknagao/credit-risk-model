import boto3
import streamlit as st

s3 = boto3.client('s3')
bucket = 'rafaelk-credit-risk-model'
key = 'data.csv'

response = s3.get_object(Bucket=bucket, Key=key)
data = response['Body'].read().decode('utf-8')
print(data)

print('sucesso')
st.write('hello world')
st.write(data)