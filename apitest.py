import requests
import pandas as pd
url = "http://54.253.228.31:8000/classfier"
emails = pd.read_csv('test.csv')
# print (emails)
for i, email in enumerate(emails['text'].values):
    if i<10:
        params = {"email": email}
        response = requests.post(url, params=params)

        print('ground truth:', emails['label'][i], response.text)

df=pd.read_csv('./results/126_cate_data.csv')
    
for i, email in enumerate(df['Body'].values):
    if i<10:
        params = {"email": email}
        response = requests.post(url, params=params)
        print('ground truth:', emails['label'][i], response.text)
