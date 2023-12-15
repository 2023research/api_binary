import requests
import pandas as pd
url = "http://54.253.228.31:8000/binaryclassfier"
emails = pd.read_csv('test.csv')
# print (emails)
for i, email in enumerate(emails['text'].values):
    params = {"email": email}
    response = requests.post(url, params=params)

    print('ground truth:', emails['label'][i],response.text)
