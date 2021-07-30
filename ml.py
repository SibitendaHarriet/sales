import dvc.api
import pandas as pd
#Load training data
path = r'data\store.csv'
repo = r'C:\Users\DELL\sales'
version = 'vstore'
data_url2 = dvc.api.get_url(
path = path,
repo = repo,
rev=version
)
train_data=pd.read_csv(data_url2)
print(train_data.head())
