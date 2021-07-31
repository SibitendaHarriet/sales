import dvc.api
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import pandas as pd
import numpy as np
import dvc.api
import mlflow
# import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from math import ceil
from statsmodels.stats.proportion import proportions_ztest, proportion_confint

#Load training data
path = r'C:\Users\DELL\sales\data\store.csv'
repo = r'C:\Users\DELL\sales'
version = 'vstoring'
data_url = dvc.api.get_url(
path = path,
repo = repo,
rev=version
)
df=pd.read_csv(data_url)
print(df.head())

mlflow.set_experiment('demo')

# Log data params
mlflow.log_param('data_url', data_url)
mlflow.log_param('data_version', version)
mlflow.log_param('input_rows', df.shape[0])
mlflow.log_param('input_cols', df.shape[1])

# Sample size calculation
effect_size = sms.proportion_effectsize(0.20, 0.25)
required_n = sms.NormalIndPower().solve_power(
    effect_size,
    power=0.8,
    alpha=0.05,
    ratio=1)
required_n = ceil(required_n)