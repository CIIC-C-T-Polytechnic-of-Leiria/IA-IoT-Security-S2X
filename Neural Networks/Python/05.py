# Logging: OFF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Code
import pandas as pd

pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 10)

df = pd.read_csv("https://data.heatonresearch.com/data/t81-558/jh-simple-dataset.csv", na_values=['NA','?'])

dummies = pd.get_dummies(df['job'], prefix="job")
df = pd.concat([df,dummies],axis=1)
df.drop('job', axis=1, inplace=True)

pd.set_option('display.max_columns', 5)
pd.set_option('display.max_rows', 10)

print(df)
