import pandas as pd

df = pd.read_csv('diabetes_prediction_dataset.csv')
print(df['diabetes'].value_counts())