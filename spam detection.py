import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('spam.csv',encoding='latin-1')
print("Original Columns:")
print(df.columns)
df = df[['v1','v2']]
df.columns = ['label','message']
print("\nFirst 5 Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
print("\nLabel Counts:")
print(df['label'].value_counts())
plt.figure(figsize=(5,6))
sns.countplot(x='label',data=df)
plt.title("spam vs ham message")
plt.show()
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
print("\n After label encoder:")
print(df.head())
print("\nSample Messages:")
print(df['message'][0])
print(df['message'][1])
print(df['message'][2])
