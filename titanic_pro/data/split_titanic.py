import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# importing data
df = pd.read_csv("titanic_all.csv")

# using the train test split function
train_df, test_df = train_test_split(df, random_state=104, test_size=0.10, shuffle=True)

train_df.to_csv("train.csv", index=False)
test_df.to_csv("test.csv", index=False)
