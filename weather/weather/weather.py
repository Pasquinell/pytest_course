import pandas as pd 
import numpy as np

def read_weather_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df=df.dropna()
    df['temperature']=df['temperature'].apply(lambda x: np.round(x,1))
    return df

def calculate_mean_temperature(df):
    mean_temperature = np.mean(df['temperature'])
    return mean_temperature

