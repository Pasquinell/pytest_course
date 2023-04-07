import pytest
import pandas as pd
import numpy as np
from weather.weather import read_weather_data, clean_data, calculate_mean_temperature

@pytest.fixture
def sample_weather_data():
    data = {'date': ['2021-01-01', '2021-01-02', '2021-01-03'],
            'temperature': [22.345, 23.678, np.nan]}
    return pd.DataFrame(data)

def test_clean_data(sample_weather_data):
    cleaned_data = clean_data(sample_weather_data)
    
    assert cleaned_data.isna().sum().sum() == 0
    assert cleaned_data['temperature'].equals(pd.Series([22.3, 23.7], name="temperature"))

def test_calculate_mean_temperature(sample_weather_data):
    cleaned_data = clean_data(sample_weather_data)
    mean_temperature = calculate_mean_temperature(cleaned_data)
    
    assert np.isclose(mean_temperature, 23.0)


# test_my_module.py
import pytest
import pandas as pd
import tempfile
import os
from unittest.mock import mock_open, patch


def create_temp_csv(content):
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
        f.write(content)
        return f.name

def test_weather_data():
    # Create a temporary CSV file
    content = "col1,col2\n1,2\n3,4"
    temp_csv_file = create_temp_csv(content)

    # Test the load_csv function
    df = read_weather_data(temp_csv_file)
    assert df.equals(pd.DataFrame({'col1': [1, 3], 'col2': [2, 4]}))