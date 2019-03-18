import pandas as pd
from os import mkdir
from os.path import isdir, isfile
from accidents_montreal import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Row
import math
from io import StringIO
from weather import get_weather
import numpy as np
import math


def test_weather():
    cols=['Dew Point Temp (°C)', 'Dew Point Temp Flag', 'Hmdx', 'Hmdx Flag',
          'Rel Hum (%)', 'Rel Hum Flag', 'Stn Press (kPa)', 'Stn Press Flag',
          'Temp (°C)', 'Temp Flag', 'Visibility (km)', 'Visibility Flag',
          'Weather', 'Wind Chill', 'Wind Chill Flag', 'Wind Dir (10s deg)',
          'Wind Dir Flag', 'Wind Spd (km/h)', 'Wind Spd Flag']
    pass


def test_fetch_one_row():
    test_dict = {'Dew Point Temp Flag': 'M',
                 'Hmdx': np.zeros(1),
                 'Hmdx Flag': np.zeros(1),
                 'Rel Hum Flag': 'M',
                 'Stn Press Flag': 'M',
                 'Temp Flag': 'M',
                 'Visibility Flag': np.zeros(1),
                 'Weather': 'Rain,Cloudy',
                 'Wind Chill': np.zeros(1),
                 'Wind Chill Flag': np.zeros(1),
                 'Wind Dir Flag': np.zeros(1),
                 'Wind Spd Flag': np.zeros(1)}

    # get accident data
    fetch_accidents_montreal()
    spark = (SparkSession
             .builder
             .appName("Road accidents prediction")
             .getOrCreate())
    df = extract_accidents_montreal_dataframe(spark)
    first_acc = df.filter('NO_SEQ_COLL == "SPVM _ 2015 _ 18203"').collect()[0]

    # get weather data
    new_row = get_weather(first_acc.LOC_LAT, first_acc.LOC_LONG, 2006, 5, 2, 0)
    new_dict = dict()
    for key in new_row.asDict().keys():
        if isinstance(new_row[key], float):
            if math.isnan(new_row[key]):
                new_dict[key] = np.zeros(1)  # np.nan not comparable
        else:
            new_dict[key] = new_row[key]

    # result
    assert new_dict == test_dict
