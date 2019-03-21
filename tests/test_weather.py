import pandas as pd
from os import mkdir
from os.path import isdir, isfile
from accidents_montreal import *
from weather import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Row
import math
from io import StringIO
from weather import get_weather
import numpy as np
import math


def init_spark():
    return (SparkSession
            .builder
            .getOrCreate())


cols = ['Dew Point Temp (°C)', 'Dew Point Temp Flag', 'Hmdx', 'Hmdx Flag',
        'Rel Hum (%)', 'Rel Hum Flag', 'Stn Press (kPa)', 'Stn Press Flag',
        'Temp (°C)', 'Temp Flag', 'Visibility (km)', 'Visibility Flag',
        'Weather', 'Wind Chill', 'Wind Chill Flag', 'Wind Dir (10s deg)',
        'Wind Dir Flag', 'Wind Spd (km/h)', 'Wind Spd Flag']

def test_preprocess_weathers(weathers):
    ''' From a dataframe containing the weather information from several
    stations, return a pyspark dataframe Row containing the averages/weighted
    averages of the retrieved data.
    '''

    # compute mean of numeric columns
    cols_to_mean = [col for col in NUMERIC_COLS if not col in ['Temp (°C)',
                        'station_denom']]
    means = weathers.loc[:, NUMERIC_COLS].mean()

    # use majority vote on non numeric columns
    non_num_weathers = (weathers.loc[:, NON_NUMERIC_COLS]
                                .apply(lambda col: get_majority_vote(col),
                                       axis=0))

    row = (Row(**dict(zip(non_num_weathers.index.values.tolist()
                           + means.index.values.tolist()
                           + ['Weather', 'Temp (°C)'],
                           non_num_weathers.values.tolist()
                           + means.values.tolist()
                           + [get_general_weather(weathers),
                              get_temperature(weathers)]))))
    print(row)
    return row

def test_weather():
    print('Initialization...')
    spark = init_spark()

    print('Fecthing accidents dataset...')
    fetch_accidents_montreal()
    accidents_df = extract_accidents_montreal_dataframe(spark)

    print('Extract useful columns...')
    acc_df = accidents_df.select('DT_ACCDN',
                                 'LOC_LAT',
                                 'LOC_LONG',
                                 'HEURE_ACCDN')  \
                         .withColumn("year",
                                     extract_date_val(0)(accidents_df
                                                         .DT_ACCDN)) \
                         .withColumn("month",
                                     extract_date_val(1)(accidents_df
                                                         .DT_ACCDN)) \
                         .withColumn("day",
                                     extract_date_val(2)(accidents_df
                                                         .DT_ACCDN)) \
                         .withColumn("HEURE_ACCDN",
                                     extract_hour(accidents_df
                                                         .HEURE_ACCDN)) \
                         .drop('DT_ACCDN') \
                         .replace('Non précisé', '00')

    print('Taking one instance...')
    inst_test = acc_df.limit(1).collect()[0]

    print('Collecting stations for that instance...')
    stations = get_stations(int(inst_test.LOC_LAT),
                            int(inst_test.LOC_LONG),
                            int(inst_test.year),
                            int(inst_test.month),
                            int(inst_test.day))
    print('Found ', len(stations), ' stations!')
    weathers = list()
    for station in stations:
        print('Station ', station[0], '...')
        s = get_station_temp(station[0],
                             int(inst_test.year),
                             int(inst_test.month),
                             int(inst_test.day),
                             int(inst_test.HEURE_ACCDN))
        if all(i == np.nan for i in s):
            print('empty answer')
            continue
        else:
            print('data found!')
            s.loc["station_denom"] = station[1]
            weathers.append(s)

    print('Creating dataframe from collected data')
    weathers_df = pd.DataFrame(weathers, columns=COLUMNS, dtype=object)
    for num_col in NUMERIC_COLS:
        weathers_df[num_col] = pd.to_numeric(weathers_df[num_col], errors='coerce')
    return weathers_df 


def test_get_weather_and_preprocess():
    weathers_df = test_weather()
    test_preprocess_weathers(weathers_df)
    return

test_get_weather_and_preprocess()

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
