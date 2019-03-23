from bs4 import BeautifulSoup
import pandas as pd
from os import mkdir
from os.path import isdir, isfile
from requests import get
from io import StringIO
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, year, month, dayofmonth
from pyspark.sql.types import StructField, FloatType, StructType
import numpy as np
import time
import shutil
import math
import os

COLUMNS_USED = ['Dew Point Temp (°C)',
                'Rel Hum (%)',
                'Wind Dir (10s deg)',
                'Wind Spd (km/h)',
                'Visibility (km)',
                'Stn Press (kPa)',
                'Hmdx',
                'Wind Chill',
                'Temp (°C)']

UNAUTH_CHARS = [' ', ',', ';', '{', '}', '(', ')', '\n', '\t', '=']


def del_unauthorized_char(a_string):
    a_string = a_string.split('(')[0].strip()
    for c in UNAUTH_CHARS:
        if c in a_string:
            a_string = a_string.replace(c, "_")
    return a_string


def get_weather(id, lat, long, year, month, day, hour):
    ''' Get the weather at a given location at a given time.
    '''

    try:
        stations = get_stations(lat, long, year, month, day)
    except Exception as e:
        print(f'Exception while fetching stations for accident {id}:')
        print(e)
        stations = []

    stations_weathers = list()
    for station in stations:
        s = get_station_weather(station[0], year, month, day, hour)
        s.loc["station_distance"] = station[1]
        stations_weathers.append(s.to_dict())

    if len(stations_weathers) == 0:
        stations_weathers.append(
            pd.Series([np.nan]*(len(COLUMNS_USED)+1))
            .reindex(COLUMNS_USED+['station_distance']))

    weather_df = pd.DataFrame(stations_weathers)
    weather_df = weather_df.dropna(how='all', subset=weather_df.columns[:-1])
    weather_df.columns = [del_unauthorized_char(c) for c in weather_df.columns]

    def weighted_average(df, c):
        t = (df.loc[:, [c, 'station_distance']]
             .dropna()
             .apply(lambda r: pd.Series([r[0] / r[1], 1 / r[1]]), axis=1))
        if len(t) == 0:
            return None
        return float(t[0].sum() / t[1].sum())

    weather_cols = weather_df.columns[:-1]
    return {c: weighted_average(weather_df, c) for c in weather_cols}


def get_pandas_dataframe(url):
    ''' Get pandas dataframe from retrieved csv file using URL argument.
    '''
    csvfile = get(url).text
    df = None
    with StringIO(csvfile) as csvfile:
        skip_header(csvfile)
        df = pd.read_csv(csvfile,
                         index_col='Date/Time',
                         parse_dates=['Date/Time'])
    return df


def get_station_weather(station_id, year, month, day, hour):
    ''' Get temperature for a given station (given its station ID).
    '''
    cache_file_path = f'data/weather/s{station_id}_{year}_{month}.parquet'

    if isfile(cache_file_path):
        df = pd.read_parquet(cache_file_path)
    else:
        url = (f'http://climate.weather.gc.ca/climate_data/bulk_data_e.html?'
               f'format=csv&stationID={station_id}&Year={year}'
               f'&Month={month}&Day={day}&'
               f'timeframe=1&submit=Download+Data')
        try:
            df = get_pandas_dataframe(url)
            if not isdir('data/weather/'):
                mkdir('data/weather/')
            df.to_parquet(cache_file_path)
        except Exception as e:
            print('Unable to fetch:', url)
            print(e)
            return pd.Series([np.nan]*len(COLUMNS_USED), index=COLUMNS_USED)

    return df.loc[f'{year}-{month}-{day} {hour}:00'].reindex(COLUMNS_USED)


def get_stations(lat, long, year, month, day):
    ''' Get data from all stations.
    '''
    lat = degree_to_DMS(lat)
    long = degree_to_DMS(long)
    url = (f'http://climate.weather.gc.ca/historical_data/'
           f'search_historic_data_stations_e.html?searchType=stnProx&'
           f'timeframe=1&txtRadius=25&selCity=&selPark=&optProxType=custom&'
           f'txtCentralLatDeg={abs(lat[0])}&txtCentralLatMin={lat[1]}&'
           f'txtCentralLatSec={lat[2]:.1f}&txtCentralLongDeg={abs(long[0])}&'
           f'txtCentralLongMin={long[1]}&txtCentralLongSec={long[2]:.1f}&'
           f'StartYear=1840&EndYear=2019&optLimit=specDate&Year={year}&'
           f'Month={month}&Day={day}&selRowPerPage=100')
    try:
        page = BeautifulSoup(get(url).content, 'lxml')
        stations = (page.body.main
                    .find('div',
                          class_='historical-data-results proximity hidden-lg')
                    .find_all('form', recursive=False))
    except Exception:
        print('Unable to fetch:', url)
        raise
    return [parse_station(s) for s in stations]


def parse_station(s):
    ''' Utility function for get_stations.
    '''
    return (
        int(s.find('input', {'name': 'StationID'})['value']),
        float(s.find_all('div', class_='row',
              recursive=False)[2].div.find_next_sibling().text.strip())
    )


def degree_to_DMS(degree):
    ''' Convert from plain degrees format to DMS format of geolocalization.
    DMS: "Degrees, Minutes, Seconds" is a format for coordinates at the surface
        of earth. Decimal Degrees = degrees + (minutes/60) + (seconds/3600)
        This measure permit to gain in precision when a degree is not precise
        enough.
    '''
    return (int(degree), int(60 * (abs(degree) % 1)),
            ((60 * (abs(degree) % 1)) % 1) * 60)


def skip_header(file):
    ''' Utility function for get_station_weather.
    '''
    n_emptyLineMet = 0
    nb_line = 0
    nb_max_header_lines = 100
    while n_emptyLineMet < 2:
        if file.readline() == '\n':
            n_emptyLineMet += 1
        if nb_line > nb_max_header_lines:
            raise Exception("Invalid file")
        nb_line += 1


def add_weather_to_row(row):
    row_dict = get_weather(
                        row.accident_id,
                        row.loc_lat,
                        row.loc_long,
                        row.year,
                        row.month,
                        row.day,
                        row.hour)
    row_dict.update(row.asDict())
    return Row(**row_dict)


def get_schema_with_weather(schema):
    for c in COLUMNS_USED:
        schema.add(StructField(del_unauthorized_char(c), FloatType(), True))
    # Since columns will be sorted in the rows of the RDD
    schema = StructType(sorted(schema, key=lambda f: f.name))
    return schema


def extract_year_month_day(df):
    return (df
            .withColumn('year', year(col('date')))
            .withColumn('month', month(col('date')))
            .withColumn('day', dayofmonth(col('date'))))


def add_weather_columns(spark, samples):
    ''' Main function which fetch weather data for each row of
    the given dataframe of samples.
    '''

    samples = extract_year_month_day(samples)
    df_schema = get_schema_with_weather(samples.schema)

    return spark.createDataFrame(
            samples.rdd.map(add_weather_to_row),
            df_schema).drop('year', 'month', 'day')
