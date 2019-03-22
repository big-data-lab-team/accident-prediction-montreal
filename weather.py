from bs4 import BeautifulSoup
import pandas as pd
from os import mkdir
from os.path import isdir, isfile
from requests import get
from io import StringIO
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
import pyspark.sql.functions as f
import numpy as np
from pyspark.sql.types import *
import time
import shutil
import math
import os
from accidents_montreal import fetch_accidents_montreal,\
                               extract_accidents_montreal_df
from road_network import fetch_road_network, extract_road_segments_df

COLUMNS_USED = ['Dew Point Temp (°C)',
                'Rel Hum (%)',
                'Wind Dir (10s deg)',
                'Wind Spd (km/h)',
                'Visibility (km)',
                'Stn Press (kPa)',
                'Hmdx',
                'Wind Chill',
                'Temp (°C)']
]

UNAUTH_CHARS = [' ', ',', ';', '{', '}', '(', ')', '\n', '\t', '=']


def extract_date_val(i):
    return udf(lambda val: val.split('/')[i])


def init_spark():
    ''' Initialize a Spark Session
    '''
    return (SparkSession
            .builder
            .getOrCreate())


def get_schema():
    ''' Create the schema needed to create a new dataframe from the
    rows created from the data that has been fetch.
    '''
    col_used = [del_unauthorized_char(c) for c in COLUMNS_USED]
    return StructType([StructField("ACCIDENT_ID", LongType(), False)] + [StructField(c, FloatType(), True) for c in col_used])


def preprocess_accidents(accidents_df):
    ''' Select/build columns of interest and format their names.
    '''
    return (accidents_df.select('ACCIDENT_ID', 'DT_ACCDN', 'LOC_LAT',
                                'LOC_LONG', 'HEURE_ACCDN')
                        .withColumn("year",
                                    extract_date_val(0)(accidents_df.DT_ACCDN))
                        .withColumn("month",
                                    extract_date_val(1)(accidents_df.DT_ACCDN))
                        .withColumn("day",
                                    extract_date_val(2)(accidents_df.DT_ACCDN))
                        .withColumn("HEURE_ACCDN",
                                    f.split(f.col('HEURE_ACCDN'), ':')[0].cast("int"))
                        .drop('DT_ACCDN')
                        .dropna())


def get_weather_(row):
    new_row = get_weather(row.ACCIDENT_ID,
                        row.LOC_LAT,
                        row.LOC_LONG,
                        row.year,
                        row.month,
                        row.day,
                        row.HEURE_ACCDN)
    return new_row


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
        stations = get_stations(lat, long, year, month, hour)
    except Exception as e:
        print(e)
        stations=[]

    stations_weathers = list()
    for station in stations:
        s = get_station_weather(station[0], year, month, day, hour)
        s.loc["station_distance"] = station[1]
        stations_weathers.append(s.to_dict())

    if len(stations_weathers) == 0:
        stations_weathers.append(
            pd.Series([np.nan]*(len(COLUMNS_USED)+1))
            .reindex(COLUMNS_USED+['station_distance']))

    weathers_df = pd.DataFrame(stations_weathers)
    weathers_df = weathers_df.dropna(how='all', subset=weathers_df.columns[:-1])
    weathers_df.columns = [del_unauthorized_char(c) for c in weathers_df.columns]

    def weighted_average(df, c):
        t = df.loc[:, [c, 'station_distance']].dropna().apply(lambda row: pd.Series([row[0] / row[1], 1 / row[1]]), axis=1)
        if len(t) == 0:
            return None
        return float(t[0].sum() / t[1].sum())

    dict_weather = {c: weighted_average(weathers_df, c) for c in weathers_df.columns[:-1]}
    dict_weather['ACCIDENT_ID'] = id
    return Row(**dict_weather)


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
    page = BeautifulSoup(get(url).content, 'lxml')
    stations = (page.body.main
                .find('div',
                      class_='historical-data-results proximity hidden-lg')
                .find_all('form', recursive=False))
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


def fetch_weather_dataset(replace=True):
    ''' Main function which fetch weather data for each row of
    the accident dataset and write the result as parquet data.
    '''

    # init spark
    spark = init_spark()
    sc = spark.sparkContext

    # retrieve accident dataset
    fetch_accidents_montreal()
    accidents_df = extract_accidents_montreal_df(spark)
    clean_acc_df = preprocess_accidents(accidents_df)

    # prepare backup parameters
    backup_file = 'data/weather_backup.parquet'
    if replace and os.path.isdir(backup_file):
        shutil.rmtree(backup_file)

    df_schema = get_schema()
    t = time.time()
    v = (spark.createDataFrame(clean_acc_df
         .sample(0.5)
         .rdd
         .map(lambda row: get_weather_(row)), df_schema)
         .write.parquet(backup_file))
    t = time.time() - t
    print('Done. Processing time:', t)
    return


# fetch_weather_dataset()
