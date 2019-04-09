from bs4 import BeautifulSoup
import datetime
import pandas as pd
from io import StringIO
import math
import numpy as np
from os import mkdir
from os.path import isdir, isfile
from pyspark.sql import Row, SparkSession
from pyspark.sql.functions import udf, col, year, month, dayofmonth, explode
from pyspark.sql.types import StructField, FloatType, StructType, \
                              IntegerType, ArrayType, DateType
import re
from requests import get
import shutil
import time
from utils import get_with_retry as get
from workdir import workdir


COLUMNS_USED = ['Dew Point Temp (°C)',
                'Rel Hum (%)',
                'Wind Dir (10s deg)',
                'Wind Spd (km/h)',
                'Visibility (km)',
                'Stn Press (kPa)',
                'Hmdx',
                'Wind Chill',
                'Temp (°C)']


def get_weather_df(spark, accident_df):
    ''' Combine weather information and return a dataframe with the weather
        data for all useful weather station for each hour of the 5 years and
        the stations GPS coordinates
    '''
    stations_id_df = get_weather_station_id_df(spark, accident_df)
    stations_coord_df = get_weather_station_coords_df(spark, stations_id_df)
    stations_weather_df = get_weather_station_weather_df(spark, stations_id_df)
    return stations_weather_df.join(stations_coord_df, 'station_id')


def get_weather_station_id(lat, long, year, month, day):
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

    return [int(s.find('input', {'name': 'StationID'})['value'])
            for s in stations]


def get_weather_station_id_df(spark, accident_df):
    ''' Generate dataframe with the station ids of all stations necessary for
        given accident dataframe
    '''
    cache_file = 'data/weather_stations_id.parquet'
    if isdir(cache_file):
        print('Skip downloading weather station ids: already done')
        return spark.read.parquet(cache_file)

    get_weather_station_id_udf = \
        udf(get_weather_station_id, ArrayType(IntegerType()))

    df = (accident_df
          .select(get_weather_station_id_udf(
                col('loc_lat'),
                col('loc_long'),
                year('date'),
                month('date'),
                dayofmonth('date')
                ).alias('stations'))
          .select(explode(col('stations').alias('station_id')))
          .distinct())

    df.write.parquet(cache_file)
    return df


def get_station_weather_month(station_id, year, month):
    ''' Fetch the weather data for a given station (given its station ID) and a
        given month
    '''
    url = (f'http://climate.weather.gc.ca/climate_data/bulk_data_e.html?'
           f'format=csv&stationID={station_id}&Year={year}'
           f'&Month={month}&Day=14&'
           f'timeframe=1&submit=Download+Data')
    df = get_pandas_dataframe(url)[['Day', 'Time']+COLUMNS_USED]
    return [(
        int(r['Day']),
        int(r['Time'].split(':')[0]),
        float(r['Dew Point Temp (°C)']),
        float(r['Rel Hum (%)']),
        float(r['Wind Dir (10s deg)']),
        float(r['Wind Spd (km/h)']),
        float(r['Visibility (km)']),
        float(r['Stn Press (kPa)']),
        float(r['Hmdx']),
        float(r['Wind Chill']),
        float(r['Temp (°C)'])
    ) for i, r in df.iterrows()]


def get_weather_station_weather_df(spark, stations_id):
    ''' Download the weather station data during all hours of
        the 5 years for given station ids and return a dataframe
    '''
    cache_file = 'data/weather_stations.parquet'
    if isdir(cache_file):
        print('Skip downloading weather station: already done')
        return spark.read.parquet(cache_file)

    get_station_weather_month_udf = \
        udf(get_station_weather_month, ArrayType(StructType([
            StructField('day', IntegerType()),
            StructField('hour', IntegerType()),
            StructField('dew_point_temp', FloatType()),
            StructField('rel_hum', FloatType()),
            StructField('wind_dir', FloatType()),
            StructField('wind_spd', FloatType()),
            StructField('visibility', FloatType()),
            StructField('stn_press', FloatType()),
            StructField('hmdx', FloatType()),
            StructField('wind_chill', FloatType()),
            StructField('temp', FloatType())
        ])))

    month_per_year_df = spark.createDataFrame(zip(range(1, 13),), ['month'])
    years_df = spark.createDataFrame(zip(range(2012, 2018),), ['year'])
    months_df = years_df.crossJoin(month_per_year_df)
    stations_months_df = stations_id.crossJoin(months_df)

    c = col('col')

    def create_date(year, month, day):
        return datetime.datetime.strptime(f'{year}-{month}-{day}', "%Y-%m-%d")
    create_date_udf = udf(create_date, DateType())

    df = (stations_months_df
          .withColumn('weather', get_station_weather_month_udf('station_id',
                                                               'year',
                                                               'month'))
          .select('station_id', 'year', 'month', explode('weather'))
          .select('station_id',
                  create_date_udf('year', 'month', c['day']).alias('date'),
                  c['hour'].alias('hour'),
                  c['dew_point_temp'].alias('dew_point_temp'),
                  c['rel_hum'].alias('rel_hum'),
                  c['wind_dir'].alias('wind_dir'),
                  c['wind_spd'].alias('wind_spd'),
                  c['visibility'].alias('visibility'),
                  c['stn_press'].alias('stn_press'),
                  c['hmdx'].alias('hmdx'),
                  c['wind_chill'].alias('wind_chill'),
                  c['temp'].alias('temp')))

    df.write.parquet(cache_file)

    return df


def get_weather_station_coords(station_id):
    ''' Download the GPS coordinates for each given station id and return
        a dataframe with the columns: station_id, lat and long
    '''
    year, month = 2013, 1

    url = (f'http://climate.weather.gc.ca/climate_data/hourly_data_e.html?'
           f'timeframe=1&Year={year}&Month={month}&Day=14&'
           f'StationID={station_id}&Prov=QC&urlExtension=_e.html&'
           f'searchType=stnProv&optLimit=specDate&StartYear=1840&EndYear=2019&'
           f'lstProvince=QC')
    web_page = BeautifulSoup(get(url).content, 'lxml')
    coords_div = web_page.body.find('div', class_='metadata').div
    lat = coords_div.div.div.find_all('div', recursive=False)[1].text
    long = (coords_div.find_all('div', recursive=False)[1].div
            .find_all('div', recursive=False)[1].text)
    return DMS_to_degree(lat), DMS_to_degree(long)


def get_weather_station_coords_df(spark, stations_id):
    cache_file = 'data/station_coords.parquet'
    if isdir(cache_file):
        print('Skip downloading weather station coordinates: already done')
        return spark.read.parquet(cache_file)

    get_weather_station_coords_udf = udf(
            get_weather_station_coords,
            StructType([
                StructField('lat', FloatType()),
                StructField('long', FloatType())
            ]))
    df = (stations_id
          .withColumn('coords', get_weather_station_coords_udf('station_id'))
          .select(
            'station_id',
            col('coords')['lat'].alias('station_lat'),
            col('coords')['long'].alias('station_long')
          ))

    df.write.parquet(cache_file)

    return df


def degree_to_DMS(degree):
    ''' Convert from plain degrees format to DMS format of geolocalization.
    DMS: "Degrees, Minutes, Seconds" is a format for coordinates at the surface
        of earth. Decimal Degrees = degrees + (minutes/60) + (seconds/3600)
        This measure permit to gain in precision when a degree is not precise
        enough.
    '''
    return (int(degree), int(60 * (abs(degree) % 1)),
            ((60 * (abs(degree) % 1)) % 1) * 60)


def DMS_to_degree(coord):
    match = re.match(
        " ([0-9]{1,2})°([0-9]{1,2})'([0-9]{1,2}\\.[0-9]{1,3})\" [NSWE]", coord)
    if match is None:
        raise ValueError('Invalid GPS coordinate')
    d = int(match[1])
    m = int(match[2])
    s = float(match[3])
    return d + m / 60 + s / 3600


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
