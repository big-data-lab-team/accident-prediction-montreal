from bs4 import BeautifulSoup
import pandas as pd
from os import mkdir
from os.path import isdir, isfile
from requests import get
from io import StringIO
import math
from pyspark.sql import Row
from pyspark.sql.functions import udf
import numpy as np


COLUMNS = ['Temp (°C)',
           'Temp Flag',
           'Dew Point Temp (°C)',
           'Dew Point Temp Flag',
           'Rel Hum (%)',
           'Rel Hum Flag',
           'Wind Dir (10s deg)',
           'Wind Dir Flag',
           'Wind Spd (km/h)',
           'Wind Spd Flag',
           'Visibility (km)',
           'Visibility Flag',
           'Stn Press (kPa)',
           'Stn Press Flag',
           'Hmdx',
           'Hmdx Flag',
           'Wind Chill',
           'Wind Chill Flag',
           'Weather',
           'station_denom']

NUMERIC_COLS = ['Dew Point Temp (°C)',
                'Rel Hum (%)',
                'Wind Dir (10s deg)',
                'Wind Spd (km/h)',
                'Visibility (km)',
                'Stn Press (kPa)',
                'Hmdx',
                'Wind Chill',
                'Temp (°C)',
                'station_denom']

NON_NUMERIC_COLS = ['Temp Flag',
                    'Dew Point Temp Flag',
                    'Rel Hum Flag',
                    'Wind Dir Flag',
                    'Wind Spd Flag',
                    'Visibility Flag',
                    'Stn Press Flag',
                    'Hmdx Flag',
                    'Wind Chill Flag']

def extract_date_val(i):
    return udf(lambda val: val.split('/')[i])


@udf
def extract_hour(val):
    return val.split('-')[0].split(':')[0]


def get_weather_(row):
    try:
        return get_weather(int(row.LOC_LAT),
                           int(row.LOC_LONG),
                           int(row.year),
                           int(row.month),
                           int(row.day),
                           int(row.HEURE_ACCDN))
    except Exception as e:
        print('An error occured.')
        print(e)
        return row


def get_general_weather(weathers):
    ''' Get general weather information from multiple stations.
    '''
    return (','.join(weathers.loc[:, 'Weather']
                             .dropna()
                             .drop_duplicates()
                             .values
                             .tolist()))


def compute_temp(row):
    ''' Compute numerator and denominator for weighted average of temperature.
    '''
    return pd.Series([row[0] / row[1], 1 / row[1]])


def get_temperature(weathers):
    ''' Get weighted average temperature from multiple stations.
    '''
    temperatures = (weathers[['Temp (°C)', 'station_denom']]
                    .dropna()
                    .apply(lambda row: compute_temp(row), axis=1))
    if len(temperatures) >= 2:
        return temperatures[0].sum() / temperatures[1].sum()
    else:
        return np.nan


def preprocess_weathers(weathers):
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


def get_weather(lat, long, year, month, day, hour):
    ''' Get the weather at a given location at a given time.
    '''
    stations = get_stations(lat, long, year, month, day)
    weathers = list()
    for station in stations:
        s = get_station_temp(station[0], year, month, day, hour)
        if all(i == np.nan for i in s):  # empty answer
            continue
        else:
            s.loc["station_denom"] = station[1]
            weathers.append(s)

    weathers_df = pd.DataFrame(weathers, columns=COLUMNS, dtype=object)

    if len(weathers) == 0:
        return np.nan
    else:
        for num_col in NUMERIC_COLS:
            weathers_df[num_col] = pd.to_numeric(weathers_df[num_col], errors='coerce')
        return preprocess_weathers(weathers_df)


def get_pandas_dataframe(url):
    ''' Get pandas dataframe from retrieved csv file using URL argument.
    '''
    print('Fetching...')
    csvfile = get(url).text
    print('Done. Extracting data...')
    df = None
    with StringIO(csvfile) as csvfile:
        skip_header(csvfile)
        df = pd.read_csv(csvfile,
                         index_col='Date/Time',
                         parse_dates=['Date/Time'])
    return df


def get_station_temp(station_id, year, month, day, hour):
    ''' Get temperature for a given station (given its station ID).
    '''
    cache_file_path = f'data/weather/s{station_id}_{year}_{month}.h5'

    """if isfile("cache_file_path"):
        df = pd.read_hdf(cache_file_path, key='w')
    else:"""

    url = (f'http://climate.weather.gc.ca/climate_data/bulk_data_e.html?'
           f'format=csv&stationID={station_id}&Year={year}'
           f'&Month={month}&Day={day}&'
           f'timeframe=1&submit=Download+Data')
    try:
        df = get_pandas_dataframe(url)
    except Exception as e:
        print('Unable to fetch:', url)
        print(e)
        return np.nan

    if not isdir('data/weather/'):
        mkdir('data/weather/')

    """df.to_hdf(cache_file_path, key='w')
    df.to_hdf(cache_file_path, key='w')"""

    if len(df.columns) > 4:
        return (df.loc[f'{year}-{month}-{day} {hour}:00']
                .drop(['Year', 'Month', 'Day', 'Time']))
    else:
        return np.nan


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


def get_majority_vote(col):
    ''' Return the most frequent item of a pandas dataframe column.
    Return a list of elements in case of a tie.
    Args:
        col:
    To be used on non numerical columns.
    '''
    vals = col.dropna().drop_duplicates().tolist()
    if len(vals) > 0:
        e = max(set(vals), key=vals.count)
        return ','.join([str(v) for v in vals
                         if vals.count(e) == vals.count(v)])
    else:
        return ''


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
    ''' Utility function for get_station_temp.
    '''
    n_emptyLineMet = 0
    while n_emptyLineMet < 2:
        if file.readline() == '\n':
            n_emptyLineMet += 1
