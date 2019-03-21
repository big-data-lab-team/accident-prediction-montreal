from bs4 import BeautifulSoup
import pandas as pd
from os import mkdir
from os.path import isdir, isfile
from requests import get
from io import StringIO
from pyspark.sql import Row
from pyspark.sql.functions import udf
import numpy as np
from pyspark.sql.types import *
import time
import shutil
import math


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


UNAUTH_CHARS = [' ', ',', ';', '{', '}', '(', ')', '\n', '\t', '=']


NB_ELEMENTS_TREATED = 0.0


def extract_date_val(i):
    return udf(lambda val: val.split('/')[i])


@udf
def extract_hour(val):
    return val.split('-')[0].split(':')[0]


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
    non_num = ['TempFlag',
               'DewPointTempFlag',
               'RelHumFlag',
               'WindDirFlag',
               'WindSpdFlag',
               'VisibilityFlag',
               'StnPressFlag',
               'HmdxFlag',
               'WindChillFlag']

    num = ['DewPointTemp°C',
           'RelHum%',
           'WindDir10sdeg',
           'WindSpdkm/h',
           'Visibilitykm',
           'StnPresskPa',
           'Hmdx',
           'WindChill'] + ['Temp°C']

    return StructType([StructField(c, StringType(), True) for c in non_num]
                      + [StructField(c, FloatType(), True) for c in num]
                      + [StructField('Weather',
                                     ArrayType(elementType=StringType()),
                                     True)])


def preprocess_accidents(accidents_df):
    ''' Select/build columns of interest and format their names.
    '''
    return (accidents_df.select('DT_ACCDN', 'LOC_LAT',
                                'LOC_LONG', 'HEURE_ACCDN')
                        .withColumn("year",
                                    extract_date_val(0)(accidents_df.DT_ACCDN))
                        .withColumn("month",
                                    extract_date_val(1)(accidents_df.DT_ACCDN))
                        .withColumn("day",
                                    extract_date_val(2)(accidents_df.DT_ACCDN))
                        .withColumn("HEURE_ACCDN",
                                    extract_hour(accidents_df.HEURE_ACCDN))
                        .drop('DT_ACCDN')
                        .replace('Non précisé', '00'))


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
    if os.path.isdir(backup_file):
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


def get_weather_(row):
    new_row = get_weather(row.LOC_LAT,
                          row.LOC_LONG,
                          row.year,
                          row.month,
                          row.day,
                          row.HEURE_ACCDN)

    global NB_ELEMENTS_TREATED
    NB_ELEMENTS_TREATED += 1
    print('progress: ', NB_ELEMENTS_TREATED*100/149886, '%')
    return new_row


def get_general_weather(weathers):
    ''' Get general weather information from multiple stations.
    '''
    print(weathers.loc[:, 'Weather'])
    result = (weathers.loc[:, 'Weather']
                      .dropna())

    print('after drop na : ', result)
    result = ','.join(result.drop_duplicates()
                      .values
                      .tolist())

    if len(result) == 0:
        return [None]
    else:
        return result


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
        return None


def del_unauthorized_char(a_string):
    for c in UNAUTH_CHARS:
        if c in a_string:
            a_string = a_string.replace(c, "")
    return a_string


def preprocess_weathers(weathers):
    ''' From a dataframe containing the weather information from several
    stations, return a pyspark dataframe Row containing the averages/weighted
    averages of the retrieved data.
    '''
    # compute mean of numeric columns
    cols_to_mean = [col for col in NUMERIC_COLS
                    if col not in ['Temp (°C)',
                                   'station_denom']]
    print('compute means')
    means = weathers.loc[:, cols_to_mean].mean()
    means_values = [None if math.isnan(i) else i
                    for i in means.values.tolist()]

    # use majority vote on non numeric columns
    print('do majority voting')
    non_num_weathers = (weathers.loc[:, NON_NUMERIC_COLS]
                                .apply(lambda col: get_majority_vote(col),
                                       axis=0))

    non_num_indexes = [del_unauthorized_char(s) for s
                       in non_num_weathers.index.values.tolist()]

    num_indexes = [del_unauthorized_char(s) for s
                   in means.index.values.tolist()]

    print('create new row')
    row = (Row(**dict(zip(non_num_indexes
                          + num_indexes
                          + ['Weather', 'Temp°C'],
                          non_num_weathers.values.tolist()
                          + means_values
                          + [get_general_weather(weathers),
                             get_temperature(weathers)]))))
    print('send result', row)
    return row


def clean_new_dict(associated_dict):
    for key in list(associated_dict.keys()):
        if isinstance(associated_dict[key],
                      float) and math.isnan(associated_dict[key]):
            associated_dict[key] = None

    # drop unuseful columns
    bad_keys = [key for key in list(associated_dict.keys())
                if key not in COLUMNS]
    print('bad keys', bad_keys)
    for key in bad_keys:
        del associated_dict[key]

    # add missing columns
    missing_keys = [key for key in COLUMNS
                    if key not in list(associated_dict.keys())]
    print('missing_keys', missing_keys)
    for key in missing_keys:
        associated_dict[key] = None

    if associated_dict['Weather'] == '':
        associated_dict['Weather'] = None

    print('clean dict', associated_dict)
    return associated_dict


def empty_row():
    print('get index')
    indexes = [del_unauthorized_char(s) for s in COLUMNS
               if not s == 'station_denom']
    print('get values')
    values = len(NON_NUMERIC_COLS) * [None]  \
        + len(NUMERIC_COLS) * [None] + [[None]]
    print('return row')
    return Row(**dict(zip(indexes, values)))


def get_weather(lat, long, year, month, day, hour):
    ''' Get the weather at a given location at a given time.
    '''
    print('input:', lat, long, year, month, day, hour)
    print('input types:', type(lat), type(long), type(year), type(month),
          type(day), type(hour))
    input_ = [None if (isinstance(x, float) and math.isnan(x)) else int(x)
              for x in [lat, long, year, month, day, hour]]
    if any(i is None for i in input_[:len(input_)]):  # all but hour
        return empty_row()

    stations = get_stations(input_[0], input_[1], input_[2],
                            input_[3], input_[4])
    weathers = list()
    for station in stations:
        print('Station ', station[0], '...')
        s = get_station_temp(station[0],
                             input_[2], input_[3], input_[4],
                             input_[5])

        if all(i is None for i in s) or all((isinstance(i, float)
                                            and math.isnan(i)) for i in s):
            continue
        else:
            print('data found!')
            s.loc["station_denom"] = station[1]
            associated_dict = s.to_dict()
            print("cleaning dict")
            associated_dict = clean_new_dict(associated_dict)
            weathers.append(associated_dict)

    if len(weathers) == 0:  # in this case, return empty row
        return empty_row()
    else:
        print('build dataframe')
        # print('Creating dataframe from collected data')
        weathers_df = pd.DataFrame(weathers, dtype=object)
        for num_col in NUMERIC_COLS:
            weathers_df[num_col] = pd.to_numeric(weathers_df[num_col],
                                                 errors='coerce')
        print('preprocess')
        return preprocess_weathers(weathers_df)


def get_pandas_dataframe(url):
    ''' Get pandas dataframe from retrieved csv file using URL argument.
    '''
    # print('Fetching...')
    csvfile = get(url).text
    # print('Done. Extracting data...')
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
        return None

    if not isdir('data/weather/'):
        mkdir('data/weather/')

    """df.to_hdf(cache_file_path, key='w')
    df.to_hdf(cache_file_path, key='w')"""

    if len(df.columns) > 4:
        if hour is not None:
            return (df.loc[f'{year}-{month}-{day} {hour}:00']
                    .drop(['Year', 'Month', 'Day', 'Time']))
        else:
            return (df.iloc[0]
                    .drop(['Year', 'Month', 'Day', 'Time']))
    else:
        return None


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
        return None


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
