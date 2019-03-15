from urllib.request import urlopen
from bs4 import BeautifulSoup
import pandas as pd
from os import mkdir
from os.path import isdir, isfile
from requests import get

def get_weather(lat, long, year, month, day, hour):
    ''' Get the weather at a given location at a given time.
    '''
    stations = get_stations(lat, long, year, month, day)
    weighted_average_num = 0
    weighted_average_denum = 0
    for station in stations:
        temp = get_station_temp(station[0], year, month, day, hour)
        if not math.isnan(temp):
            weighted_average_num += temp / station[1]
            weighted_average_denum += 1 / station[1]
    return weighted_average_num/weighted_average_denum


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


def get_station_temp(id, year, month, day, hour):
    ''' Get temperature for a given station.
    '''
    cache_file_path = f'data/weather/s{id}_{year}_{month}.h5'
    if isfile(cache_file_path):
        return (pd.read_hdf(cache_file_path, key='w')
                .loc[f'{year}-{month}-{day} {hour}:00'][0])
    url = (f'http://climate.weather.gc.ca/climate_data/bulk_data_e.html?'
           f'format=csv&stationID={id}&Year={year}&Month={month}&Day={day}&'
           f'timeframe=1&submit=Download+Data')
    csvfile = urlopen(url)
    skip_header(csvfile)
    df = pd.read_csv(csvfile, usecols=['Date/Time', 'Temp (°C)'],
                     index_col='Date/Time', parse_dates=['Date/Time'])
    if not isdir('data/weather/'):
        mkdir('data/weather/')
    df.to_hdf(cache_file_path, key='w')
    return df.loc[f'{year}-{month}-{day} {hour}:00'][0]


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
        if file.readline() == b'\n':
            n_emptyLineMet += 1
