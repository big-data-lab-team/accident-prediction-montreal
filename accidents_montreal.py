from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
from os.path import isfile
from zipfile import ZipFile
from io import BytesIO
import dask.dataframe as dd
import pandas as pd

def fetch_accidents_montreal():
    if isfile('data/accidents-montreals.zip.lock'):
        print('Skip fetching montreal accidents dataset: already downloaded')
        return
    url = 'http://donnees.ville.montreal.qc.ca/dataset/cd722e22-376b-4b89-9bc2-7c7ab317ef6b/resource/05deae93-d9fc-4acb-9779-e0942b5e962f/download/accidents_2012_2017.zip'
    print('Fetching montreal accidents dataset...')
    try:
        urlretrieve(url, 'data/accidents-montreal.zip')
        print('Fetching montreal accidents dataset: done')
        open('data/accidents-montreals.zip.lock', 'w').close()
    except (URLError, HTTPError):
        print('Unable to reach montreal accidents dataset server')

def get_accidents_montreal():
    return BytesIO(ZipFile('data/accidents-montreal.zip', 'r').read('Accidents_2012_2017/Accidents_2012_2017.csv'))

def extract_accidents_montreal_dataframe():
    file = get_accidents_montreal()
    return dd.from_pandas(pd.read_csv(file), npartitions=8)
