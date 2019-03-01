from urllib.request import urlopen, urlretrieve
from urllib.parse import quote
from urllib.error import URLError, HTTPError
from os.path import isfile
from os import mkdir, listdir
from bs4 import BeautifulSoup
import re
from zipfile import ZipFile
from io import BytesIO
from dask import bag as db

def fetch_road_network():
    if isfile('data/road-network.lock'):
        print('Skip fetching road network: already downloaded')
        return
    print('Fetching road network...')
    try:
        url = 'http://ftp.maps.canada.ca/pub/nrcan_rncan/vector/geobase_nrn_rrn/qc/kml_en/'
        html_list = urlopen(url)
        bs = BeautifulSoup(html_list, 'lxml')
    except (URLError, HTTPError):
        print('Unable to reach dataset server')
        raise
    files = map(lambda tr: tr.a.text, bs.body.table.find_all('tr')[3:-1])
    files = filter(lambda f: re.match('.*_[4-7]_(5[5-9]|60).kmz', f), files)
    mkdir('data/road-network')
    for file in files:
        urlretrieve(f'{url}{quote(file)}', f'data/road-network/{file}')
    open('data/road-network.lock', 'wb').close()
    print('Fetching road network done')

def get_road_network():
    return db.from_sequence(listdir('data/road-network/'))         
        .map(lambda f: BytesIO(ZipFile(f'data/road-network/{f}', 'r').read('doc.kml')))
