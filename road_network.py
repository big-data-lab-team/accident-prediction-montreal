from urllib.request import urlopen, urlretrieve
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import os, re, pyspark
from bs4 import BeautifulSoup
from zipfile import ZipFile
from io import BytesIO
from bs4 import BeautifulSoup
import pandas as pd
from pyspark.sql.functions import col, abs, hash

def fetch_road_network():
    if not os.path.isdir('data'):
        os.mkdir('data')
    if os.path.isfile('data/road-network.lock'):
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
    if not os.path.isdir('data/road-network'):
        os.mkdir('data/road-network')
    for file in files:
        urlretrieve(f'{url}{quote(file)}', f'data/road-network/{file}')
    open('data/road-network.lock', 'wb').close()
    print('Fetching road network done')

def get_kml_content(soup):
    ''' Function to extract kml file content and store relevant information into a pandas dataframe.
    Args:
        soup: File content extracted using beautiful soup
    '''
    rows=list()
    folders = soup.find_all('Folder')
    for folder in folders:
        street_type = folder.find('name').text
        placemarks = folder.find_all('Placemark')

        for placemark in placemarks:
            street_name = placemark.find('name').text
            center = placemark.MultiGeometry.Point.coordinates.text.split(',')
            coordinates_list = placemark.MultiGeometry.LineString.coordinates.text.split(' ')

            for coord in coordinates_list:
                coords = coord.split(',')
                if len(coords) > 1:
                    rows.append([
                            street_name,
                            street_type,
                            float(center[0]),
                            float(center[1]),
                            float(coords[0]),
                            float(coords[1])])
            # Add center of the street as a point of the street
            rows.append([
                    street_name,
                    street_type,
                    float(center[0]),
                    float(center[1]),
                    float(center[0]),
                    float(center[1])])
    
    return rows

def kml_extract_RDD(xml_file):
    ''' Function to extract the content of a kml input file and to store it into a csv output file.
    Args:
        xml_file_path: input kml file (kml is an xml file)
    '''

    try:
        soup = BeautifulSoup(xml_file, "lxml-xml")
    except:
        print('Unable to open input file.')
        raise ValueError('Unable to open input file')

    try:
        rows = get_kml_content(soup)
    except:
        print('An error occured while extracting the content of the input file into a dataframe.')
        raise ValueError('An error occured while extracting the content of the input file into a dataframe.')
    return rows

def get_road_segments_RDD(spark):
    return spark.sparkContext.parallelize(os.listdir('data/road-network/')) \
        .map(lambda f: BytesIO(ZipFile(f'data/road-network/{f}', 'r').read('doc.kml')))

def extract_road_segments_DF(spark):
    if os.path.isdir('data/road-network.parquet'):
        print('Skip extraction of road network dataframe: already done, reading from file')
        try:
            return spark.read.parquet('data/road-network.parquet')
        except:
            pass

    print('Extracting road network dataframe...')
    cols=['street_name','street_type','center_long', 'center_lat', 'coord_long', 'coord_lat']
    road_seg_df = (get_road_segments_RDD(spark)
        .flatMap(kml_extract_RDD)
        .toDF(cols)
        .withColumn('street_id', abs(hash(col('center_long'), col('center_lat')))))

    road_seg_df.write.parquet('data/road-network.parquet')
    print('Extracting road network dataframe done')
    return road_seg_df
