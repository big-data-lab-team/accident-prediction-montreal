from bs4 import BeautifulSoup
import pandas as pd
from fetch_road_network import get_road_network
import dask.dataframe as df
from os.path import isdir

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
                    rows.append({
                            0: street_name,
                            1: street_type,
                            2: float(center[0]),
                            3: float(center[1]),
                            4: float(coords[0]),
                            5: float(coords[1])
                        })
    return rows

def kml_extract_dataframe(xml_file):
    ''' Function to extract the content of a kml input file and to store it into a csv output file.
    Args:
        xml_file_path: input kml file (kml is an xml file)
    '''

    try:
        soup = BeautifulSoup(xml_file, "lxml-xml")
    except:
        print('[Error] Unable to open input file {0}'.format(xml_file_path))
        raise

    try:
        rows = get_kml_content(soup)
    except:
        print('[Error] An error occured while extracting the content of the input file into a dataframe.')
        raise
    return rows

def extract_road_network_dataframe():
    if isdir('data/road-network.parquet'):
        print('Skip extraction of road network dataframe: already done, reading from file')
        return df.read_parquet('data/road-network.parquet')
    print('Extracting road network dataframe...')
    road_network_dataframe = (get_road_network()
            .map(kml_extract_dataframe)
            .flatten()
            .to_dataframe()
            .rename(columns={
                    0:'street_name',
                    1: 'street_type',
                    2: 'center_long',
                    3: 'center_lat',
                    4: 'coord_long',
                    5: 'coord_lat'
                })
            .persist())
    df.to_parquet(road_network_dataframe, 'data/road-network.parquet')
    print('Extracting road network dataframe done')
    return df
