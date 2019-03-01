from bs4 import BeautifulSoup
import pandas as pd

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

            coords_list = list()

            for coords in coordinates_list:
                coords.split(',')
                if len(coords) > 1:
                    rows.append({'street_name':street_name,
                        'street_type':street_type,
                        'center_long':center[0],
                        'center_lat':center[1],
                        'center_dummy':center[2],
                        'coord_long':coords[0],
                        'coord_lat':coords[1],
                        'coord_dummy':coords[2]})
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
