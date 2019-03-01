from bs4 import BeautifulSoup
import pandas as pd

xml_file_path='data_tmp/doc.kml'
output_csv_file_path = 'out.csv'

def get_kml_content(soup):
    ''' Function to extract kml file content and store relevant information into a pandas dataframe.
    Args:
        soup: File content extracted using beautiful soup
    '''
    #create dataframe
    cols=['street_name', 'street_type', 'center_long', 'center_lat', 'center_dummy', 'coord_long', 'coord_lat', 'coord_dummy']
    df = pd.DataFrame(columns=cols)

    #extract information
    i=0
    folders = soup.find_all('Folder')
    for folder in folders:
        street_type = folder.find('name').text
        placemarks = folder.find_all('Placemark')

        for placemark in placemarks:
            street_name = placemark.find('name').text
            center = placemark.MultiGeometry.Point.coordinates.text
            coordinates_list = placemark.MultiGeometry.LineString.coordinates.text.split(' ')

            coords_list = list()
            for coords in coordinates_list:
                if len(coords) > 1:
                    df.loc[i]=[street_name, street_type] + center.split(',') + coords.split(',')
                    i+=1
    return df

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
        df = get_kml_content(soup)
    except:
        print('[Error] An error occured while extracting the content of the input file into a dataframe.')
        raise
    return df

with open(xml_file_path) as xml_file:
    df = kml_extract_dataframe(xml_file)
