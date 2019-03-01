from bs4 import BeautifulSoup
import pandas as pd

xml_file_path='data_tmp/doc.kml'

#retrieve data from xml file
rows=list()
with open(xml_file_path) as xml_file:
    soup = BeautifulSoup(xml_file, "lxml-xml")
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
                coords_list += coords.split(',')

            row = [street_name,street_type] + center.split(',') + coords_list
            rows.append(row[:len(row)-1])

#create dataframe
nb_cols = max([len(row) for row in rows])
cols=['street_name', 'street_type']  \
    + ['center_'+str(i) for i in range(3)] \
    + ['coords_'+str(i)+'_'+str(j) for i in range(int((nb_cols-5)/3)) for j in range(3)]

df = pd.DataFrame(columns=cols)

for i, row in enumerate(rows):
    if len(row) < nb_cols:
        row += [''] * (nb_cols-len(row))
    df.loc[i]=row

df.to_csv('out.csv', sep=',')
