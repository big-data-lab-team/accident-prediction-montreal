from accidents_montreal import extract_accidents_montreal_dataframe
from road_network import extract_road_network_dataframe
import math

def euclidian_dist(center, location):
    return math.sqrt((center[0]-center[1])**2 + (location[0]-location[1])**2)

accidents=extract_accidents_montreal_dataframe()
road=extract_road_network_dataframe()
road['index']= road.index.to_series()
centers = road.iloc[:,[2,3,6]].drop_duplicates(['center_long','center_lat']).persist()
location = (-73.861616,45.45505)
k=10

def get_nearest_neighbours(centers, location, k):
    centers['dist'] = centers.apply(lambda center : euclidian_dist(center, location), axis=1)
    centers = centers.set_index('dist', inplace=True) #sort
    return centers['index'].head(k)

def get_most_probable_section(accidents, centers, road, k):
    accidents['probable_centers'] = accidents[['LOC_LONG', 'LOC_LAT']] \
        .apply(lambda row: get_nearest_neighbours(centers, (row[0],row[1]), k), axis=1)
    accidents.apply(lambda row: get_best_candidate((row[0],row[1]), road, row[2]))

def get_best_candidate(location, road, candidate_list):
    for candidate in candidate_list:
        road.filter('index' == candidate)[['index', 'coord_long', 'coord_lat']] \
            .apply(lambda row: euclidian_dist((row[0],row[1]),location)) \
            .min()
