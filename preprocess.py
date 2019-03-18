from accidents_montreal import fetch_accidents_montreal,\
                               extract_accidents_montreal_dataframe
from road_network import fetch_road_network, extract_road_segments_DF
from weather import get_weather
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import atan2, sqrt, row_number, cos, sin, radians,\
                                  col, rank, avg


def match_accidents_with_weather(accident_df):
    return (accidents_df.select('DT_ACCDN',
                                'LOC_LAT',
                                'LOC_LONG',
                                'HEURE_ACCDN')
            .withColumn("year",
                        extract_date_val(0)(accidents_df.DT_ACCDN))
            .withColumn("month",
                        extract_date_val(1)(accidents_df.DT_ACCDN))
            .withColumn("day",
                        extract_date_val(2)(accidents_df.DT_ACCDN))
            .withColumn("HEURE_ACCDN",
                        extract_hour(accidents_df.HEURE_ACCDN))
            .drop('DT_ACCDN')
            .replace('Non précisé', '00')
            .rdd
            .map(lambda row: get_weather_(row)))


def match_accidents_with_roads(road_df, accident_df):
    nb_top_road_center_preselected = 5
    max_distance_accepted = 10  # in meters

    # Source: https://www.movable-type.co.uk/scripts/latlong.html
    def distance_intermediate_formula(lat1, long1, lat2, long2):
        return (pow(sin(radians(col(lat1) - col(lat2))/2), 2)
                + (pow(sin(radians(col(long1) - col(long2))/2), 2)
                * cos(radians(col(lat1))) * cos(radians(col(lat2)))))
    distance_measure = atan2(sqrt(col('distance_inter')),
                             sqrt(1-col('distance_inter')))
    earth_diameter = 6371 * 2 * 1000  # in meter

    # Compute distance between accident and road centers to identify the
    # top nb_top_road_center_preselected closest roads
    road_centers = (road_df
                    .select(['street_id', 'center_long', 'center_lat'])
                    .drop_duplicates()
                    .persist())

    accident_window = (Window.partitionBy("ACCIDENT_ID")
                       .orderBy("distance_measure"))
    accidents_top_k_roads = (accident_df
                             .select('LOC_LAT', 'LOC_LONG', 'ACCIDENT_ID')
                             .crossJoin(road_centers)
                             .withColumn('distance_inter',
                                         distance_intermediate_formula(
                                                        'LOC_LAT',
                                                        'LOC_LONG',
                                                        'center_lat',
                                                        'center_long'))
                             .withColumn('distance_measure', distance_measure)
                             .select('ACCIDENT_ID', 'street_id',
                                     'distance_measure', 'LOC_LAT', 'LOC_LONG',
                                     rank().over(accident_window)
                                     .alias('distance_rank'))
                             .filter(col('distance_rank') <=
                                     nb_top_road_center_preselected)
                             .drop('distance_measure', 'distance_rank')
                             .persist())

    # For each accident identify road point closest
    accidents_roads_first_match = (accidents_top_k_roads
                                   .join(road_df, 'street_id')
                                   .withColumn('distance_inter',
                                               distance_intermediate_formula(
                                                              'LOC_LAT',
                                                              'LOC_LONG',
                                                              'coord_lat',
                                                              'coord_long'))
                                   .withColumn('distance_measure',
                                               distance_measure)
                                   .select('ACCIDENT_ID', 'LOC_LAT',
                                           'LOC_LONG', 'coord_lat',
                                           'coord_long', 'street_id',
                                           'street_name',
                                           row_number()
                                           .over(accident_window)
                                           .alias('distance_rank'),
                                           'distance_measure')
                                   .filter(col('distance_rank') == 1)
                                   .withColumn('distance',
                                               col('distance_measure')
                                               * earth_diameter)
                                   .drop('distance_rank', 'distance_measure',
                                         'LOC_LAT', 'LOC_LONG', 'coord_lat',
                                         'coord_long')
                                   .persist())

    # If the distance is lower than max_distance_accepted we keep the
    # accident/street matches
    accidents_road_correct_match = (accidents_roads_first_match
                                    .filter(col('distance')
                                            < max_distance_accepted)
                                    .select('ACCIDENT_ID', 'street_id'))

    # If not, we try to get a better match by adding intermediate points on
    # the preselected streets

    # For unsatisfying matches, retrieves the top
    # nb_top_road_center_preselected streets with their points
    accidents_close_streets_coords = (accidents_roads_first_match
                                      .filter(col('distance')
                                              >= max_distance_accepted)
                                      .select('ACCIDENT_ID')
                                      .join(accidents_top_k_roads,
                                            'ACCIDENT_ID')
                                      .join(road_df, 'street_id')
                                      .select('ACCIDENT_ID', 'street_id',
                                              'LOC_LAT', 'LOC_LONG',
                                              'coord_long', 'coord_lat')
                                      .persist())

    # Add the intermediate points
    street_rolling_window = (Window
                             .partitionBy('street_id')
                             .orderBy("coord_long")
                             .rowsBetween(0, +1))
    accidents_close_streets_with_additional_coords = \
        (accidents_close_streets_coords
         .select('ACCIDENT_ID', 'street_id', 'LOC_LAT', 'LOC_LONG',
                 avg('coord_long')
                 .over(street_rolling_window)
                 .alias('coord_long'),
                 avg('coord_lat')
                 .over(street_rolling_window)
                 .alias('coord_lat'))
         .union(accidents_close_streets_coords)
         .dropDuplicates())

    # Recompute distances between accident and new set of points
    # and use closest point to identify street
    accidents_roads_first_match_with_additional_coords = \
        (accidents_close_streets_with_additional_coords
         .withColumn('distance_inter', distance_intermediate_formula(
                                                      'LOC_LAT',
                                                      'LOC_LONG',
                                                      'coord_lat',
                                                      'coord_long'))
         .withColumn('distance_measure', distance_measure)
         .select('ACCIDENT_ID', 'street_id', 'LOC_LAT', 'LOC_LONG',
                 'coord_lat', 'coord_long',
                 row_number().over(accident_window).alias('distance_rank'))
         .filter(col('distance_rank') == 1)
         .drop('distance_rank', 'LOC_LAT', 'LOC_LONG',
               'coord_lat', 'coord_long'))

    # Union accidents matched correctly with first method with the accidents
    # for which we used more street points
    return (accidents_road_correct_match
            .union(accidents_roads_first_match_with_additional_coords))


def init_spark():
    return (SparkSession
            .builder
            .appName("Road accidents prediction")
            .getOrCreate())


# init spark
spark = init_spark()

# retrieve datasets
fetch_road_network()
fetch_accidents_montreal()
accidents_df = extract_accidents_montreal_dataframe(spark)
road_df = extract_road_segments_DF(spark)
