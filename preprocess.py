from accidents_montreal import fetch_accidents_montreal,\
                               extract_accidents_montreal_df,\
                               get_accident_df
from road_network import distance_intermediate_formula,\
                         distance_measure,\
                         get_road_features_df,\
                         get_road_df
from weather import add_weather_columns, extract_year_month_day
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import row_number, col, rank, avg, split, to_date, \
                                  rand, monotonically_increasing_id
from os.path import isdir
from shutil import rmtree
import datetime


def preprocess_accidents(accidents_df):
    ''' Select/build columns of interest and format their names.
    '''
    return (accidents_df
            .select('ACCIDENT_ID', 'DT_ACCDN', 'LOC_LAT',
                    'LOC_LONG', 'HEURE_ACCDN')
            .withColumn('date', to_date(col('DT_ACCDN'), format='yyyy/MM/dd'))
            .withColumn("hour", split(col('HEURE_ACCDN'), ':')[0].cast("int"))
            .drop('DT_ACCDN', 'HEURE_ACCDN')
            .withColumnRenamed('LOC_LAT', 'loc_lat')
            .withColumnRenamed('LOC_LONG', 'loc_long')
            .withColumnRenamed('ACCIDENT_ID', 'accident_id')
            .dropna())


def match_accidents_with_roads(road_df, accident_df):
    nb_top_road_center_preselected = 5
    max_distance_accepted = 10  # in meters

    # Compute distance between accident and road centers to identify the
    # top nb_top_road_center_preselected closest roads
    road_centers = (road_df
                    .select(['street_id', 'center_long', 'center_lat'])
                    .drop_duplicates()
                    .persist())

    accident_window = (Window.partitionBy("accident_id")
                       .orderBy("distance_measure"))
    accidents_top_k_roads = (accident_df
                             .select('loc_lat', 'loc_long', 'accident_id')
                             .crossJoin(road_centers)
                             .withColumn('distance_inter',
                                         distance_intermediate_formula(
                                                        'loc_lat',
                                                        'loc_long',
                                                        'center_lat',
                                                        'center_long'))
                             .withColumn('distance_measure',
                                         distance_measure())
                             .select('accident_id', 'street_id',
                                     'distance_measure', 'loc_lat', 'loc_long',
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
                                                              'loc_lat',
                                                              'loc_long',
                                                              'coord_lat',
                                                              'coord_long'))
                                   .withColumn('distance_measure',
                                               distance_measure())
                                   .select('accident_id', 'loc_lat',
                                           'loc_long', 'coord_lat',
                                           'coord_long', 'street_id',
                                           'street_name',
                                           row_number()
                                           .over(accident_window)
                                           .alias('distance_rank'),
                                           'distance_measure')
                                   .filter(col('distance_rank') == 1)
                                   .withColumn('distance',
                                               col('distance_measure')
                                               * (6371 * 2 * 1000))
                                   .drop('distance_rank', 'distance_measure',
                                         'loc_lat', 'loc_long', 'coord_lat',
                                         'coord_long')
                                   .persist())

    # If the distance is lower than max_distance_accepted we keep the
    # accident/street matches
    accidents_road_correct_match = (accidents_roads_first_match
                                    .filter(col('distance')
                                            < max_distance_accepted)
                                    .select('accident_id', 'street_id'))

    # If not, we try to get a better match by adding intermediate points on
    # the preselected streets

    # For unsatisfying matches, retrieves the top
    # nb_top_road_center_preselected streets with their points
    accidents_close_streets_coords = (accidents_roads_first_match
                                      .filter(col('distance')
                                              >= max_distance_accepted)
                                      .select('accident_id')
                                      .join(accidents_top_k_roads,
                                            'accident_id')
                                      .join(road_df, 'street_id')
                                      .select('accident_id', 'street_id',
                                              'loc_lat', 'loc_long',
                                              'coord_long', 'coord_lat')
                                      .persist())

    # Add the intermediate points
    street_rolling_window = (Window
                             .partitionBy('street_id')
                             .orderBy("coord_long")
                             .rowsBetween(0, +1))
    accidents_close_streets_with_additional_coords = \
        (accidents_close_streets_coords
         .select('accident_id', 'street_id', 'loc_lat', 'loc_long',
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
                                                      'loc_lat',
                                                      'loc_long',
                                                      'coord_lat',
                                                      'coord_long'))
         .withColumn('distance_measure', distance_measure())
         .select('accident_id', 'street_id', 'loc_lat', 'loc_long',
                 'coord_lat', 'coord_long',
                 row_number().over(accident_window).alias('distance_rank'))
         .filter(col('distance_rank') == 1)
         .drop('distance_rank', 'loc_lat', 'loc_long',
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


def generate_dates_df(start, end, spark):
    ''' Generate all dates and all hours between datetime start and
    datetime end.
    '''
    date = datetime.datetime.strptime(start, "%d/%m/%Y")
    end = datetime.datetime.strptime(end, "%d/%m/%Y")
    dates = list()
    while(date != end):
        date += datetime.timedelta(days=1)
        for i in range(24):
            dates.append((date.strftime("%d/%m/%Y"), i))
    return spark.createDataFrame(dates, ['date', 'hour']).persist()


def get_negative_samples(spark):
    cache_path = 'data/negative-samples.parquet'
    if isdir(cache_path):
        try:
            return spark.read.parquet(cache_path)
        except Exception:
            print('Failed reading from disk cache')
            rmtree(cache_path)

    dates_df = generate_dates_df("01/01/2012", "01/01/2017", spark)
    road_df = get_road_df(spark)
    road_features_df = get_road_features_df(spark, road_df=road_df)
    road_df = (road_df.select(['center_long', 'center_lat', 'street_id'])
                      .withColumnRenamed('center_lat', 'loc_lat')
                      .withColumnRenamed('center_long', 'loc_long')
                      .orderBy(rand())
                      .persist())

    negative_samples = (dates_df.rdd
                                .cartesian(road_df.rdd)
                                .map(lambda row: row[0] + row[1])
                                .toDF(['date', 'hour', 'loc_long',
                                       'loc_lat', 'street_id'])
                                .withColumn('accident_id',
                                            monotonically_increasing_id())
                                .persist())

    negative_samples = (add_weather_columns(spark, negative_samples)
                        .join(road_features_df, 'street_id'))

    negative_samples.write.parquet(cache_path)
    return negative_samples


def get_positive_samples(spark, road_df=None):
    cache_path = 'data/positive-samples.parquet'
    if isdir(cache_path):
        try:
            return spark.read.parquet(cache_path)
        except Exception:
            print('Failed reading from disk cache')
            rmtree(cache_path)

    if road_df is None:
        road_df = get_road_df(spark)

    accident_df = preprocess_accidents(get_accident_df(spark))
    road_features_df = get_road_features_df(spark, road_df=road_df)
    match_accident_road = match_accidents_with_roads(road_df, accident_df)
    accident_with_weather = add_weather_columns(spark, accident_df)
    positive_samples = extract_year_month_day(
            accident_with_weather
            .join(match_accident_road, 'accident_id')
            .join(road_features_df, 'street_id'))

    positive_samples.write.parquet(cache_path)
    return positive_samples
