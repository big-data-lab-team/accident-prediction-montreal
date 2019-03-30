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
                                  rand, monotonically_increasing_id, year, udf
from pyspark.sql.types import *
from os.path import isdir
from shutil import rmtree
import datetime
from utils import raise_parquet_not_del_error, init_spark
import pyspark


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
                    .drop_duplicates())

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
                                         'coord_lat', 'coord_long')
                                   .persist())

    # If the distance is lower than max_distance_accepted we keep the
    # accident/street matches
    accidents_road_correct_match = (accidents_roads_first_match
                                    .filter(col('distance')
                                            < max_distance_accepted)
                                    .select('accident_id', 'street_id'))

    # If not, we try to get a better match by adding intermediate points on
    # the preselected streets
    # For unsatisfying matches, recompute the k closests roads
    # Recomputing is probably faster than reading from disk
    # cache + joining on accident_ids
    accidents_close_streets_coords = \
        (accidents_roads_first_match
         .filter(col('distance') >= max_distance_accepted)
         .select('accident_id', 'loc_lat', 'loc_long')
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
         .join(
             road_df.select('street_id', 'coord_lat', 'coord_long'),
             'street_id'))

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
    accidents_close_streets_coords.unpersist()

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
            dates.append((date.strftime("%Y-%m-%d"), i))

    # .persist(pyspark.StorageLevel.MEMORY_AND_DISK_SR)
    return spark.createDataFrame(dates, ['date', 'hour'])


def extract_years(dates_df, year_limit, year_ratio):
    test_year_values = udf(lambda x: True if x in year_limit else False,
                           BooleanType())
    if isinstance(year_limit, tuple):
        return (dates_df.withColumn('year', year(col('date')))
                        .filter(test_year_values(col('year')))
                        .drop('year')
                        .sample(year_ratio))
    elif isinstance(year_limit, int):
        return (dates_df.withColumn('year', year(col('date')))
                        .filter(col('year') == year_limit)
                        .drop('year')
                        .sample(year_ratio))
    else:
        if year_limit is not None:
            print("Type of year_limit not authorized. Generating everything..")
        return dates_df.sample(year_ratio)


def get_negative_samples(spark, replace_cache=False,
                         road_limit=None, year_limit=None, year_ratio=None,
                         sample_ratio=None):
    """
    Note to self: 539 293 road, 43 848 generated dates,
    nb dates for 1 year : 8760

    year_limit: int or tuple of int
    """
    cache_path = 'data/negative-samples.parquet'
    if isdir(cache_path):
        try:
            if replace_cache:
                print('Removing cache...')
                rmtree(cache_path)
                raise_parquet_not_del_error(cache_path)
            else:
                print('Reading from cache...')
                return spark.read.parquet(cache_path)
        except Exception:
            print('Failed reading from disk cache')
            rmtree(cache_path)
            raise_parquet_not_del_error(cache_path)

    road_df = get_road_df(spark, replace_cache)
    road_features_df = get_road_features_df(spark, road_df=road_df,
                                            replace_cache=False)
    road_df = (road_df.select(['center_long', 'center_lat', 'street_id'])
                      .withColumnRenamed('center_lat', 'loc_lat')
                      .withColumnRenamed('center_long', 'loc_long'))

    dates_df = generate_dates_df("01/01/2012", "31/12/2017", spark)
    dates_df = extract_years(dates_df, year_limit, year_ratio)

    if road_limit is not None:
        road_df = road_df.limit(road_limit)

    negative_samples = (dates_df.crossJoin(road_df)
                                .withColumn('accident_id',
                                            monotonically_increasing_id()))

    negative_samples = (add_weather_columns(spark, negative_samples)
                        .join(road_features_df, 'street_id'))

    if sample_ratio is not None:
        negative_samples = negative_samples.sample(sample_ratio)

    negative_samples.write.parquet(cache_path)
    return negative_samples


def get_positive_samples(spark, road_df=None, replace_cache=False, limit=None):
    cache_path = 'data/positive-samples.parquet'
    if isdir(cache_path):
        try:
            if replace_cache:
                rmtree(cache_path)
                raise_parquet_not_del_error(cache_path)
            else:
                return spark.read.parquet(cache_path)
        except Exception:
            print('Failed reading from disk cache')
            rmtree(cache_path)
            raise_parquet_not_del_error(cache_path)

    if road_df is None:
        road_df = get_road_df(spark, replace_cache)

    if limit is None:
        accident_df = preprocess_accidents(get_accident_df(spark,
                                                           replace_cache))
    else:
        accident_df = preprocess_accidents(get_accident_df(spark,
                                                           replace_cache)) \
                                                           .limit(limit)

    road_features_df = get_road_features_df(spark, road_df=road_df,
                                            replace_cache=replace_cache)
    match_accident_road = match_accidents_with_roads(road_df, accident_df)
    accident_with_weather = add_weather_columns(spark, accident_df)
    positive_samples = extract_year_month_day(
            accident_with_weather
            .join(match_accident_road, 'accident_id')
            .join(road_features_df, 'street_id'))

    positive_samples.write.parquet(cache_path)
    return positive_samples
