from math import pi
from os.path import isdir
from shutil import rmtree
import datetime
from functools import reduce
from pyspark.sql import Window, DataFrame
from pyspark.sql.functions import row_number, col, rank, avg, split, to_date, \
                                  monotonically_increasing_id, when, isnan, \
                                  year, cos, sin, month, dayofmonth, lit, \
                                  dayofweek, isnull
from pyspark.sql.types import BooleanType
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from utils import raise_parquet_not_del_error
from accidents_montreal import get_accident_df
from road_network import distance_intermediate_formula,\
                         distance_measure,\
                         get_road_features_df,\
                         get_road_df
from weather import get_weather_df
from workdir import workdir


def preprocess_accidents(accidents_df):
    ''' Select/build columns of interest and format their names.
    '''
    return (accidents_df
            .select('ACCIDENT_ID', 'DT_ACCDN', 'LOC_LAT',
                    'LOC_LONG', 'HEURE_ACCDN')
            .withColumn('date', to_date(col('DT_ACCDN'), format='yyyy/MM/dd'))
            .withColumn("hour", split(col('HEURE_ACCDN'), ':')[0].cast("int"))
            .drop('DT_ACCDN', 'HEURE_ACCDN')
            .withColumn('loc_lat', col('LOC_LAT').astype('double'))
            .withColumn('loc_long', col('LOC_LONG').astype('double'))
            .drop('LOC_LAT', 'LOC_LONG')
            .withColumnRenamed('ACCIDENT_ID', 'accident_id')
            .dropna())


def match_accidents_with_roads(spark, road_df, accident_df, use_cache=True):
    cache_path = workdir + 'data/matches_accident-road.parquet'
    if isdir(cache_path) and use_cache:
        print('Reading accident-road matches from cache...')
        return spark.read.parquet(cache_path)

    nb_top_road_center_preselected = 5
    max_distance_accepted = 10  # in meters

    # Compute distance between accident and road centers to identify the
    # top nb_top_road_center_preselected closest roads
    road_centers = (road_df
                    .select(['street_id', 'center_long', 'center_lat'])
                    .drop_duplicates())

    acc_window = (Window.partitionBy("accident_id")
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
                                     rank().over(acc_window)
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
                                           .over(acc_window)
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
                 rank().over(acc_window)
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
                 row_number().over(acc_window).alias('distance_rank'))
         .filter(col('distance_rank') == 1)
         .drop('distance_rank', 'loc_lat', 'loc_long',
               'coord_lat', 'coord_long'))

    # Union accidents matched correctly with first method with the accidents
    # for which we used more street points
    final_match = (accidents_road_correct_match
                   .union(accidents_roads_first_match_with_additional_coords))

    # Make sure there is only one road per accident
    final_match = (final_match
                   .join(road_centers, 'street_id')
                   .join(accident_df.select('loc_lat',
                                            'loc_long',
                                            'accident_id'),
                         'accident_id')
                   .withColumn('distance_inter',
                               distance_intermediate_formula(
                                    'loc_lat',
                                    'loc_long',
                                    'center_lat',
                                    'center_long'))
                   .withColumn('distance_measure', distance_measure())
                   .withColumn('dist_rank', row_number().over(acc_window))
                   .filter(col('dist_rank') == 1)
                   .select('accident_id', 'street_id'))

    return final_match


def generate_dates_in_year_df(year, spark):
    ''' Generate all dates and all hours between in the given year
    '''
    date = datetime.datetime.strptime(f"01/01/{year}", "%d/%m/%Y")
    end = datetime.datetime.strptime(f"01/01/{year+1}", "%d/%m/%Y")
    dates = list()
    while(date != end):
        for i in range(24):
            dates.append((date.strftime("%Y-%m-%d"), i))
        date += datetime.timedelta(days=1)

    return spark.createDataFrame(dates, ['date', 'hour'])


def generate_dates_df(spark, years, year_ratio):
    if years is None:
        years = (2012, 2013, 2014, 2015, 2016, 2017)

    if years is not None and isinstance(years, int):
        df = generate_dates_in_year_df(years, spark)
    elif years is not None and isinstance(years, tuple):
        dfs = (generate_dates_in_year_df(y, spark) for y in years)
        df = reduce(DataFrame.union, dfs)
    else:
        raise ValueError("Type of year_limit not authorized.")

    if year_ratio is not None:
        df = df.sample(year_ratio)

    return df.distinct()


def get_negative_samples(spark, use_cache=True, save_to=None, road_limit=None,
                         year_limit=None, year_ratio=None, weather_df=None,
                         sample_ratio=None, accident_df=None):
    """
    Note to self: 539 293 road, 43 848 generated dates,
    nb dates for 1 year : 8760

    year_limit: int or tuple of int
    """
    cache_path = workdir + 'data/negative-samples.parquet'
    if isdir(cache_path) and use_cache and save_to is None:
        return spark.read.parquet(cache_path)
    if save_to is not None:
        cache_path = workdir + save_to
        if isdir(cache_path):
            raise ValueError(f"Directory {save_to} already exists")

    road_df = get_road_df(spark, use_cache)
    road_features_df = \
        get_road_features_df(spark, road_df=road_df, use_cache=use_cache)
    road_df = road_features_df.select('street_id')
    dates_df = generate_dates_df(spark, year_limit, year_ratio)

    if road_limit is not None:
        road_df = road_df.limit(road_limit)

    negative_samples = (dates_df.crossJoin(road_df))

    if sample_ratio is not None:
        negative_samples = negative_samples.sample(sample_ratio)

    negative_samples = \
        negative_samples.withColumn('sample_id',
                                    monotonically_increasing_id())
    accident_df = preprocess_accidents(accident_df or get_accident_df(spark))
    weather_df = weather_df or get_weather_df(spark, accident_df)
    negative_samples = negative_samples.join(road_features_df, 'street_id')
    negative_sample_weather = \
        get_weather_information(negative_samples, weather_df)
    negative_samples = \
        negative_samples.join(negative_sample_weather, 'sample_id')
    negative_samples = add_date_features(negative_samples)
    negative_samples = negative_samples.persist()

    if use_cache:
        negative_samples.write.parquet(cache_path)
    return negative_samples


def get_positive_samples(spark, road_df=None, weather_df=None,
                         year_limit=None, use_cache=True, limit=None):
    if isinstance(year_limit, int):
        year_limit = [year_limit]
    elif isinstance(year_limit, tuple):
        year_limit = list(year_limit)
    elif not ((year_limit is None) or isinstance(year_limit, list)):
        raise ValueError('Type of year_limit not authorized.')

    cache_path = workdir + 'data/positive-samples.parquet'
    if isdir(cache_path) and use_cache:
        return spark.read.parquet(cache_path)

    road_df = road_df or get_road_df(spark, use_cache)
    accident_df = get_accident_df(spark, use_cache)
    accident_df = preprocess_accidents(accident_df)

    if year_limit is not None:
        accident_df = accident_df.filter(year('date').isin(year_limit))
    if limit is not None:
        accident_df = accident_df.limit(limit)

    weather_df = weather_df or get_weather_df(spark, accident_df)
    road_features_df = \
        (get_road_features_df(spark, road_df=road_df, use_cache=use_cache)
         .drop('loc_lat', 'loc_long'))
    match_acc_road = match_accidents_with_roads(spark, road_df, accident_df)
    accident_df = accident_df.withColumnRenamed('accident_id', 'sample_id')
    accident_weather = get_weather_information(accident_df, weather_df)
    positive_samples = (accident_df
                        .join(accident_weather, 'sample_id')
                        .withColumnRenamed('sample_id', 'accident_id')
                        .join(match_acc_road, 'accident_id')
                        .join(road_features_df, 'street_id')
                        .withColumnRenamed('accident_id', 'sample_id'))

    positive_samples = add_date_features(positive_samples)
    positive_samples = positive_samples.persist()

    if use_cache:
        positive_samples.write.parquet(cache_path)
    return positive_samples


def get_weather_information(samples, weather_df):
    '''Add weather coloumn to samples dataframe. '''
    p = 1
    weather_cols = list(set(weather_df.columns)
                        - set(['station_id', 'hour', 'station_lat',
                               'station_long', 'date']))
    weighted_weather_cols = [
        when(isnan(col(c)), 0).otherwise(col('inv_dist_to_station')*col(c))
        .alias(c+'_weighted') for c in weather_cols
        ]
    coeffs_weather_cols = [
        when(isnan(col(c)), 0)
        .otherwise(col('inv_dist_to_station')).alias(c+'_coeff')
        for c in weather_cols
        ]
    weighted_weather_cols_name = [c+'_weighted' for c in weather_cols]
    coeffs_weather_cols_name = [c+'_coeff' for c in weather_cols]
    final_weather_cols = [
        (col(f'sum({c}_weighted)')/col(f'sum({c}_coeff)')).alias(c)
        for c in weather_cols
        ]

    return (samples
            .join(weather_df, on=['date', 'hour'])
            .withColumn('distance_inter',
                        distance_intermediate_formula(
                                        'loc_lat',
                                        'loc_long',
                                        'station_lat',
                                        'station_long'))
            .withColumn('dist_to_station',
                        distance_measure()*(6371 * 2 * 1000))
            .withColumn('inv_dist_to_station', 1/col('dist_to_station')**p)
            .select('sample_id',
                    *(weighted_weather_cols + coeffs_weather_cols))
            .groupBy('sample_id')
            .sum(*(weighted_weather_cols_name + coeffs_weather_cols_name))
            .select('sample_id', *final_weather_cols))


def add_cyclic_feature(df, column, col_name, period):
    period_scale = (2 * pi) / period
    return (df
            .withColumn(col_name+'_cos', cos(column * lit(period_scale)))
            .withColumn(col_name+'_sin', sin(column * lit(period_scale)))
            .drop(col_name))


def add_date_features(samples):
    samples = add_cyclic_feature(samples, month('date'), 'month', 12)
    samples = add_cyclic_feature(samples, dayofmonth('date'), 'day', 31)

    samples = samples.withColumn('dayofweek', dayofweek('date'))
    encoder = OneHotEncoder(inputCols=['dayofweek'],
                            outputCols=["dayofweek_onehot"])
    encoder_model = encoder.fit(samples)
    samples = encoder_model.transform(samples).drop('dayofweek')

    return samples


features_col = ['hour',
                'loc_long',
                'loc_lat',
                'street_level_indexed',
                'street_length',
                'street_type_indexed',
                'wind_dir',
                'rel_hum',
                'wind_spd',
                'dew_point_temp',
                'visibility',
                'stn_press',
                'wind_chill',
                'hmdx',
                'temp',
                'month_cos',
                'month_sin',
                'day_cos',
                'day_sin',
                'dayofweek_onehot']


def remove_positive_samples_from_negative_samples(neg_samples, pos_samples):
    pos_samples_to_remove = pos_samples.select('date', 'hour', 'street_id',
                                               lit(1).alias('exists'))
    neg_samples = (neg_samples
                   .join(pos_samples_to_remove,
                         ['date', 'hour', 'street_id'],
                         "left_outer")
                   .filter(isnull('exists'))
                   .drop('exists'))
    return neg_samples


def get_dataset_df(spark, pos_samples, neg_samples):
    neg_samples = remove_positive_samples_from_negative_samples(neg_samples,
                                                                pos_samples)
    pos_samples = pos_samples.withColumn('label', lit(1.0))
    neg_samples = neg_samples.withColumn('label', lit(0.0))

    pos_samples = pos_samples.select(*neg_samples.columns)
    df = pos_samples.union(neg_samples)
    street_level_indexer = StringIndexer(inputCol="street_level",
                                         outputCol="street_level_indexed",
                                         stringOrderType="alphabetAsc")
    street_type_indexer = StringIndexer(inputCol="street_type",
                                        outputCol="street_type_indexed",
                                        handleInvalid='keep')
    df = street_level_indexer.fit(df).transform(df).drop('street_level')
    df = street_type_indexer.fit(df).transform(df).drop('street_type')

    assembler = VectorAssembler(outputCol="features",
                                inputCols=features_col,
                                handleInvalid='keep'
                                )
    df = (assembler.transform(df)
          .select('sample_id',
                  'street_id',
                  'date', 'hour', 'features', 'label'))

    return df
