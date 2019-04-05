from road_network import get_road_df
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
from utils import init_spark
from preprocess import preprocess_accidents, \
                    get_positive_samples, get_negative_samples
import time


def test_get_positive_samples():
    spark = init_spark()
    positive_samples = get_positive_samples(spark, road_df=None,
                                            replace_cache=True, limit=10)
    positive_samples.show()
    assert positive_samples.count() > 0
    return


def get_negative_samples_(spark, params):
    return get_negative_samples(spark,
                                replace_cache=params['replace_cache'],
                                road_limit=params['road_limit'],
                                year_limit=params['year_limit'],
                                year_ratio=params['year_ratio'],
                                sample_ratio=params['sample_ratio'])


def test_get_negative_samples():
    spark = init_spark()
    params = {'replace_cache': True,
              'road_limit': 20,
              'year_limit': 2017,
              'year_ratio': 0.01,
              'sample_ratio': 0.1}

    nb_samples = 8760 * params['year_ratio'] * params['road_limit']  \
        * params['sample_ratio']

    print("generating", str(nb_samples), "samples...")
    t = time.time()
    negative_samples = get_negative_samples_(spark, params)
    t = time.time() - t
    print("total time: ", t)
    negative_samples.show()
    assert negative_samples.count() > 0

    params['year_limit'] = (2012, 2013)
    params['sample_ratio'] = params['sample_ratio'] / 10
    nb_samples = 8760 * 2 * params['year_ratio'] * params['road_limit'] \
        * params['sample_ratio']

    print("generating", str(nb_samples), "samples...")
    t = time.time()
    negative_samples = get_negative_samples_(spark, params)
    t = time.time() - t
    print("total time: ", t)
    negative_samples.show()
    assert negative_samples.count() > 0
    return
