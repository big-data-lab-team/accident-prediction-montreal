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


def test_get_positive_samples():
    spark = init_spark()
    positive_samples = get_positive_samples(spark, road_df=None,
                                            replace_cache=True, limit=10)
    positive_samples.show()
    assert positive_samples.count() > 0
    return


def test_get_negative_samples():
    spark = init_spark()
    negative_samples = get_negative_samples(spark,
                                            replace_cache=True, limit=10)
    negative_samples.show()
    assert negative_samples.count() > 0
    return
