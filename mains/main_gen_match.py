#!/usr/bin/env python
import accident_prediction_montreal
import os
from functools import reduce
import datetime
from os.path import isdir
from pyspark.sql.functions import col
from pyspark.sql import Window
from pyspark.sql import DataFrame
from utils import init_spark
from workdir import workdir
from preprocess import preprocess_accidents, \
                       get_positive_samples, \
                       get_negative_samples, \
                       match_accidents_with_roads
from road_network import get_road_df
from accidents_montreal import get_accident_df


def generate_match_accident_road_of_one_month(year, month):
    filepath = workdir + f'data/match_accident-road_{year}-{month}.parquet'
    if isdir(filepath):  # Skip if already done
        return
    print(f'Generating {year}-{month}')
    spark = init_spark()
    road_df = get_road_df(spark, use_cache=True)
    accident_df = preprocess_accidents(get_accident_df(spark))

    start_day_str = f'{year}-{month:02}-01'
    if month == 12:
        end_year = year + 1
        month = 0
    else:
        end_year = year
    end_day_str = f'{end_year}-{(month + 1):02}-01'

    start_day = datetime.datetime.fromisoformat(start_day_str)
    end_day = datetime.datetime.fromisoformat(end_day_str)
    accident_df = (accident_df
                   .filter((col('date') >= start_day)
                           & (col('date') < end_day)))

    match_accident_road = \
        match_accidents_with_roads(spark, road_df, accident_df,
                                   use_cache=False)

    match_accident_road.write.parquet(filepath)
    spark.stop()  # Force garbage collection and empty temp dir


# Make sure to generate accidents DF
spark = init_spark()
get_accident_df(spark)
spark.stop()

for year in range(2012, 2018):
    for month in range(1, 13):
        generate_match_accident_road_of_one_month(year, month)

spark = init_spark()


def match_accident_road_samples_reader():
    for year in range(2012, 2018):
        for month in range(1, 13):
            filepath = workdir \
                + f'data/match_accident-road_{year}-{month}' \
                + '.parquet'
            yield spark.read.parquet(filepath)


full_matching_accident_road = reduce(DataFrame.union,
                                     match_accident_road_samples_reader())
filepath = workdir + 'data/match_accident-road.parquet'
full_matching_accident_road.write.parquet(filepath)
