#!/usr/bin/env python

import os
from utils import init_spark
spark = init_spark()

def myfun(x):
    os.system("pip install pandas beautifulsoup4 lxml matplotlib requests")
    return x

sc = spark.sparkContext
log4jLogger = sc._jvm.org.apache.log4j
log = log4jLogger.LogManager.getLogger(__name__)

rdd = sc.parallelize([1,2,3,4]) ## assuming 4 worker nodes
rdd.map(lambda x: myfun(x)).collect() 

from preprocess import preprocess_accidents, \
                       get_positive_samples, \
                       get_negative_samples, \
                       match_accidents_with_roads
from road_network import get_road_df, \
                         get_road_features_df
from accidents_montreal import get_accident_df
import datetime
import pyspark.sql.functions as f


workdir = "/home/tguedon/projects/def-glatard/tguedon/accident-prediction-montreal"


log.warn("road df...")
road_df = get_road_df(spark, replace_cache=False)

log.warn("accident_df...")
accident_df = preprocess_accidents(get_accident_df(spark, False))

# At this point, we have not choice but to use a sample since we
# don't have access to a cluster yet. Let's take one year, the last one
# the accident dataset contains all accident in years [2012-2017]
# Since one Year is too much, let's take half
last_day_of_2016 = datetime.datetime.fromisoformat('2016-12-31')

log.warn("filtering...")
accident_df = accident_df.filter(f.col('date') > last_day_of_2016)

log.warn("matching...")
match_accident_road = match_accidents_with_roads(road_df, accident_df)

log.warn("writing...")
(match_accident_road
    .write.parquet(workdir + '/data/matching_accident_road_last_6_months.parquet'))
