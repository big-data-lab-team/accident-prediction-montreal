#!/usr/bin/env python

from preprocess import preprocess_accidents, \
                       get_positive_samples, \
                       get_negative_samples, \
                       match_accidents_with_roads
from road_network import get_road_df, \
                         get_road_features_df
from accidents_montreal import get_accident_df
from utils import init_spark
import datetime
import pyspark.sql.functions as f


spark = init_spark()

road_df = get_road_df(spark, replace_cache=False)
accident_df = preprocess_accidents(get_accident_df(spark,False))

# At this point, we have not choice but to use a sample since we don't have access to a cluster yet. Let's take one year, the last one
# the accident dataset contains all accident in years [2012-2017]
# Since one Year is too much, let's take half 
last_day_of_2016 = datetime.datetime.fromisoformat('2016-12-31')
accident_df = accident_df.filter(f.col('date') > last_day_of_2016)
match_accident_road = match_accidents_with_roads(road_df, accident_df)
match_accident_road.write.parquet('data/matching_accident_road_last_6_months.parquet')
