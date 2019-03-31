#!/usr/bin/env python

from accidents_montreal import get_accident_df
from weather import get_useful_stations_id_df
from utils import init_spark
from preprocess import preprocess_accidents

spark = init_spark()

accident_df = preprocess_accidents(get_accident_df(spark, False))

(get_useful_stations_id_df(spark, accident_df)
    .write.parquet('data/weather_stations_id.parquet'))
