#!/usr/bin/env python

from accidents_montreal import get_accident_df
from weather import get_weather_df
from utils import init_spark
from preprocess import preprocess_accidents

spark = init_spark()

accident_df = preprocess_accidents(get_accident_df(spark, False))
df = get_weather_df(spark, accident_df)
