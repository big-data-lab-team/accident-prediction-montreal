import sys
sys.path.insert(0, '.')  # noqa: E402
from accidents_montreal import get_accident_df
from weather import *
from pyspark.sql import SparkSession
import math
from weather import get_weather
import numpy as np


def test_get_weather():
    test_dict = {
                'Dew_Point_Temp': 0.358609581839746,
                'Hmdx': None,
                'Rel_Hum': 36.990385903958106,
                'Stn_Press': 101.84800521657381,
                'Temp': 14.956627338772165,
                'Visibility': 19.691895512391156,
                'Wind_Chill': None,
                'Wind_Dir': 8.40962398220665,
                'Wind_Spd': 13.166442191875918}

    spark = (SparkSession
             .builder
             .appName("Road accidents prediction")
             .getOrCreate())
    df = get_accident_df(spark)
    acc = df.filter('NO_SEQ_COLL == "SPVM _ 2015 _ 18203"').collect()[0]

    res_dict = get_weather(1, acc.LOC_LAT, acc.LOC_LONG, 2006, 5, 2, 0)

    assert res_dict == test_dict
