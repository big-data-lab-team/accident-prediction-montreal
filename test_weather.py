import pandas as pd
from os import mkdir
from os.path import isdir, isfile
from accidents_montreal import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Row
import math
from io import StringIO

fetch_accidents_montreal()
spark = (SparkSession
            .builder
            .appName("Road accidents prediction")
            .getOrCreate())
df=extract_accidents_montreal_dataframe(spark)
test=df.take(5)
test = list(map(lambda e: (e.LOC_LAT, e.LOC_LONG), test))
t= test[0]
lat, long = t
year, month, day = (2006,5,2)
hour = 0
get_weather(lat, long, year, month, day, hour)
