from urllib.request import urlretrieve, urlopen
from urllib.error import URLError, HTTPError
from os.path import isfile
from zipfile import ZipFile
from io import BytesIO
import pyspark
import pandas as pd
import os
from pyspark.sql.types import StringType, BooleanType, IntegerType, \
        FloatType, StructType, StructField, DataType
from pyspark.sql.functions import monotonically_increasing_id
from utils import raise_parquet_not_del_error
from shutil import rmtree
from workdir import workdir


def get_accident_df(spark, use_cache=True):
    fetch_accidents_montreal()
    return read_accidents_montreal_df(spark, use_cache)


def fetch_accidents_montreal():
    if not os.path.isdir(workdir + 'data'):
        os.mkdir(workdir + 'data')
    if isfile(workdir + 'data/accidents-montreal.lock'):
        print('Skip fetching montreal accidents dataset: already downloaded')
        return
    url = 'http://donnees.ville.montreal.qc.ca/dataset/'\
          'cd722e22-376b-4b89-9bc2-7c7ab317ef6b/resource/'\
          '05deae93-d9fc-4acb-9779-e0942b5e962f/download/'\
          'accidents_2012_2017.zip'
    url_variable_desc = 'https://saaq.gouv.qc.ca/donnees-ouvertes/'\
                        'rapports-accident/rapports-accident-'\
                        'documentation.pdf'
    print('Fetching montreal accidents dataset...')
    try:
        (ZipFile(BytesIO(urlopen(url).read()))
         .extract('Accidents_2012_2017/Accidents_2012_2017.csv',
                  path=workdir + 'data'))
        urlretrieve(url_variable_desc,
                    workdir + 'data/accident-montreal-documentation.pdf')
        print('Fetching montreal accidents dataset: done')
        open(workdir + 'data/accidents-montreal.lock', 'w').close()
    except (URLError, HTTPError):
        print('Unable to find montreal accidents dataset.')


def read_accidents_montreal_df(spark, use_cache=True):
    cache = workdir + 'data/accidents_montreal.parquet'
    if (os.path.isdir(cache) or os.path.isfile(cache)) and use_cache:
        print('Skip extraction of accidents montreal dataframe:'
              ' already done, reading from file')
        return spark.read.parquet(cache)

    df = (spark
          .read
          .csv(workdir + 'data/Accidents_2012_2017/Accidents_2012_2017.csv',
               header=True)
          .repartition(200)
          .withColumn('ACCIDENT_ID', monotonically_increasing_id()))

    if use_cache:
        df.write.parquet(cache)
    return df
