from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
from os.path import isfile
from zipfile import ZipFile
from io import BytesIO
import pyspark
import pandas as pd
import os
from pyspark.sql.types import StringType, BooleanType, IntegerType, FloatType, StructType, StructField, DataType
from pyspark.sql.functions import monotonically_increasing_id

def fetch_accidents_montreal():
    if not os.path.isdir('data'):
        os.mkdir('data')
    if isfile('data/accidents-montreal.lock'):
        print('Skip fetching montreal accidents dataset: already downloaded')
        return
    url = 'http://donnees.ville.montreal.qc.ca/dataset/cd722e22-376b-4b89-9bc2-7c7ab317ef6b/resource/05deae93-d9fc-4acb-9779-e0942b5e962f/download/accidents_2012_2017.zip'
    url_variable_desc = 'https://saaq.gouv.qc.ca/donnees-ouvertes/rapports-accident/rapports-accident-documentation.pdf'
    print('Fetching montreal accidents dataset...')
    try:
        urlretrieve(url, 'data/accidents-montreal.zip')
        urlretrieve(url_variable_desc, 'data/accident-montreal-documentation.pdf')
        print('Fetching montreal accidents dataset: done')
        open('data/accidents-montreal.lock', 'w').close()
    except (URLError, HTTPError):
        print('Unable to find montreal accidents dataset.')

def extract_accidents_montreal_dataframe(spark):
    if os.path.isdir('data/accidents-montreal.parquet'):
        print('Skip extraction of accidents montreal dataframe: already done, reading from file')
        try:
            return spark.read.parquet('data/accidents-montreal.parquet')
        except:
            pass

    # We read directly from ZIP to avoid disk IO
    file = BytesIO(ZipFile('data/accidents-montreal.zip', 'r').read('Accidents_2012_2017/Accidents_2012_2017.csv'))
    pddf=pd.read_csv(file)
    
    # Create Spark schema, necessary to convert Pandas DF to Spark DF since dataframe contains different data types
    cols=pddf.columns.tolist()
    types=pddf.dtypes \
        .replace('object',StringType()) \
        .replace('int64',IntegerType()) \
        .replace('float64',FloatType()) \
        .replace('string',StringType()) \
        .tolist()
    fields = list(map(lambda u: StructField(u[0], u[1], True), zip(cols,types)))
    sch = StructType(fields)
    df = (spark.createDataFrame(data=pddf, schema=sch).repartition(200)
            .withColumn('ACCIDENT_ID', monotonically_increasing_id()))
    df.write.parquet('data/accidents-montreal.parquet')
    return df
