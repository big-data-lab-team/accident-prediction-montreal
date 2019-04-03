from os.path import isdir
from pyspark.sql import SparkSession
from pyspark import SparkConf


def raise_parquet_not_del_error(cache):
    ''' Raise an error if cache parquet has not been deleted.
    '''
    if isdir(cache):
        print('Failed to remove parquet directory/file')
        raise Exception('Failed to remove parquet directory/file')
    return


def init_spark():

    conf = SparkConf() \
            .set('spark.executor.memory', '5g') \
            .set('spark.serializer',
                 'org.apache.spark.serializer.KryoSerializer') \
            .set('spark.rdd.compress', 'True') \
            .set('spark.eventLog.enabled', 'True')

    return (SparkSession
            .builder
            .getOrCreate())
