from os.path import isdir
from pyspark.sql import SparkSession

def raise_parquet_not_del_error(cache):
    ''' Raise an error if cache parquet has not been deleted.
    '''
    if isdir(cache):
        print('Failed to remove parquet directory/file')
        raise Exception('Failed to remove parquet directory/file')
    return


def init_spark():
    return (SparkSession
            .builder
            .appName("Road accidents prediction")
            .getOrCreate())
