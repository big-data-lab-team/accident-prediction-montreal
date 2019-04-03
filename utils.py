from os.path import isdir
from pyspark.sql import SparkSession
from pyspark import SparkConf
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


def raise_parquet_not_del_error(cache):
    ''' Raise an error if cache parquet has not been deleted.
    '''
    if isdir(cache):
        print('Failed to remove parquet directory/file')
        raise Exception('Failed to remove parquet directory/file')
    return


def init_spark():

    conf = SparkConf() \
            .set('spark.executor.memory', '4g') \
            .set('spark.serializer',
                 'org.apache.spark.serializer.KryoSerializer') \
            .set('spark.rdd.compress', 'True') \
            .set('spark.eventLog.enabled', 'True')

    return (SparkSession
            .builder
            .getOrCreate())


def get_with_retry(
    url,
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session.get(url)
