import time
import sys
from utils import init_spark
from preprocess import get_positive_samples, get_negative_samples


def test_get_positive_samples():
    spark = init_spark()
    positive_samples = get_positive_samples(spark, road_df=None,
                                            use_cache=False, limit=10)
    positive_samples.show()
    assert positive_samples.count() > 0
    return


def get_negative_samples_(spark, params):
    return get_negative_samples(spark,
                                use_cache=params['use_cache'],
                                road_limit=params['road_limit'],
                                year_limit=params['year_limit'],
                                year_ratio=params['year_ratio'],
                                sample_ratio=params['sample_ratio'])


def test_get_negative_samples():
    spark = init_spark()
    params = {'use_cache': False,
              'road_limit': 20,
              'year_limit': 2017,
              'year_ratio': 0.01,
              'sample_ratio': 0.1}

    nb_samples = 8760 * params['year_ratio'] * params['road_limit']  \
        * params['sample_ratio']

    print("generating", str(nb_samples), "samples...")
    t = time.time()
    negative_samples = get_negative_samples_(spark, params)
    t = time.time() - t
    print("total time: ", t)
    negative_samples.show()
    assert negative_samples.count() > 0

    params['year_limit'] = (2012, 2013)
    params['sample_ratio'] = params['sample_ratio'] / 10
    nb_samples = 8760 * 2 * params['year_ratio'] * params['road_limit'] \
        * params['sample_ratio']

    print("generating", str(nb_samples), "samples...")
    t = time.time()
    negative_samples = get_negative_samples_(spark, params)
    t = time.time() - t
    print("total time: ", t)
    negative_samples.show()
    assert negative_samples.count() > 0
    return
