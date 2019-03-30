#!/usr/bin/env python3
from road_network import get_road_df
from preprocess import get_positive_samples, get_negative_samples
from utils import init_spark

spark = init_spark()

print('Testing negatives generation...')
negative_samples = get_negative_samples(spark,
                                        replace_cache=True,
                                        road_limit=None,
                                        year_limit=2017,
                                        year_ratio=0.5,
                                        sample_ratio=0.5)
negative_samples.show()

"""print('testing positives generation...')
positive_samples = get_positive_samples(spark, replace_cache=True)
positive_samples.show()"""
