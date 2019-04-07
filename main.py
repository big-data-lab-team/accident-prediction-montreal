#!/usr/bin/env python3
from road_network import get_road_df
from preprocess import get_positive_samples, get_negative_samples
from utils import init_spark

spark = init_spark()


print('Testing negatives generation...')
#539 293 * 43 848 = 23 646 919 464
#sample size wanted = ~100 000 => 0.00001
#for 1 year : 539 293 * 8760 = 4 724 206 680
negative_samples = get_negative_samples(spark,
                                        replace_cache=False,
                                        road_limit=None,
                                        year_limit=None,
                                        year_ratio=None,
                                        sample_ratio=0.00001)
negative_samples.show()
print("NUMBER GENERATED : ", negative_samples.count())

"""
print('testing positives generation...')
positive_samples = get_positive_samples(spark, replace_cache=True, limit=100)

"""
