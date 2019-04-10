#!/usr/bin/env python3
from road_network import get_road_df
from preprocess import get_positive_samples, get_negative_samples
from utils import init_spark

spark = init_spark()

print('testing positives generation...')
positive_samples = get_positive_samples(spark, limit=400)
