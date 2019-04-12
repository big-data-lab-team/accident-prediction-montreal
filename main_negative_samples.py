#!/usr/bin/env python
from preprocess import get_negative_samples, get_positive_samples
from utils import init_spark
spark = init_spark()
neg_samples = get_negative_samples(spark,
                                   save_to='data/negative_sample_new.parquet',
                                   sample_ratio=1e-2)
print(neg_samples.count())
