#!/usr/bin/env python
from preprocess import get_negative_samples, get_positive_samples
from utils import init_spark
from workdir import workdir
spark = init_spark()
neg_samples = \
    get_negative_samples(spark,
                         save_to=workdir + 'data/negative-sample-new.parquet',
                         sample_ratio=1e-3)
print(neg_samples.count())
