#!/usr/bin/env python
from preprocess import get_positive_samples
from utils import init_spark
spark = init_spark()
pos_samples = get_positive_samples(spark)
print(pos_samples.count())
