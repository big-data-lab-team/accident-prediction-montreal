#!/usr/bin/env python
from pyspark.sql.functions import udf, min, max, col
from pyspark.sql.types import FloatType
from pyspark.ml.feature import VectorAssembler
from preprocess import get_negative_samples, get_positive_samples
from utils import init_spark
from preprocess import get_dataset_df
from export_results import *


result_dir = create_result_dir('base')
spark = init_spark()
neg_samples = get_negative_samples(spark).sample(0.5)
pos_samples = get_positive_samples(spark)

imbalance_ratio = (neg_samples.count()/pos_samples.count())

train_set, test_set = get_dataset_df(spark, pos_samples, neg_samples)
train_set, test_set = train_set.persist(), test_set.persist()

get_accidents_count=udf(lambda v: float(v[7]), FloatType())


def fit(train_set):
    accidents_count = train_set.select(get_accidents_count('features').alias('accidents_count'), 'label')
    accidents_count_to_proba = []
    for i in range(377):
        accidents_count_higher = accidents_count.filter(col('accidents_count') >= i)
        proba = (accidents_count_higher.filter(col('label') == 1.0).count()
                / accidents_count_higher.count())
        accidents_count_to_proba.append(proba)
    model_df = spark.createDataFrame(zip(range(377), accidents_count_to_proba), ['accidents_count', 'pos_probability'])
    va = VectorAssembler(outputCol='probability', inputCols=['neg_probability', 'pos_probability'])
    model_df = (va.transform(model_df
                             .withColumn('neg_probability', 1 - col('pos_probability')))
                .drop('neg_probability', 'pos_probability'))
    return model_df


def transform(set, model):
    return(test_set
           .join(model, get_accidents_count('features') == model.accidents_count)
           .drop('accidents_count')
           .withColumn('rawPrediction', col('probability')))

model = fit(train_set).persist()

predictions = transform(test_set, model).persist()
train_predictions = transform(train_set, model).persist()

write_results(predictions, train_predictions, result_dir)


