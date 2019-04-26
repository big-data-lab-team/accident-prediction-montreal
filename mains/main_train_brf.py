#!/usr/bin/env python
from preprocess import get_negative_samples, get_positive_samples
from utils import init_spark
from preprocess import get_dataset_df
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml import Pipeline
from class_weighter import ClassWeighter
from random_forest import get_feature_importances
from export_results import *

result_dir = create_result_dir('brf')
spark = init_spark()
neg_samples = get_negative_samples(spark).sample(0.5)
pos_samples = get_positive_samples(spark)

imbalance_ratio = (neg_samples.count()/pos_samples.count())

train_set, test_set = get_dataset_df(spark, pos_samples, neg_samples)
train_set, test_set = train_set.persist(), test_set.persist()

brf = RandomForestClassifier(labelCol="label",
                             featuresCol="features",
                             cacheNodeIds=True,
                             maxDepth=25,
                             impurity='entropy',
                             featureSubsetStrategy='13',
                             weightCol='weight',
                             minInstancesPerNode=10,
                             numTrees=100,
                             subsamplingRate=1.0,
                             maxMemoryInMB=256)
cw = ClassWeighter().setClassWeight([1/imbalance_ratio, 1.0])
pipeline = Pipeline().setStages([cw, brf])
model = pipeline.fit(train_set)
predictions = model.transform(test_set).persist()
train_predictions = model.transform(train_set).persist()

write_params(model, result_dir)
write_results(predictions, train_predictions, result_dir)

# Write feature importances
feature_importances = get_feature_importances(model.stages[1])
feature_importances.to_csv(result_dir + '/feature_importances.csv')


