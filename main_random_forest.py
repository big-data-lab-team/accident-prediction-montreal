#!/usr/bin/env python3
from datetime import datetime
from pyspark.sql.functions import col
from random_forest import random_forest_tuning, \
                          evaluate_model, \
                          compute_precision_recall, \
                          compute_precision_recall_graph
from preprocess import get_positive_samples, \
                       get_negative_samples, \
                       get_dataset_df
from utils import init_spark
from workdir import workdir

spark = init_spark()

neg_samples = get_negative_samples(spark)
pos_samples = get_positive_samples(spark)
df = get_dataset_df(spark, pos_samples, neg_samples)
df_sample = df.filter(col('date') > datetime.fromisoformat('2017-01-01'))
(train_set, test_set) = df_sample.randomSplit([0.7, 0.3])
(train_set, test_set) = (train_set.persist(), test_set.persist())

model = random_forest_tuning(train_set)

model.save(workdir + 'data/random_forest.model')
paramsMap = model.bestModel.extractParamMap()
params = {k.name: paramsMap[k] for k in paramsMap}
print('Params:')
print(params)

predictions = model.transform(test_set)
evaluate_model(predictions)
precision, recall = compute_precision_recall(predictions, 0.01)
print(f"Precision: {precision}, Recall: {recall}")

graph = compute_precision_recall_graph(predictions, 20)
graph.to_parquet(workdir + 'data/precision_recall_graph_rf.parquet')
