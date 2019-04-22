#!/usr/bin/env python3
import accident_prediction_montreal
from datetime import datetime
from pyspark.sql.functions import col
from random_forest import random_forest_tuning, \
                          compute_precision_recall, \
                          compute_precision_recall_graph
from preprocess import get_positive_samples, \
                       get_negative_samples, \
                       get_dataset_df
from evaluate import evaluate_binary_classifier
from utils import init_spark
from workdir import workdir

spark = init_spark()

neg_samples = get_negative_samples(spark).sample(0.3)
pos_samples = get_positive_samples(spark).sample(0.3)
df = get_dataset_df(spark, pos_samples, neg_samples)
df_sample = df.filter(col('date') > datetime.fromisoformat('2017-01-01'))
(train_set, test_set) = df_sample.randomSplit([0.8, 0.2])

model = random_forest_tuning(train_set)

with open(workdir + 'data/random_forest_tuning_results_9.txt', 'w') as file:
    for model, result in zip(model.subModels, model.validationMetrics):
        file.write('==================================\n')
        for stage in model.stages:
            params = stage.extractParamMap()
            for k in params:
                file.write(f'{k.name}: {params[k]}\n')
        file.write(f"Area under PR: {result}\n")

predictions = model.transform(test_set)
area_under_PR, f1_score = evaluate_binary_classifier(predictions)
with open(workdir + 'data/random_forest_tuning_perf_9.txt', 'w') as file:
    file.write(f"Area Under PR = {area_under_PR}\nF1 score = {f1_score}")

graph = compute_precision_recall_graph(predictions, 20)
graph.to_csv(workdir + 'data/precision_recall_graph_rf_9.csv')
