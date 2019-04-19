#!/usr/bin/env python3
from datetime import datetime
from pyspark.sql.functions import col
from random_forest import balanced_random_forest_tuning, \
                          compute_precision_recall, \
                          compute_precision_recall_graph
from preprocess import get_positive_samples, \
                       get_negative_samples, \
                       get_dataset_df
from evaluate import evaluate_binary_classifier
from utils import init_spark
from workdir import workdir

spark = init_spark()

i=1
sampleFraction=0.01

neg_samples = get_negative_samples(spark).sample(sampleFraction)
pos_samples = get_positive_samples(spark).sample(sampleFraction)
df = get_dataset_df(spark, pos_samples, neg_samples)
(train_set, test_set) = df.randomSplit([0.8, 0.2])
(train_set, test_set) = (train_set.persist(), test_set.persist())

model = balanced_random_forest_tuning(train_set)

with open(workdir + f'data/brf_tuning_results_{i}.txt', 'w') as file:
    for model, result in zip(model.subModels[0], model.avgMetrics):
        file.write('==================================\n')
        for stage in model.stages:
            params = stage.extractParamMap()
            for k in params:
                file.write(f'{k.name}: {params[k]}\n')
        file.write(f"Area under PR: {result}\n")

predictions = model.transform(test_set)
area_under_PR, f1_score = evaluate_binary_classifier(predictions)
with open(workdir + f'data/brf_tuning_perf_{i}.txt', 'w') as file:
    file.write(f"Area Under PR = {area_under_PR}\nF1 score = {f1_score}")

graph = compute_precision_recall_graph(predictions, 20)
graph.to_csv(workdir + f'data/precision_recall_graph_brf_{i}.csv')
