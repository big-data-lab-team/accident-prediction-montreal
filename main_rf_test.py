#!/usr/bin/env python3
from datetime import datetime
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import col
from preprocess import get_positive_samples, \
                       get_negative_samples, \
                       get_dataset_df, \
                       random_undersampling
from utils import init_spark
from workdir import workdir
from random_forest import compute_precision_recall_graph
from evaluate import evaluate_binary_classifier

spark = init_spark()

neg_samples = get_negative_samples(spark)
pos_samples = get_positive_samples(spark)

neg_samples = random_undersampling(pos_samples, neg_samples, target_ratio=2.0)

df = get_dataset_df(spark, pos_samples, neg_samples)
df_sample = (df
             .filter(col('date') > datetime.fromisoformat('2017-01-01'))
             .persist())

(train_set, test_set) = df_sample.randomSplit([0.7, 0.3])

rf = RandomForestClassifier(labelCol="label",
                            featuresCol="features",
                            cacheNodeIds=True,
                            maxDepth=30)
model = rf.fit(train_set)
predictions = model.transform(test_set)
evaluate_binary_classifier(predictions)
graph = compute_precision_recall_graph(predictions, 20)
graph.to_csv(workdir + 'data/precision_recall_graph_rf_'
             + 'randomundersampling2.csv')
