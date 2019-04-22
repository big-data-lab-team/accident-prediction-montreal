from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, \
                              ParamGridBuilder, \
                              CrossValidator
from pyspark.ml import Pipeline
from pyspark.sql import Window
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col, floor, sum
import pandas as pd
import numpy as np
from preprocess import features_col
from random_undersampler import RandomUnderSampler
from class_weighter import ClassWeighter


def random_forest_tuning(train_samples):
    rf = RandomForestClassifier(labelCol="label",
                                featuresCol="features",
                                cacheNodeIds=True)
    ru = RandomUnderSampler().setIndexCol('id')
    pipeline = Pipeline().setStages([ru, rf])
    paramGrid = \
        (ParamGridBuilder()
         .addGrid(rf.numTrees, [50, 75, 100])
         .addGrid(rf.featureSubsetStrategy, ['sqrt'])
         .addGrid(rf.impurity, ['gini', 'entropy'])
         .addGrid(rf.maxDepth, [5, 15, 30])
         .addGrid(rf.minInstancesPerNode, [1])
         .addGrid(rf.subsamplingRate, [1.0, 0.6, 0.4])
         .addGrid(ru.targetImbalanceRatio, [1.0, 1.5, 2.0])
         .build())
    pr_evaluator = \
        BinaryClassificationEvaluator(labelCol="label",
                                      rawPredictionCol="rawPrediction",
                                      metricName="areaUnderPR")
    tvs = TrainValidationSplit(estimator=pipeline,
                               estimatorParamMaps=paramGrid,
                               evaluator=pr_evaluator,
                               trainRatio=0.8,
                               collectSubModels=True)

    model = tvs.fit(train_samples)

    return model


def balanced_random_forest_tuning(train_samples):
    rf = RandomForestClassifier(labelCol="label",
                                featuresCol="features",
                                cacheNodeIds=True,
                                weightCol="weight")
    ru = RandomUnderSampler().setIndexCol('id')
    cw = ClassWeighter()
    pipeline = Pipeline().setStages([ru, cw, rf])
    paramGrid = \
        (ParamGridBuilder()
         .addGrid(rf.numTrees, [50, 75, 100])
         .addGrid(rf.featureSubsetStrategy, ['sqrt'])
         .addGrid(rf.impurity, ['gini', 'entropy'])
         .addGrid(rf.maxDepth, [5, 15, 30])
         .addGrid(rf.minInstancesPerNode, [1])
         .addGrid(rf.subsamplingRate, [1.0, 0.66, 0.4])
         .addGrid(cw.classWeight, [[1/36, 1.0], [1/9.0, 1.0]])
         .addGrid(ru.targetImbalanceRatio, [9.0, 36.0])
         .build())
    pr_evaluator = \
        BinaryClassificationEvaluator(labelCol="label",
                                      rawPredictionCol="rawPrediction",
                                      metricName="areaUnderPR")
    tvs = CrossValidator(estimator=pipeline,
                         estimatorParamMaps=paramGrid,
                         evaluator=pr_evaluator,
                         numFolds=4,
                         collectSubModels=True)

    model = tvs.fit(train_samples)

    return model


def compute_precision_recall(predictions, threshold):
    def prob_positive(v):
        try:
            return float(v[1])
        except ValueError:
            return None

    prob_positive_udf = udf(prob_positive, DoubleType())
    true_positive = (predictions
                     .select('label', 'prediction')
                     .filter((col('label') == 1.0)
                             & (prob_positive_udf('probability') > threshold))
                     .count())
    false_positive = (predictions
                      .select('label', 'prediction')
                      .filter((col('label') == 0.0)
                              & (prob_positive_udf('probability') > threshold))
                      .count())
    true_negative = (predictions
                     .select('label', 'prediction')
                     .filter((col('label') == 0.0)
                             & (prob_positive_udf('probability') < threshold))
                     .count())
    false_negative = (predictions
                      .select('label', 'prediction')
                      .filter((col('label') == 1.0)
                              & (prob_positive_udf('probability') < threshold))
                      .count())
    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:
        precision = None
    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:
        recall = None
    return (precision, recall)


def compute_precision_recall_graph_slow(predictions, n_points):
    def gen_row(threshold):
        result = compute_precision_recall(predictions, threshold)
        return (threshold, result[0], result[1])

    space = np.linspace(0, 1, n_points)
    graph = pd.DataFrame([gen_row(t) for t in space],
                         columns=['Threshold', 'Precision', 'Recall'])

    return graph


def compute_precision_recall_graph(predictions, n_points):
    inf_cumulative_window = \
        (Window
         .partitionBy('label')
         .orderBy('id_bucket')
         .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    sup_cumulative_window = \
        (Window
         .partitionBy('label')
         .orderBy('id_bucket')
         .rowsBetween(1, Window.unboundedFollowing))

    def prob_positive(v):
        try:
            return float(v[1])
        except ValueError:
            return None

    prob_positive = udf(prob_positive, DoubleType())

    return \
        (predictions
         .select('label',
                 floor(prob_positive('probability') * n_points)
                 .alias('id_bucket'))
         .groupBy('label', 'id_bucket').count()
         .withColumn('count_negatives',
                     sum('count').over(inf_cumulative_window))
         .withColumn('count_positives',
                     sum('count').over(sup_cumulative_window))
         .groupBy('id_bucket').pivot('label', [0, 1])
         .sum('count_negatives', 'count_positives')
         .select(((col('id_bucket') + 1) / n_points).alias('threshold'),
                 col('0_sum(count_negatives)').alias('true_negative'),
                 col('0_sum(count_positives)').alias('false_positive'),
                 col('1_sum(count_negatives)').alias('false_negative'),
                 col('1_sum(count_positives)').alias('true_positive'))
         .select(col('threshold').alias('Threshold'),
                 (col('true_positive')
                 / (col('true_positive') + col('false_positive')))
                 .alias('Precision'),
                 (col('true_positive')
                 / (col('true_positive') + col('false_negative')))
                 .alias('Recall'))
         .orderBy('Threshold')
         .toPandas())


def get_feature_importances(model):
    feature_importances = pd.DataFrame(model.featureImportances.toArray())
    dayofweek_features = [f'{features_col[-1]}_{i}' for i in range(1, 8)]
    feature_names = features_col[:-1] + dayofweek_features
    feature_importances.index = feature_names
    feature_importances.columns = ["Feature importances"]
    feature_importances.sort_values(['Feature importances'], ascending=False)
    return feature_importances
