from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf, col
import pandas as pd
import numpy as np


def random_forest_tuning(train_samples):
    rf = RandomForestClassifier(labelCol="label",
                                featuresCol="features")

    paramGrid = \
        (ParamGridBuilder()
         .addGrid(rf.numTrees, [10, 50, 100])
         .addGrid(rf.featureSubsetStrategy, ['onethird', 'sqrt', 'log2'])
         .addGrid(rf.impurity, ['entropy', 'gini'])
         .addGrid(rf.maxDepth, [5, 10, 30])
         .addGrid(rf.minInstancesPerNode, [1, 5, 10, 50, 100])
         .addGrid(rf.subsamplingRate, [1.0, 0.66, 0.5, 0.33])
         .build())
    pr_evaluator = \
        BinaryClassificationEvaluator(labelCol="label",
                                      rawPredictionCol="rawPrediction",
                                      metricName="areaUnderPR")
    tvs = TrainValidationSplit(estimator=rf,
                               estimatorParamMaps=paramGrid,
                               evaluator=pr_evaluator,
                               trainRatio=0.7)

    model = tvs.fit(train_samples)

    return model


def evaluate_model(predictions):
    PR_evaluator = \
        BinaryClassificationEvaluator(labelCol="label",
                                      rawPredictionCol="rawPrediction",
                                      metricName="areaUnderPR")
    area_under_ROC = PR_evaluator.evaluate(predictions)
    f1_evaluator = \
        MulticlassClassificationEvaluator(predictionCol='prediction',
                                          labelCol='label',
                                          metricName='f1')
    f1_score = f1_evaluator.evaluate(predictions)
    print(f"Area Under ROC = {area_under_ROC}\nF1 score = {f1_score}")

    return (area_under_ROC, f1_score)


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


def compute_precision_recall_graph(predictions, n_points):
    def gen_row(threshold):
        result = compute_precision_recall(predictions, threshold)
        return (threshold, result[0], result[1])

    space = np.linspace(0, 1, n_points)
    graph = pd.DataFrame([gen_row(t) for t in space],
                         columns=['Threshold', 'Precision', 'Recall'])

    return graph
