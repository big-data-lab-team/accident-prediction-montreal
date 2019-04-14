from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def evaluate_binary_classifier(predictions):
    PR_evaluator = \
        BinaryClassificationEvaluator(labelCol="label",
                                      rawPredictionCol="rawPrediction",
                                      metricName="areaUnderPR")
    area_under_PR = PR_evaluator.evaluate(predictions)
    f1_evaluator = \
        MulticlassClassificationEvaluator(predictionCol='prediction',
                                          labelCol='label',
                                          metricName='f1')
    f1_score = f1_evaluator.evaluate(predictions)
    print(f"Area Under PR = {area_under_PR}\nF1 score = {f1_score}")

    return (area_under_PR, f1_score)
