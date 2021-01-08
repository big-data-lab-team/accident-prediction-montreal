from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def evaluate_binary_classifier(predictions):
    PR_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    area_under_PR = PR_evaluator.evaluate(predictions)
    ROC_evaluator = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
    )
    area_under_ROC = ROC_evaluator.evaluate(predictions)

    print(f"Area Under PR = {area_under_PR}")
    print(f"Area Under ROC = {area_under_ROC}")

    return (area_under_PR, area_under_ROC)
