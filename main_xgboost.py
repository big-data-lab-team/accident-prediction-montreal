import numpy as np
import shutil
import os

from evaluate import *
from random_forest import *
from utils import init_spark
from preprocess import get_positive_samples, \
                       get_negative_samples, \
                       get_dataset_df
from workdir import workdir


from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col


from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


def save_model():
    workdir = "./"
    path = workdir + "data/xgboost.model"
    if os.path.isdir(path):
        shutil.rmtree(path)
    model.save(path)
    return


def grid_search(spark):
    sample_ratio = 0.05
    neg_samples = get_negative_samples(spark).sample(sample_ratio).na.fill(0)
    pos_samples = get_positive_samples(spark).sample(sample_ratio).na.fill(0)
    df = get_dataset_df(spark, pos_samples, neg_samples).na.fill(0)
    trainDF, testDF = df.randomSplit([0.8, 0.2], seed=0)

    xgboost = XGBoostEstimator(featuresCol="features",
                               labelCol="label",
                               predictionCol="prediction")

    pipeline = Pipeline().setStages([xgboost])
    model = pipeline.fit(trainDF)
    paramGrid = (ParamGridBuilder()
                 .addGrid(xgboost.max_depth, [x for x in range(3, 20, 6)])
                 .addGrid(xgboost.eta, [x for x in np.linspace(0.2, 0.6, 4)])
                 .addGrid(xgboost.scale_pos_weight,
                          [x for x in np.linspace(0.03, 1.0, 3)])
                 .build())
    evaluator = BinaryClassificationEvaluator(labelCol="label",
                                              rawPredictionCol="probabilities",
                                              metricName="areaUnderPR")
    cv = (CrossValidator()
          .setEstimator(pipeline)
          .setEvaluator(evaluator)
          .setEstimatorParamMaps(paramGrid)
          .setNumFolds(3))

    cvModel = cv.fit(trainDF)

    bestModel = (cvModel.bestModel
                        .asInstanceOf[PipelineModel]
                        .stages(2)
                        .asInstanceOf[XGBoostClassificationModel])

    with open(workdir + 'data/xgboost_tuning_results_1.txt', 'w') as file:
        for model, result in zip(model.subModels[0], model.avgMetrics):
            file.write('==================================\n')
            for stage in model.stages:
                params = stage.extractParamMap()
                for k in params:
                    file.write(f'{k.name}: {params[k]}\n')
            file.write(f"Area under PR: {result}\n")

    prediction = bestModel.transform(testDF)
    prediction = prediction.withColumn("rawPrediction",
                                       prediction['probabilities'])
    area_under_PR, f1_score = evaluate_binary_classifier(prediction)

    with open(workdir + 'data/xgboost_tuning_perf_1.txt', 'w') as file:
        file.write(f"Area Under PR = {area_under_PR}\nF1 score = {f1_score}")

    return


spark = init_spark()
spark.sparkContext.addPyFile(workdir + "data/sparkxgb.zip")
from sparkxgb import XGBoostEstimator  # noqa
grid_search(spark)
