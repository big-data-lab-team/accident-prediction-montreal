# credits for integration of xgboost in pyspark: https://towardsdatascience.com/pyspark-and-xgboost-integration-tested-on-the-kaggle-titanic-dataset-4e75a568bdb

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.types import *
from pyspark.sql.functions import col
import numpy as np
from pyspark.ml.feature import StringIndexer, VectorAssembler
import os
from utils import init_spark
from preprocess import get_positive_samples, \
                       get_negative_samples, \
                       get_dataset_df
os.environ['PYSPARK_SUBMIT_ARGS'] = '--jars data/xgboost4j-spark-0.72.jar,data/xgboost4j-0.72.jar pyspark-shell'
spark = init_spark()
spark.sparkContext.addPyFile("data/sparkxgb.zip")
from sparkxgb import XGBoostEstimator
sc = spark.sparkContext
sc.setLogLevel("OFF")

# load dataset
neg_samples = get_negative_samples(spark).sample(0.01)
pos_samples = get_positive_samples(spark).sample(0.01)

df = get_dataset_df(spark, pos_samples, neg_samples).na.fill(0)
feature_list = [e for e in list(df.columns) if e != "label"]
# create one vector to feed xgboost
vectorAssembler = VectorAssembler()\
  .setInputCols(feature_list)\
  .setOutputCol("features")

# create xgboost estimator
xgboost = XGBoostEstimator(
    featuresCol="features",
    labelCol="label",
    predictionCol="prediction"
)
pipeline = Pipeline().setStages([xgboost])
trainDF, testDF = df.randomSplit([0.7, 0.3], seed=0)
model = pipeline.fit(trainDF)
