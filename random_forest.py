from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from preprocess import get_dataset_df
from utils import init_spark

spark = init_spark()

neg_samples = spark.read.parquet('data/negative-samples.parquet')
pos_samples = spark.read.parquet('data/positive-samples.parquet')
df = get_dataset_df(spark, pos_samples, neg_samples)

(train_set, test_set) = df.randomSplit([0.7, 0.3])

rf = RandomForestClassifier(labelCol="label",
                            featuresCol="features",
                            numTrees=10)
model = rf.fit(train_set)

predictions = model.transform(test_set)

binaryEvaluator = \
    BinaryClassificationEvaluator(labelCol="label",
                                  rawPredictionCol="rawPrediction",
                                  metricName="areaUnderROC")
area_under_ROC = binaryEvaluator.evaluate(predictions)
evaluator = MulticlassClassificationEvaluator(predictionCol='prediction',
                                              labelCol='label',
                                              metricName='f1')
f1_score = evaluator.evaluate(predictions)
print(f"Area Under ROC = {area_under_ROC}\nF1 score = {f1_score}")
