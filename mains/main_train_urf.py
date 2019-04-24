#!/usr/bin/env python
from os import mkdir
from os.path import isdir
from preprocess import get_negative_samples, get_positive_samples
from utils import init_spark
from preprocess import get_dataset_df
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml import Pipeline
from random_undersampler import RandomUnderSampler
from workdir import workdir
from random_forest import compute_precision_recall_graph
from random_forest import get_feature_importances
from evaluate import evaluate_binary_classifier

# Prepare result directory
i = 1
while isdir(workdir + f'data/urf_{i}'):
    i += 1
mkdir(workdir + f'data/urf_{i}')

spark = init_spark()
neg_samples = get_negative_samples(spark).sample(1.0)
pos_samples = get_positive_samples(spark)

imbalance_ratio = (neg_samples.count()/pos_samples.count())

train_set, test_set = get_dataset_df(spark, pos_samples, neg_samples)
train_set, test_set = train_set.persist(), test_set.persist()

rf = RandomForestClassifier(labelCol="label",
                             featuresCol="features",
                             cacheNodeIds=True,
                             maxDepth=17,
                             impurity='entropy',
                             featureSubsetStrategy='sqrt',
                             minInstancesPerNode=10,
                             numTrees=100,
                             subsamplingRate=1.0,
                             maxMemoryInMB=768)
ru = (RandomUnderSampler()
      .setIndexCol('sample_id')
      .setTargetImbalanceRatio(1.0))
pipeline = Pipeline().setStages([ru, rf])
model = pipeline.fit(train_set)


# Write model hyper-parameters
def write_params(model, path):
  with open(path, 'w') as file:    
    for stage in model.stages:
          params = stage.extractParamMap()
          for k in params:
              file.write(f'{k.name}: {params[k]}\n')

write_params(model, workdir + f'data/urf_{i}/params')


predictions = model.transform(test_set).persist()
train_predictions = model.transform(train_set).persist()

# Write results
with open(workdir + f'data/urf_{i}/results', 'w') as file:
    test_res = evaluate_binary_classifier(predictions)
    train_res = evaluate_binary_classifier(train_predictions)
    file.write('Test set:\n')
    file.write(f"\tArea Under PR = {test_res[0]}\n")
    file.write(f"\tF1 score = {test_res[1]}\n")
    file.write(f"\tArea Under ROC = {test_res[2]}\n")
    file.write(f"\tAccuracy = {test_res[3]}\n")
    file.write('Train set:\n')
    file.write(f"\tArea Under PR = {train_res[0]}\n")
    file.write(f"\tF1 score = {train_res[1]}\n")
    file.write(f"\tArea Under ROC = {train_res[2]}\n")
    file.write(f"\tAccuracy = {train_res[3]}\n")

# Write feature importances
feature_importances = get_feature_importances(model.stages[1])
feature_importances.to_csv(workdir + f'data/urf_{i}/feature_importances.csv')

# Write precision recall curve
precision_recall = compute_precision_recall_graph(predictions, 20)
precision_recall.to_csv(workdir + f'data/urf_{i}/precision_recall.csv')

