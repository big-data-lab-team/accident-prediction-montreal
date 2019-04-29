from os import mkdir
from os.path import isdir
from pyspark.sql import SparkSession
from evaluate import evaluate_binary_classifier
from random_forest import compute_threshold_dependent_metrics
from workdir import workdir


def create_result_dir(algorithm):
    i = 1
    while isdir(workdir + f'data/{algorithm}_{i}'):
        i += 1
    mkdir(workdir + f'data/{algorithm}_{i}')
    return workdir + f'data/{algorithm}_{i}'


def write_params(model, n_neg_samples, result_dir):
    with open(result_dir + '/params', 'w') as file:
        file.write(f'count_negative_samples: {n_neg_samples}\n')
        def write_params(model):
            params = model.extractParamMap()
            for k in params:
                file.write(f'{k.name}: {params[k]}\n')
        if hasattr(model, 'stages'):
            for stage in model.stages:
                write_params(stage)
        else:
            write_params(model)



def write_results(test_predictions, train_predictions, result_dir):
    spark = SparkSession.builder.getOrCreate()
    with open(result_dir + '/results', 'w') as file:
        test_res = evaluate_binary_classifier(test_predictions)
        train_res = evaluate_binary_classifier(train_predictions)
        file.write('Test set:\n')
        file.write(f"\tArea Under PR = {test_res[0]}\n")
        file.write(f"\tArea Under ROC = {test_res[1]}\n")
        file.write('Train set:\n')
        file.write(f"\tArea Under PR = {train_res[0]}\n")
        file.write(f"\tArea Under ROC = {train_res[1]}\n")

    metrics = compute_threshold_dependent_metrics(spark, test_predictions, 20)
    metrics.set_index('Threshold').to_csv(result_dir + '/metrics.csv')

