from pyspark.ml.param.shared import Param, Params, HasSeed, HasLabelCol
from pyspark.ml import Transformer, Estimator
from pyspark.sql.functions import col, lit, isnull
from pyspark.ml.param import TypeConverters


class HasTargetImbalanceRatio(Params):
    targetImbalanceRatio = Param(
        Params._dummy(),
        "targetImbalanceRatio",
        "Target imbalance ratio after transformation "
        + "defined as the number of negative samples "
        + "divided by the number of positive samples",
    )

    def __init__(self):
        super(HasTargetImbalanceRatio, self).__init__()
        self._setDefault(targetImbalanceRatio=1.0)

    def getTargetImbalanceRatio(self):
        """
        Gets the value of seetargetImbalanceRatio or its default value.
        """
        return self.getOrDefault(self.targetImbalanceRatio)

    def setTargetImbalanceRatio(self, value):
        """
        Sets the value of :py:attr:`targetImbalanceRatio`.
        """
        return self._set(targetImbalanceRatio=value)


class HasIndexCol(Params):
    """
    Mixin for param indexCol: index column name.
    """

    indexCol = Param(
        Params._dummy(),
        "indexCol",
        "index column name.",
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super(HasIndexCol, self).__init__()
        self._setDefault(indexCol="index")

    def setIndexCol(self, value):
        """
        Sets the value of :py:attr:`indexCol`.
        """
        return self._set(indexCol=value)

    def getIndexCol(self):
        """
        Gets the value of indexCol or its default value.
        """
        return self.getOrDefault(self.indexCol)


class RandomUnderSampler(
    Estimator, HasTargetImbalanceRatio, HasSeed, HasLabelCol, HasIndexCol
):
    def _fit(self, dataset):
        neg_samples = dataset.filter(col(self.getLabelCol()) == 0.0)
        pos_samples = dataset.filter(col(self.getLabelCol()) == 1.0)
        current_ratio = neg_samples.count() / pos_samples.count()
        sampling = self.getTargetImbalanceRatio() / current_ratio
        if sampling > 1.0:
            # Nothing to do imbalance ratio already lower than target
            indexes_to_remove = None
        else:
            indexes_to_remove = neg_samples.select(self.getIndexCol()).sample(
                1 - sampling, seed=self.getSeed()
            )

        return RandomUnderSamplerModel(
            indexes_to_remove, self.getSeed()
        ).setTargetImbalanceRatio(self.getTargetImbalanceRatio())


class RandomUnderSamplerModel(Transformer, HasTargetImbalanceRatio):
    def __init__(self, indexesToRemove, seed):
        super(RandomUnderSamplerModel, self).__init__()
        self.seed = seed
        self.indexesToRemove = indexesToRemove

    def _transform(self, dataset):
        if self.indexesToRemove is None:
            return dataset
        index_col = self.indexesToRemove.columns[0]
        return (
            dataset.join(
                self.indexesToRemove.withColumn("exists", lit(1.0)),
                index_col,
                "left_outer",
            )
            .filter(isnull("exists"))
            .drop("exists")
        )
