from pyspark.ml.param.shared import Param, Params, HasSeed, HasLabelCol
from pyspark.ml import Transformer
from pyspark.sql.functions import col


class HasTargetImbalanceRatio(Params):
    targetImbalanceRatio = Param(Params._dummy(),
                                 "targetImbalanceRatio",
                                 "Target imbalance ratio after transformation "
                                 + "defined as the number of negative samples "
                                 + "divided by the number of positive samples")

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


class RandomUnderSampler(Transformer, HasTargetImbalanceRatio, HasSeed,
                         HasLabelCol):
    def _transform(self, dataset):
        neg_samples = dataset.filter(col(self.getLabelCol()) == 0.0)
        pos_samples = dataset.filter(col(self.getLabelCol()) == 1.0)
        current_ratio = neg_samples.count()/pos_samples.count()
        sampling = self.getTargetImbalanceRatio()/current_ratio
        neg_samples = neg_samples.sample(sampling, seed=self.getSeed())
        return neg_samples.union(pos_samples)
