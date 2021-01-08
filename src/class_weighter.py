from pyspark.ml.param.shared import Param, Params, HasLabelCol, HasWeightCol
from pyspark.ml.param import TypeConverters
from pyspark.ml import Transformer
from pyspark.sql.functions import col, when, lit


class HasClassWeight(Params):
    classWeight = Param(
        Params._dummy(),
        "classWeight",
        "Array containing the weight to give to each class",
        typeConverter=TypeConverters.toListFloat,
    )

    def __init__(self):
        super(HasClassWeight, self).__init__()

    def getClassWeight(self):
        """
        Gets the value of seetargetImbalanceRatio or its default value.
        """
        return self.getOrDefault(self.classWeight)

    def setClassWeight(self, value):
        """
        Sets the value of :py:attr:`targetImbalanceRatio`.
        """
        return self._set(classWeight=value)


class ClassWeighter(Transformer, HasWeightCol, HasLabelCol, HasClassWeight):
    def __init__(self):
        super(ClassWeighter, self).__init__()
        self._setDefault(weightCol="weight")

    def _transform(self, dataset):
        class_weight = self.getClassWeight()
        return dataset.withColumn(
            self.getWeightCol(),
            when(col(self.getLabelCol()) == 0.0, lit(class_weight[0])).otherwise(
                lit(class_weight[1])
            ),
        )
