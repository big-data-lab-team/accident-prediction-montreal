from road_network import get_road_df
from preprocess import get_positive_samples, get_negative_samples
from preprocess import init_spark

spark = init_spark()
negative_samples = get_negative_samples(spark)

# road_df = get_road_df(spark)
# positive_samples = get_positive_samples(spark, road_df=road_df)
