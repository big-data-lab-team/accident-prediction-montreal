from accidents_montreal import extract_accidents_montreal_dataframe
from road_network import extract_road_segments_DF
import math
import pyspark
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Row

"""
def get_best_candidate(location, road, candidate_list):
for candidate in candidate_list:
road.filter('index' == candidate)[['index', 'coord_long', 'coord_lat']] \
.apply(lambda row: euclidian_dist((row[0],row[1]),location)) \
.min()"""

def euclidian_dist(center, location):
    if not len(center)==len(location)==2:
        raise ValueError('Bad argument(s)')
    return math.sqrt((center[0]-location[0])**2 + (center[1]-location[1])**2)

def get_nearest_neighbours(centers_rdd, location, k):
    return centers_rdd \
        .map(lambda row : Row(center_long=row[0], center_lat=row[1], id=row[2], dist=euclidian_dist((row[0], row[1]), location))) \
        .sortBy(lambda x: x.dist) \
        .take(k)

def get_most_probable_section(centers_rdd, center_neighbours, location):
    bests=list()
    for cn in center_neighbours :
        bests.append(centers_rdd \
            .filter(lambda c: c.center_long==cn.center_long and c.center_lat==cn.center_lat) \
            .map(lambda c: Row(center_long=c[0], center_lat=c[1], id=c[2], dist=euclidian_dist((c[0], c[1]), location))) \
            .union(sc.parallelize([cn])) \
            .sortBy(lambda x: x.dist) \
            .take(1))

    bests = list(map(lambda el:el[0], bests))
    return sc.parallelize(bests) \
            .sortBy(lambda x: x.dist) \
            .take(1)

sc = pyspark.SparkContext("local", "First App")
spark = pyspark.sql.SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
sqlContext = pyspark.sql.SQLContext(sc)

accidents_df=extract_accidents_montreal_dataframe(sqlContext)
road_df=extract_road_segments_DF(sc, sqlContext)
centers = road_df.select("*") \
    .withColumn("id", monotonically_increasing_id()) \
    .select(['center_long', 'center_lat', 'id']) \
    .drop_duplicates(['center_long','center_lat'])

location = (-73.861616, 45.45505)
k = 10
centers_rdd=centers.rdd
center_neighbours = get_nearest_neighbours(centers_rdd, location, k)
val = get_most_probable_section(centers_rdd, center_neighbours, location)
