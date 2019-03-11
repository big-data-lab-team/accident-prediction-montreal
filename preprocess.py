from accidents_montreal import extract_accidents_montreal_dataframe
from road_network import extract_road_segments_DF
import math
import pyspark
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Row

def euclidian_dist(center, location):
    ''' Euclidian distance between two 2D points.
    '''
    if not len(center)==len(location)==2:
        raise ValueError('Bad argument(s)')
    return math.sqrt((center[0]-location[0])**2 + (center[1]-location[1])**2)

def get_nearest_neighbours(centers_rdd, location, k):
    ''' Get the k nearest centers from a given location (longitude,latitude)
    '''
    return centers_rdd \
        .map(lambda row : Row(center_long=row[0], center_lat=row[1], id=row[2], dist=euclidian_dist((row[0], row[1]), location))) \
        .sortBy(lambda x: x.dist) \
        .take(k)

def get_most_probable_section(spark, road_df, center_neighbours, location):
    ''' Return the nearest road segment from a given location (long,lat) given the center of this segment.
    Procedure:
        Given a list of segment's centers that could be the nearest from location 'location':
            for each segment's center:
                retrieve its coordinates and return the nearest coordinate from location 'location'
        Now that we got the nearest coordinate of all most probable segments, find the nearest coordinate and return the corresponding segment.
    '''
    bests=list()
    for cn in center_neighbours :
        bests.append((road_df
            .filter(lambda c: c.center_long==cn.center_long and c.center_lat==cn.center_lat)
            .withColumn('dist', road_df.center_long **2 + road_df.center_lat**2)
            )


    """.map(lambda c: Row(center_long=c[0], center_lat=c[1], id=c[2], dist=euclidian_dist((c[0], c[1]), location)))
    .union(spark.parallelize([cn]))
    .sortBy(lambda x: x.dist)
    .take(1)))"""

    """bests = list(map(lambda el:el[0], bests))
    return spark.parallelize(bests) \
            .sortBy(lambda x: x.dist) \
            .take(1)"""
    return

#init spark
spark = pyspark.sql.SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

#retrieve datasets
accidents_df=extract_accidents_montreal_dataframe(spark)
road_df=extract_road_segments_DF(spark)

#get centers of road segments from road_df
centers = road_df.select("*") \
    .withColumn("id", monotonically_increasing_id()) \
    .select(['center_long', 'center_lat', 'id']) \
    .drop_duplicates(['center_long','center_lat'])

location = (-73.861616, 45.45505)
k = 10
centers_rdd=centers.rdd
accidents_rdd = accidents_df.select(['LOC_LONG', 'LOC_LAT']).rdd

'''test=accidents_rdd.map(lambda row: Row(value=(row.LOC_LONG,row.LOC_LAT)))
combine = centers_rdd.cartesian(test)'''

center_neighbours = get_nearest_neighbours(centers_rdd, location, k)
val = get_most_probable_section(spark, road_df, center_neighbours, location)
