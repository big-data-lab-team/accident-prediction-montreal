from datetime import datetime
from math import pi
from pyspark.sql.functions import col, cos, sin, asin, when, dayofyear, radians, degrees


def add_solar_features(df):
    return (
        df.withColumn(
            "declination_angle",
            radians(-23.45 * cos(((2 * pi) / 365) * (dayofyear("date") + 10))),
        )
        .withColumn("diff_local_time_UTC", timezone_from_date("date"))
        .withColumn("d", (2 * pi * dayofyear("date")) / 365)
        .withColumn(
            "equation_of_time",
            -7.655 * sin(col("d")) + 9.873 * sin(2 * col("d") + 3.588),
        )
        .drop("d")
        .withColumn(
            "time_correction",
            4 * (col("loc_long") - (15 * col("diff_local_time_UTC")))
            + col("equation_of_time"),
        )
        .withColumn("local_solar_hour", col("hour") + 0.5 + col("time_correction") / 60)
        .withColumn("hour_angle", 0.2618 * (col("local_solar_hour") - 12))
        .drop(
            "diff_local_time_UTC",
            "equation_of_time",
            "time_correction",
            "local_solar_hour",
        )
        .withColumn(
            "solar_elevation",
            degrees(
                asin(
                    sin("declination_angle") * sin(radians("loc_lat"))
                    + cos("declination_angle")
                    * cos(radians("loc_lat"))
                    * cos("hour_angle")
                )
            ),
        )
        .drop("declination_angle", "hour_angle")
    )


def timezone_from_date(date):
    todate = datetime.fromisoformat
    return when(
        (
            (col(date) > todate("2012-03-11 02:00:00"))
            & (col(date) < todate("2012-11-04 02:00:00"))
        )
        | (
            (col(date) > todate("2013-03-10 02:00:00"))
            & (col(date) < todate("2013-11-03 02:00:00"))
        )
        | (
            (col(date) > todate("2014-03-09 02:00:00"))
            & (col(date) < todate("2014-11-02 02:00:00"))
        )
        | (
            (col(date) > todate("2015-03-08 02:00:00"))
            & (col(date) < todate("2015-11-01 02:00:00"))
        )
        | (
            (col(date) > todate("2016-03-13 02:00:00"))
            & (col(date) < todate("2016-11-06 02:00:00"))
        )
        | (
            (col(date) > todate("2017-03-12 02:00:00"))
            & (col(date) < todate("2017-11-05 02:00:00"))
        )
        | (
            (col(date) > todate("2018-03-11 02:00:00"))
            & (col(date) < todate("2018-11-04 02:00:00"))
        ),
        -4,
    ).otherwise(-5)
