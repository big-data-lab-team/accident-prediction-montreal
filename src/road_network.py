from urllib.request import urlopen, urlretrieve
from urllib.parse import quote
from urllib.error import URLError, HTTPError
import os
import re
from zipfile import ZipFile
from shutil import rmtree
from io import BytesIO
from bs4 import BeautifulSoup
from pyspark.sql import Window
from pyspark.sql.functions import (
    col,
    abs,
    hash,
    atan2,
    sqrt,
    cos,
    sin,
    radians,
    udf,
    monotonically_increasing_id,
    concat,
    row_number,
)
from pyspark.sql.types import StringType
from utils import raise_parquet_not_del_error
from workdir import workdir
from road_network_nids import unknow_file_included_nids


def get_road_df(spark, use_cache=True):
    fetch_road_network()
    return extract_road_segments_df(spark, use_cache)


def get_road_features_df(spark, road_df=None, use_cache=True):
    cache = workdir + "data/road-features.parquet"
    if os.path.isdir(cache) and use_cache:
        print("Skip extracting road features: already done")
        return spark.read.parquet(workdir + "data/road-features.parquet")

    road_df = road_df or get_road_df(spark)

    print("Extracting road features...")
    assign_street_type_udf = udf(assign_street_type, StringType())
    earth_diameter = 6371 * 2 * 1000  # in meters
    road_features = (
        road_df.select(
            "street_id",
            "street_type",
            "street_name",
            "coord_lat",
            "coord_long",
            "center_lat",
            "center_long",
        )
        .join(
            road_df.select(
                "street_id",
                col("coord_lat").alias("coord2_lat"),
                col("coord_long").alias("coord2_long"),
            ),
            "street_id",
        )
        .withColumn(
            "distance_inter",
            distance_intermediate_formula(
                "coord_lat", "coord_long", "coord2_lat", "coord2_long"
            ),
        )
        .withColumn("dist_measure", distance_measure())
        .select(
            "street_id",
            "street_type",
            "street_name",
            "dist_measure",
            "center_lat",
            "center_long",
        )
        .groupBy("street_id", "street_type", "street_name", "center_lat", "center_long")
        .max("dist_measure")
        .withColumn("street_length", col("max(dist_measure)") * earth_diameter)
        .select(
            "street_id",
            col("street_type").alias("street_level"),
            "street_name",
            "street_length",
            "center_lat",
            "center_long",
        )
        .withColumnRenamed("center_lat", "loc_lat")
        .withColumnRenamed("center_long", "loc_long")
        .withColumn("street_type", assign_street_type_udf(col("street_name")))
        .drop("street_name")
    )

    if use_cache:
        road_features.write.parquet(cache)
    print("Extracting road features: done")
    return road_features.distinct()


def fetch_road_network():
    if not os.path.isdir(workdir + "data"):
        os.mkdir(workdir + "data")
    if os.path.isfile(workdir + "data/road-network.lock"):
        print("Skip fetching road network: already downloaded")
        return
    print("Fetching road network...")
    url = (
        "http://ftp.maps.canada.ca/pub/nrcan_rncan" "/vector/geobase_nrn_rrn/qc/kml_en/"
    )
    if not os.path.isdir(workdir + "data/road-network"):
        os.mkdir(workdir + "data/road-network")
    files = [
        "TNO_terrestre_du_TE_de_Montréal_6_60.kmz",
        "Mont-Royal_4_58.kmz",
        "L'Île-Dorval_4_57.kmz",
        "Montréal_4_58.kmz",
        "Unknown_5_57.kmz",
        "Unknown_5_58.kmz",
        "Unknown_4_55.kmz",
        "Montréal_5_59.kmz",
        "Unknown_5_56.kmz",
        "Montréal_4_55.kmz",
        "L'Île-Dorval_4_58.kmz",
        "Mont-Royal_5_59.kmz",
        "Montréal_4_60.kmz",
        "Montréal_4_56.kmz",
        "Montréal_6_60.kmz",
        "Montréal-Ouest_4_58.kmz",
        "Senneville_4_56.kmz",
        "Montréal_6_58.kmz",
        "Unknown_4_59.kmz",
        "Montréal_4_57.kmz",
        "Montréal_5_58.kmz",
        "Montréal_7_60.kmz",
        "Unknown_6_60.kmz",
        "Mont-Royal_5_58.kmz",
        "Hampstead_4_58.kmz",
        "Montréal-Est_6_60.kmz",
        "Sainte-Anne-de-Bellevue_4_56.kmz",
        "Hampstead_4_59.kmz",
        "Dorval_4_57.kmz",
        "TNO_aquatique_du_TE_de_Montréal_5_60.kmz",
        "Montréal_5_60.kmz",
        "Unknown_4_60.kmz",
        "TNO_terrestre_du_TE_de_Montréal_5_60.kmz",
        "Montréal_5_56.kmz",
        "Côte-Saint-Luc_5_58.kmz",
        "Dollard-Des_Ormeaux_5_57.kmz",
        "Dorval_4_58.kmz",
        "Baie-D'Urfé_4_56.kmz",
        "TNO_aquatique_du_TE_de_Montréal_5_58.kmz",
        "Senneville_4_55.kmz",
        "Unknown_6_59.kmz",
        "Beaconsfield_4_57.kmz",
        "Sainte-Anne-de-Bellevue_4_55.kmz",
        "Montréal_5_57.kmz",
        "Westmount_4_59.kmz",
        "Beaconsfield_4_56.kmz",
        "Unknown_4_56.kmz",
        "Unknown_5_59.kmz",
        "Unknown_4_57.kmz",
        "Kirkland_4_57.kmz",
        "Westmount_5_59.kmz",
        "Côte-Saint-Luc_4_58.kmz",
        "Pointe-Claire_4_57.kmz",
        "Montréal_4_59.kmz",
        "Unknown_5_60.kmz",
        "Côte-Saint-Luc_4_59.kmz",
        "Unknown_4_58.kmz",
        "Dollard-Des_Ormeaux_4_56.kmz",
        "Dorval_5_57.kmz",
        "Montréal-Ouest_4_59.kmz",
        "Pointe-Claire_4_56.kmz",
        "Montréal-Est_6_59.kmz",
        "Dollard-Des_Ormeaux_4_57.kmz",
        "Montréal_6_59.kmz",
    ]
    for file in files:
        urlretrieve(
            f"{url}{quote(file)}", workdir + "data/road-network/{0}".format(file)
        )
    open(workdir + "data/road-network.lock", "wb").close()
    print("Fetching road network done")


def get_kml_content(soup):
    """Function to extract kml file content and store relevant information
    into a pandas dataframe.
    Args:
        soup: File content extracted using beautiful soup
    """
    rows = list()
    folders = soup.find_all("Folder")
    for folder in folders:
        street_type = folder.find("name").text
        placemarks = folder.find_all("Placemark")

        for placemark in placemarks:
            street_name = placemark.find("name").text
            center = placemark.MultiGeometry.Point.coordinates.text.split(",")
            coordinates_list = (
                placemark.MultiGeometry.LineString.coordinates.text.split(" ")
            )
            description = placemark.find("description").text
            nid = re.search("<th>nid</th>\n<td>([a-f0-9]+)</td>", description).group(1)
            is_unknown = (
                re.search(
                    "<th>left_OfficialPlaceName</th>\n<td>Unknown</td>", description
                )
                is not None
            )
            if is_unknown and (nid not in unknow_file_included_nids):
                continue

            for coord in coordinates_list:
                coords = coord.split(",")
                if len(coords) > 1:
                    rows.append(
                        [
                            street_name,
                            street_type,
                            float(center[0]),
                            float(center[1]),
                            float(coords[0]),
                            float(coords[1]),
                            nid,
                        ]
                    )
            # Add center of the street as a point of the street
            rows.append(
                [
                    street_name,
                    street_type,
                    float(center[0]),
                    float(center[1]),
                    float(center[0]),
                    float(center[1]),
                    nid,
                ]
            )
    return rows


def kml_extract_RDD(xml_file):
    """Function to extract the content of a kml input file and to store it
    into a csv output file.
    Args:
        xml_file_path: input kml file (kml is an xml file)
    """
    soup = BeautifulSoup(xml_file, "lxml-xml")
    return get_kml_content(soup)


def get_road_segments_RDD(spark):
    def read_doc_from_zip_file(file):
        file_path = workdir + "data/road-network/{0}".format(file)
        return BytesIO(ZipFile(file_path, "r").read("doc.kml"))

    return spark.sparkContext.parallelize(
        os.listdir(workdir + "data/road-network/")
    ).map(read_doc_from_zip_file)


def extract_road_segments_df(spark, use_cache=True):
    cache = workdir + "data/road-network.parquet"
    if os.path.isdir(cache) and use_cache:
        print(
            "Skip extraction of road network dataframe: already done,"
            " reading from file"
        )
        return spark.read.parquet(cache)

    print("Extracting road network dataframe...")
    cols = [
        "street_name",
        "street_type",
        "center_long",
        "center_lat",
        "coord_long",
        "coord_lat",
        "nid",
    ]

    road_seg_df = get_road_segments_RDD(spark).flatMap(kml_extract_RDD).toDF(cols)

    # Some specific road segments have the same nid
    w = Window.partitionBy("nid").orderBy("center_lat")
    street_ids = (
        road_seg_df.select("nid", "center_lat", "center_long")
        .distinct()
        .select(
            "center_lat",
            "center_long",
            concat("nid", row_number().over(w)).alias("street_id"),
        )
    )

    road_seg_df = road_seg_df.join(street_ids, ["center_lat", "center_long"]).drop(
        "nid"
    )

    if use_cache:
        road_seg_df.write.parquet(cache)
    print("Extracting road network dataframe done")
    return road_seg_df


def distance_intermediate_formula(lat1, long1, lat2, long2):
    """Returns spark expression computing intermediate result
    to compute the distance between to GPS coordinates
    Source: https://www.movable-type.co.uk/scripts/latlong.html
    """
    return pow(sin(radians(col(lat1) - col(lat2)) / 2), 2) + (
        pow(sin(radians(col(long1) - col(long2)) / 2), 2)
        * cos(radians(col(lat1)))
        * cos(radians(col(lat2)))
    )


def distance_measure():
    return atan2(sqrt(col("distance_inter")), sqrt(1 - col("distance_inter")))


def assign_street_type(street_name):
    possible_keywords = street_name.split(" ")[0:1]
    possible_keywords = [pk.lower() for pk in possible_keywords]
    assignation = {
        "allée": ["allée"],
        "autoroute": ["autoroute"],
        "avenue": ["avenue"],
        "boulevard": ["boulevard"],
        "carré": ["carré"],
        "square": ["square"],
        "carref.": ["carref."],
        "chemin": ["chemin"],
        "circle": ["circle", "cercle"],
        "côte": ["côte"],
        "cours": ["cours"],
        "court": ["court"],
        "crescent": ["crescent", "croissant"],
        "drive": ["drive"],
        "esplanade": ["esplanade"],
        "island": ["Île"],
        "impasse": ["impasse"],
        "lane": ["lane"],
        "lieu": ["lieu"],
        "montée": ["montée"],
        "park": ["parc", "park"],
        "passage": ["passage"],
        "place": ["place"],
        "pont": ["pont"],
        "promenade": ["promenade"],
        "rang": ["rang"],
        "road": ["road", "route"],
        "ruelle": ["ruelle"],
        "street": ["street", "rue"],
        "terrasse": ["terrasse"],
    }
    for street_type in assignation:
        for keyword in assignation[street_type]:
            if keyword in possible_keywords:
                return street_type
