from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.types import *
import os
import time

SCHEMA = StructType([StructField("Arrival_Time", LongType(), True),
                     StructField("Creation_Time", LongType(), True),
                     StructField("Device", StringType(), True),
                     StructField("Index", LongType(), True),
                     StructField("Model", StringType(), True),
                     StructField("User", StringType(), True),
                     StructField("gt", StringType(), True),
                     StructField("x", DoubleType(), True),
                     StructField("y", DoubleType(), True),
                     StructField("z", DoubleType(), True)])

spark = SparkSession.builder.appName('demo_app') \
    .config("spark.kryoserializer.buffer.max", "512m") \
    .getOrCreate()

os.environ['PYSPARK_SUBMIT_ARGS'] = \
    "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8,com.microsoft.azure:spark-mssql-connector:1.0.1"
kafka_server = 'dds2020s-kafka.eastus.cloudapp.azure.com:9092'
topic = "activities"

streaming = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option("subscribe", topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", False) \
    .option("maxOffsetsPerTrigger", 1000000) \
    .load() \
    .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")) \
    .select("value.*")

activityCounts = streaming.groupBy("gt").count()
activityQuery = activityCounts \
    .writeStream \
    .queryName("activity_counts") \
    .format("memory") \
    .outputMode("complete") \
    .start()

batch_id = 0
current_total = 0
while True:
    snapshot = spark.sql("SELECT * FROM activity_counts")
    new_total = snapshot.groupBy().sum("count").collect()[0]["sum(count)"]
    if new_total == current_total:
        continue
    current_total = new_total
    print("--------------------------------")
    print("Batch ID: {}".format(batch_id))
    print("--------------------------------")
    snapshot.show()
    print("total samples processed: {}".format(current_total))
    batch_id += 1
    activityQuery.awaitTermination()
