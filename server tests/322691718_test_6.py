from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.window import Window
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

print(spark.version)
print('CPUs:', os.cpu_count())

streaming = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_server) \
    .option("subscribe", topic) \
    .option("startingOffsets", "earliest") \
    .option("failOnDataLoss", False) \
    .option("maxOffsetsPerTrigger", 400) \
    .load()

kafka_columns = list(streaming.columns)
kafka_columns.remove("key")
kafka_columns.remove("topic")
kafka_columns.remove("partition")
kafka_columns.remove("timestampType")

streaming = streaming \
    .select(*kafka_columns, f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value_decoded")) \
    .select(*kafka_columns, "value_decoded.*") \
    .drop("value")

stream_query = streaming \
    .writeStream \
    .queryName("Kafka_stream") \
    .format("memory") \
    .outputMode("append") \
    .start()
snapshot = spark.sql("select * from Kafka_stream") \
    .withColumn("monotonically_increasing_id", f.monotonically_increasing_id()) \
    .withColumn("row_number", f.row_number().over(Window.orderBy("monotonically_increasing_id")))
top_n = 5
last_offset = 0  # index of last loaded row
batch_id = 0  # iteration/batch/Kafka_stream epoch
while True:
    current_offset = snapshot.groupBy().max("offset").collect()[0]["max(offset)"]

    if current_offset is None or current_offset == last_offset:
        time.sleep(5)
        continue

    last_offset = current_offset
    print("Batch ID: {}".format(batch_id))
    print("----------------")
    print("offset: {}".format(last_offset))
    print("First {}:".format(top_n))
    snapshot.show(top_n, truncate=False)
    print("Last:")
    snapshot.where(last_offset - top_n <= f.col("offset")).show(truncate=False)

    print("Rows where row_number != offset:")
    snapshot.where(f.col("row_number") != f.col("offset")).show(truncate=False)

    timelapse_offset = snapshot.groupBy().max("offset").collect()[0]["max(offset)"]
    if current_offset == timelapse_offset:
        print("snapshot wasn't updated during iteration")
    else:
        print("snapshot was updated during iteration.")
        print("{} Row(s) added".format(timelapse_offset - current_offset))

    batch_id += 1
    time.sleep(5)

# output example: output_ex_4_1.jpg, output_ex_4_2.jpg
