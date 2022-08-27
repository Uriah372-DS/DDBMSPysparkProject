import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, VectorSizeHint, StringIndexer, SQLTransformer, Interaction
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType
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
    .option("maxOffsetsPerTrigger", 432) \
    .load() \
    .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value")) \
    .select("value.*")

print(spark.version)
print('CPUs:', os.cpu_count())

original_features = ['Arrival_Time', 'Creation_Time', 'Device', 'User', 'x', 'y', 'z', 'gt']
features = ['Arrival_Time', 'Creation_Time', 'DeviceIndex', 'userIndex', 'x', 'y', 'z']
features2 = ['Arrival_Time', 'Creation_Time', 'ModelIndex_ohe', 'DeviceIndex_ohe', 'userIndex_ohe', 'x', 'y', 'z']
f_statement = str(original_features).strip("[]").replace("'", "")
# sql1 = "SELECT " + f_statement + " FROM __THIS__"
# sql2 = "SELECT " + f_statement + ", (DeviceIndex + 1) AS DeviceIndex, (userIndex + 1) AS userIndex, (label + 1) AS label FROM __THIS__"
# sql3 = "SELECT " + f_statement + ", features, label FROM __THIS__"
s1 = "SELECT Arrival_Time, Creation_Time, Device, Model, User, ModelIndex, DeviceIndex, userIndex, x, y, z, label FROM __THIS__"
s2 = "SELECT Arrival_Time, Creation_Time, Device, Model, User, x, y, z, (DeviceIndex + 1) AS DeviceIndex, (ModelIndex + 1) AS ModelIndex, (userIndex + 1) AS userIndex, (label + 1) AS label FROM __THIS__"
s3 = "SELECT Arrival_Time, Creation_Time, Device, Model, User, ModelIndex, DeviceIndex, userIndex, x, y, z, features, label FROM __THIS__"

feature_interactions_names = [str(f1 + ' ' + f2)
                              for i, f1 in enumerate(features)
                              for j, f2 in enumerate(features)
                              if i < j]

features_to_remove = ['Arrival_Time Creation_Time', 'Creation_Time userIndex', 'Arrival_Time x',
                      'userIndex x', 'Creation_Time x', 'x z', 'DeviceIndex x', 'DeviceIndex z']
for feature in features_to_remove:
    feature_interactions_names.remove(feature)

feature_interactions = [Interaction(inputCols=s.split(sep=' '),
                                    outputCol=s)
                        for s in feature_interactions_names]

final_features = [*features, *feature_interactions_names]

preprocessing_pipeline = Pipeline(stages=[
    SQLTransformer(statement=s1),
    StringIndexer(inputCol="gt", outputCol="label", handleInvalid="keep"),
    StringIndexer(inputCol="Device", outputCol="DeviceIndex", handleInvalid="keep"),
    StringIndexer(inputCol="User", outputCol="userIndex", handleInvalid="keep"),
    SQLTransformer(statement=s2),
    VectorAssembler(inputCols=final_features,
                    outputCol='features'),
    VectorSizeHint(inputCol='features',
                   size=len(final_features),
                   handleInvalid='optimistic'),
    SQLTransformer(statement=s3)])

evaluator = MulticlassClassificationEvaluator()

learning_pipeline = RandomForestClassifier(numTrees=15,
                                           maxDepth=10)

pipeline = Pipeline(stages=[preprocessing_pipeline, learning_pipeline])

print("Starting Streaming...")
# create a query that reads all the data and saves it to memory sink
streamQuery = streaming \
    .writeStream \
    .format("memory") \
    .queryName("sink") \
    .start()

# let the Kafka_stream run for a while first so that the table gets populated
while len(spark.sql("select * from sink").head(1)) == 0:
    time.sleep(5)

snapshot = spark.sql("select * from sink")

# after there is data to read, give initial, arbitrary prediction and report accuracy
print("Initial Evaluation:")
eval = evaluator.evaluate(snapshot.withColumn("prediction", f.lit(1)).withColumn("label", f.col("gt")))
print("Accuracy: " + str(eval))

# now give estimator initial training on the data we predicted
print("Initial Training...")
snapshot = snapshot.select(*original_features)
model = pipeline.fit(snapshot)

# every 5 seconds scan for new arrived data, predict and report accuracy and then fit estimator on the predicted data.
# do this until no new data arrives for a sufficient amount of time, then stop Kafka_stream.

n_loaded = snapshot.count()  # total number of rows loaded so far
epoch_id = 0  # iteration/batch/Kafka_stream epoch.
cold_streams = 0  # cold Kafka_stream = Kafka_stream where no new data arrived
termination_buffer = 3  # terminate after "termination_buffer" cold streams
while cold_streams < termination_buffer:

    snapshot = spark.sql("select * from sink")
    n_current = snapshot.count()

    if n_current == n_loaded:
        cold_streams += 1
        time.sleep(5)
        continue
    else:
        n_loaded = n_current

        print("==" * 20)
        print("Batch: " + str(epoch_id))
        print("==" * 20)

        print("predicting...")
        snapshot = model.transform(snapshot)

        print("evaluating...")
        print("Accuracy: " + str(evaluator.evaluate(snapshot)))

        print("training estimator...")
        snapshot = snapshot.select(*original_features)
        model = pipeline.fit(snapshot)

        epoch_id += 1
        time.sleep(5)

print("ending Kafka_stream...")
streamQuery.stop()
