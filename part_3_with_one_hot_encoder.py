import pyspark.sql.functions as f
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, VectorSizeHint, StringIndexer, SQLTransformer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors, VectorUDT
import pyspark.sql.types as t
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

spark = SparkSession.builder.appName('demo_app')\
    .config("spark.kryoserializer.buffer.max", "512m")\
    .getOrCreate()

os.environ['PYSPARK_SUBMIT_ARGS'] = \
    "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8,com.microsoft.azure:spark-mssql-connector:1.0.1"
kafka_server = 'dds2020s-kafka.eastus.cloudapp.azure.com:9092'
topic = "activities"

streaming = spark.readStream\
                .format("kafka")\
                .option("kafka.bootstrap.servers", kafka_server)\
                .option("subscribe", topic)\
                .option("startingOffsets", "earliest")\
                .option("failOnDataLoss", False)\
                .option("maxOffsetsPerTrigger", 10000)\
                .load()\
                .select(f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value"))\
                .select("value.*")

print(spark.version)
print('CPUs:', os.cpu_count())


original_features = ['Arrival_Time', 'Creation_Time', 'Device', 'Model', 'User', 'x', 'y', 'z', 'gt']
features = ['Arrival_Time', 'Creation_Time', 'ModelIndex', 'DeviceIndex', 'userIndex', 'x', 'y', 'z']
features2 = ['Arrival_Time', 'Creation_Time','ModelIndex_ohe', 'DeviceIndex_ohe', 'userIndex_ohe', 'x', 'y', 'z']
f_statement = str(original_features).strip("[]").replace("'", "")
# sql1 = "SELECT " + f_statement + " FROM __THIS__"
# sql2 = "SELECT " + f_statement + ", (DeviceIndex + 1) AS DeviceIndex,(ModelIndex + 1) AS ModelIndex, (userIndex + 1) AS userIndex, (label + 1) AS label FROM __THIS__"
# sql3 = "SELECT " + f_statement + ", features, label FROM __THIS__"
# sql1 = "SELECT Arrival_Time, Creation_Time,Device,Model, User,ModelIndex, DeviceIndex, userIndex, x, y, z, label FROM __THIS__"
# sql2 = "SELECT Arrival_Time, Creation_Time,Device,Model, User, x, y, z, (DeviceIndex + 1) AS DeviceIndex,(ModelIndex + 1) AS ModelIndex, (userIndex + 1) AS userIndex, (label + 1) AS label FROM __THIS__"
# sql3 = "SELECT Arrival_Time, Creation_Time,Device,Model, User,ModelIndex, DeviceIndex, userIndex, x, y, z, features, label FROM __THIS__"
sql1 = "SELECT Arrival_Time, Creation_Time, Device, Model, User, ModelIndex, DeviceIndex, userIndex, x, y, z, label FROM __THIS__"
sql2 = "SELECT Arrival_Time, Creation_Time, Device, Model, User, x, y, z, (DeviceIndex + 1) AS DeviceIndex, (ModelIndex + 1) AS ModelIndex, (userIndex + 1) AS userIndex, (label + 1) AS label FROM __THIS__"
sql3 = "SELECT Arrival_Time, Creation_Time, Device, Model, User, ModelIndex, DeviceIndex, userIndex, x, y, z, features, label FROM __THIS__"

feature_interactions_names = [str(f1+' '+f2)
                              for i, f1 in enumerate(features)
                              for j, f2 in enumerate(features)
                              if i < j]

features_to_remove = ['Arrival_Time Creation_Time', 'Creation_Time userIndex', 'Arrival_Time x',
                      'userIndex x', 'Creation_Time x', 'x z', 'DeviceIndex x', 'DeviceIndex z']
for feature in features_to_remove:
    feature_interactions_names.remove(feature)

# interactions_features = ['Arrival_Time Device_OneHotEncoder', 'Arrival_Time User_OneHotEncoder', 'Arrival_Time y', 'Arrival_Time z',
#                          'Creation_Time Device_OneHotEncoder', 'Creation_Time y', 'Creation_Time z',
#                          'User_OneHotEncoder y', 'User_OneHotEncoder z', 'x y', 'y z']


# feature_interactions = [Interaction(inputCols=s.split(sep=' '),
#                                     outputCol=s)
#                         for s in feature_interactions_names]

final_features = [*features, *feature_interactions_names]

preprocessing_pipeline = Pipeline(stages=[
    StringIndexer(inputCol="gt", outputCol="label", handleInvalid="keep"),
    StringIndexer(inputCol="Model", outputCol="ModelIndex", handleInvalid="keep"),
    StringIndexer(inputCol="Device", outputCol="DeviceIndex", handleInvalid="keep"),
    StringIndexer(inputCol="User", outputCol="userIndex", handleInvalid="keep"),
    # OneHotEncoder(inputCol="ModelIndex", outputCol="ModelIndex_ohe", handleInvalid="keep"),
    # OneHotEncoder(inputCol="DeviceIndex", outputCol="DeviceIndex_ohe", handleInvalid="keep"),
    # OneHotEncoder(inputCol="userIndex", outputCol="userIndex_ohe", handleInvalid="keep"),
    SQLTransformer(statement=sql1),
    # SQLTransformer(statement=sql2),
    # *feature_interactions,
    # VectorAssembler(inputCols=features, ######### final_features
    #                 outputCol='features'),
    # VectorSizeHint(inputCol='features',
    #                 size=len(features),######## final_features
    #                 handleInvalid='optimistic'),
    # SQLTransformer(statement=sql3)
    ])

evaluator = MulticlassClassificationEvaluator()

learning_pipeline = RandomForestClassifier(numTrees=15,
                                           maxDepth=10)

pipeline = Pipeline(stages=[preprocessing_pipeline])


def interactions(df, col1, col2):
    to_array = f.udf(lambda v: v.toArray().tolist(), t.ArrayType(t.FloatType()))
    to_svector = f.udf(lambda v: Vectors.sparse(9, [(index, x) for index, x in enumerate(v) if x != 0]), VectorUDT())
    to_dvector = f.udf(lambda v: Vectors.dense(v), VectorUDT())
    multiplying_scalar_col = f.udf(lambda s, v: v * s, t.FloatType())
    multiplying_vec_col = f.udf(lambda s, v: [x * s for x in v], t.ArrayType(t.FloatType()))
    col1type = df.schema[col1].dataType
    col2type = df.schema[col2].dataType
    ncol = col1 + ' ' + col2
    if col1type == VectorUDT() and col2type != VectorUDT():
        df = df.withColumn(ncol, to_array(f.col(col1)))
        df = df.withColumn(ncol, multiplying_vec_col(f.col(col2), f.col(ncol)))
        df = df.withColumn(ncol, to_dvector(f.col(ncol)))

    if col1type != VectorUDT() and col2type == VectorUDT():
        df = df.withColumn(ncol, to_array(f.col(col2)))
        df = df.withColumn(ncol, multiplying_vec_col(f.col(col1), f.col(ncol)))
        df = df.withColumn(ncol, to_dvector(f.col(ncol)))

    if col1type != VectorUDT() and col2type != VectorUDT():
        df = df.withColumn(ncol, multiplying_scalar_col(f.col(col1), f.col(col2)))

    if col1type == VectorUDT() and col2type == VectorUDT():
        return df

    return df


# select(['*']+[interactions(Kafka_stream,x.split(" ")[0],x.split(" ")[1]).select(x) for x in feature_interactions_names])\

print("Starting Streaming...")
# create a query that reads all the data and saves it to memory sink
streamQuery = streaming\
    .writeStream \
    .format("memory") \
    .queryName("sink") \
    .start()

# let the Kafka_stream run for a while first so that the table gets populated
while len(spark.sql("select * from sink").head(1)) == 0:
    time.sleep(5)

snapshot = spark.sql("select * from sink")

gtst = StringIndexer(inputCol="gt", outputCol="label", handleInvalid="keep")
model = gtst.fit(snapshot)
snapshot = model.transform(snapshot)

# after there is data to read, give initial, arbitrary prediction and report accuracy
print("Initial Evaluation:")
eval = evaluator.evaluate(snapshot.withColumn("prediction", f.lit(1.0)))
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
        # num_cat_of_user = pretrain.select('userIndex').distinct().count()
        # num_cat_of_device = pretrain.select('DeviceIndex').distinct().count()
        # num_cat_of_model = pretrain.select('ModelIndex').distinct().count()
        preprocessing_pipeline = Pipeline(stages=[
            OneHotEncoder(inputCol="ModelIndex", outputCol="ModelIndex_ohe"),
            OneHotEncoder(inputCol="DeviceIndex", outputCol="DeviceIndex_ohe"),
            OneHotEncoder(inputCol="userIndex", outputCol="userIndex_ohe"),
            VectorAssembler(inputCols=features2,  # final_features
                            outputCol='features'),
            # VectorSizeHint(inputCol='features',
            #                size=len(features2)+num_cat_of_device + num_cat_of_model + num_cat_of_user - 3,  ######## final_features
            #                handleInvalid='optimistic'),
            SQLTransformer(statement=sql3)])
        pipeline = Pipeline(stages=[preprocessing_pipeline, learning_pipeline])
        model = pipeline.fit(snapshot)
        snapshot = model.transform(snapshot)

        print("evaluating...")
        print("Accuracy: " + str(evaluator.evaluate(snapshot)))

        print("training estimator...")
        original_features.remove('gt')
        original_features.append('label')
        snapshot = snapshot.select(*original_features)
        snapshot = snapshot.withColumnRenamed("label", "gt")
        original_features.remove('label')
        original_features.append('gt')
        model = pipeline.fit(snapshot)

        epoch_id += 1
        time.sleep(5)

print("ending Kafka_stream...")
streamQuery.stop()
