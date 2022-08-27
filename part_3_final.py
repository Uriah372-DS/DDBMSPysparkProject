import pyspark
import pyspark.sql.functions as f
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer, SQLTransformer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType
import os
import time


class pysparkPrequentialEvaluation:
    """ Our version of the prequential evaluation method or interleaved test-then-train method in pyspark,
        using kafka data source.

        The prequential evaluation is designed specifically for Kafka_stream settings,
        in the sense that each sample serves two purposes, and that samples are
        analysed sequentially, in order of arrival.
        This method consists of using each sample to test the estimator, which means
        to make a prediction, and then the same sample is used to train the estimator.
        This way the estimator is always tested on samples that it hasn't seen yet.

        Our implementation works on any data source by adding an index column artificially,
        by using the :func:`row_number` function
        over the :func:`monotonically_increasing_id` function.

        """

    def __init__(self,
                 batch_size=100,
                 pretrain_size=200,
                 measure_time=False,
                 use_distribution=False,
                 metrics: list[str] | None = None,
                 metric_label: float = 0.0):
        """

        Parameters
        ----------
        batch_size: int (Default: 100)
            The number of samples to pass at a time to the estimator.
        pretrain_size: int (Default: 200)
            The number of samples to use to train the estimator before starting the evaluation.
        measure_time: bool (Default: False)
            Weather to measure the duration of the simulation (in seconds).
        use_distribution: bool (Default: False)
            Weather to measure the label distribution of the data when deciding to re-train the estimator.
            Helpful when the estimator has high variance and is easily affected by imbalanced data.
            Note - not implemented yet.
        metrics: list, optional (Default: ['accuracy'])
            | The list of metrics to track during the evaluation. Also defines the metrics
                that will be displayed in plots and/or logged into the output file. Valid options are
            |"accuracy"
            |"f1"
            |"weightedPrecision"
            |"weightedRecall"
            |"weightedTruePositiveRate"
            |"weightedFalsePositiveRate"
            |"weightedFMeasure"
            |"truePositiveRateByLabel"
            |"falsePositiveRateByLabel"
            |"precisionByLabel"
            |"recallByLabel"
            |"fMeasureByLabel"
        metric_label:
            The class whose metric will be computed in
            truePositiveRateByLabel|falsePositiveRateByLabel|precisionByLabel|recallByLabel|fMeasureByLabel.
            Must be >= 0. The default value is 0. (default: 0.0)

        """
        self.batch_size = batch_size
        self.pretrain_size = pretrain_size
        self.measure_time = measure_time
        self.use_distribution = use_distribution
        if metrics is None:
            self.metrics = ['accuracy']
        elif not isinstance(metrics, list):
            raise ValueError(
                "Attribute 'metrics' must be 'None' or 'list', passed {}".format(
                    type(metrics)))
        else:
            self.metrics = metrics
        self.n_metrics = len(self.metrics)
        self.metric_label = metric_label

    def evaluate(self,
                 termination_buffer: int,
                 estimator: pyspark.ml.Estimator,
                 stream: pyspark.sql.DataFrame,
                 label_col: str = "label",
                 prediction_col: str = "prediction",
                 index_col: str = None):
        """
        perform prequential evaluation on the given estimator.

        Parameters
        ----------
        termination_buffer: int
            number of Kafka_stream batches without new data after which to terminate the evaluation.
        estimator: pyspark.ml.Estimator
            an estimator to train on the data Kafka_stream.
        stream: pyspark.sql.DataFrame
            a Kafka_stream dataframe.
        label_col: str
            name of the label column.
        prediction_col: str
            name of the prediction column.
        index_col: str or None
            name of the index column, assumed to be consecutive.

        Returns
        -------
        pyspark.ml.Model
            the last trained estimator.

        """
        print("Starting Stream...")
        # create a query that reads all the data and saves it to memory sink
        stream_query = stream \
            .writeStream \
            .format("memory") \
            .queryName("sink") \
            .start()

        start_time = 0
        if self.measure_time:
            start_time = time.time()

        # let the Kafka_stream run for a while first so that the table gets populated
        cold_streams = 0  # cold Kafka_stream = Kafka_stream where no new data arrived
        session = SparkSession.builder.getOrCreate()
        while session.sql("select * from sink").count() < self.pretrain_size and \
                cold_streams < termination_buffer:
            cold_streams += 1
            time.sleep(10)

        sink = session.sql("select * from sink")
        if index_col is None:
            index_col = "row_number"
            sink = sink \
                .withColumn("monotonically_increasing_id", f.monotonically_increasing_id()) \
                .withColumn(index_col, f.row_number().over(Window.orderBy("monotonically_increasing_id")))

        # after there is enough data to read, give initial, arbitrary prediction and report accuracy
        model = self._initial_evaluation(pretrain=sink.filter(f.col(index_col) < self.pretrain_size),
                                         estimator=estimator,
                                         label_col=label_col,
                                         prediction_col=prediction_col)

        # every 5 seconds scan for new arrived data, predict and report accuracy and then fit estimator on the predicted data.
        # do this until no new data arrives for a sufficient amount of time, then stop Kafka_stream.

        n_used = self.pretrain_size  # total number of rows used in training so far
        batch_id = 1  # iteration/batch/Kafka_stream epoch.
        while cold_streams < termination_buffer:

            n_current = sink.groupBy().max(index_col).collect()[0]["max(" + index_col + ")"]

            if n_current == n_used:
                cold_streams += 1
                time.sleep(5)
                continue

            print("==" * 20)
            print("Batch ID: " + str(batch_id))
            print("==" * 20)
            batch = sink.filter(n_used <= f.col(index_col)).filter(f.col(index_col) < n_used + self.batch_size)

            print("Predicting on new batch...")
            predicted = model.transform(batch)

            print("Evaluation:")
            self._print_evaluation(df=predicted,
                                   label_col=label_col,
                                   prediction_col=prediction_col)

            print("Training Estimator on predicted batch...")
            model = estimator.fit(batch)  # training on all data, both used and unused

            n_used += batch.count()
            batch_id += 1
            time.sleep(5)

        print("Stopping Stream...")
        stream_query.stop()
        if self.measure_time:
            print("Elapsed Time: {} Seconds".format(time.time() - start_time))
        return model

    def _initial_evaluation(self, pretrain: pyspark.sql.DataFrame,
                            estimator: pyspark.ml.Estimator,
                            label_col: str,
                            prediction_col: str):
        # # get most frequent label:  -- this is training!!!
        # some_label = pretrain.groupBy(label_col).count().groupBy().max("count").collect()[0][label_col]
        # here we are not using the estimator's predictions, only it's label values.
        # so this is NOT considered training, it's only for column names compatibility
        temp_pretrain = estimator.fit(pretrain).transform(pretrain)
        some_label = temp_pretrain.select(label_col).collect()[0][label_col]
        # give label as prediction:
        temp_snapshot = temp_pretrain.withColumn(prediction_col, f.lit(some_label))
        print("Initial Prediction & Evaluation:")
        self._print_evaluation(temp_snapshot, label_col=label_col, prediction_col=prediction_col)

        print("Initial Training:")
        return estimator.fit(pretrain)

    def _print_evaluation(self, df: pyspark.sql.DataFrame, label_col: str, prediction_col: str):
        evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol=prediction_col)
        metric_values = []
        for metric in self.metrics:
            evaluation_result = evaluator.evaluate(df, {evaluator.metricName: metric,
                                                        evaluator.metricLabel: self.metric_label})
            r_dict = {"metric": metric, "value": evaluation_result}
            row = Row(**r_dict)
            metric_values.append(row)
        df.sparkSession \
            .createDataFrame(data=metric_values, verifySchema=False) \
            .show(n=self.n_metrics, truncate=False)


if __name__ == '__main__':
    print("=" * 80)
    print("Prequential Evaluation Implementation Test in Spark Structured Streaming")
    print("Initiating Spark Session...")
    spark = SparkSession.builder.appName('prequential_evaluation') \
        .config("spark.kryoserializer.buffer.max", "512m") \
        .getOrCreate()

    os.environ['PYSPARK_SUBMIT_ARGS'] = \
        "--packages=org.apache.spark:spark-sql-kafka-0-10_2.12:2.4.8,com.microsoft.azure:spark-mssql-connector:1.0.1"

    print("done")
    print("Spark Version: {}".format(spark.version))
    print('CPUs:', os.cpu_count())
    print("=" * 40)
    print("Configuring Kafka Stream Settings And Features...")
    kafka_server = 'dds2020s-kafka.eastus.cloudapp.azure.com:9092'
    topic = "activities"
    Kafka_stream = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_server) \
        .option("subscribe", topic) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", False) \
        .option("maxOffsetsPerTrigger", 400) \
        .load()

    kafka_columns = list(Kafka_stream.columns)
    kafka_columns_to_remove = ["key", "topic", "partition", "timestampType", "timestamp"]
    for col in kafka_columns_to_remove:
        kafka_columns.remove(col)

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
    initial_features = ['Arrival_Time', 'Creation_Time', 'Device', 'Index', 'Model', 'User', 'x', 'y', 'z', 'gt']

    Kafka_stream = Kafka_stream \
        .select(*kafka_columns, f.from_json(f.decode("value", "US-ASCII"), schema=SCHEMA).alias("value_decoded")) \
        .select(*kafka_columns, "value_decoded.*") \
        .drop("value")

    offset_col = "offset"
    print("done")
    print("=" * 40)

    print("Building Estimator...")
    features = ['Arrival_Time', 'Creation_Time', 'DeviceIndex', 'ModelIndex', 'UserIndex', 'x', 'y', 'z']
    target = 'gt'

    # # For Creating Feature Interactions on Server With spark v3:
    # feature_interactions_names = [str(f1 + ' ' + f2)
    #                               for i, f1 in enumerate(features)
    #                               for j, f2 in enumerate(features)
    #                               if i < j]
    # features_to_remove = ['Arrival_Time Creation_Time', 'Creation_Time UserIndex', 'Arrival_Time x',
    #                       'UserIndex x', 'Creation_Time x', 'x z', 'DeviceIndex x', 'DeviceIndex z']
    # for feature in features_to_remove:
    #     feature_interactions_names.remove(feature)

    final_features = [*features]

    sql1 = "SELECT Arrival_Time, Creation_Time, Device, Model, User, x, y, z, gt FROM __THIS__"
    sql3 = "SELECT gt, features, label FROM __THIS__"

    preprocessing_pipeline = Pipeline(stages=[
        SQLTransformer(statement=sql1),
        StringIndexer(inputCol=target, outputCol="label", handleInvalid="keep"),
        StringIndexer(inputCol="Device", outputCol="DeviceIndex", handleInvalid="keep"),
        StringIndexer(inputCol="Model", outputCol="ModelIndex", handleInvalid="keep"),
        StringIndexer(inputCol="User", outputCol="UserIndex", handleInvalid="keep"),
        VectorAssembler(inputCols=final_features,
                        outputCol='features'),
        SQLTransformer(statement=sql3)])

    learning_pipeline = RandomForestClassifier(numTrees=15,
                                               maxDepth=10)

    est = Pipeline(stages=[preprocessing_pipeline, learning_pipeline])
    print("done")
    print("=" * 40)

    print("Configuring Prequential Evaluation Settings...")
    prequential_evaluator = pysparkPrequentialEvaluation(batch_size=400,
                                                         pretrain_size=400,
                                                         measure_time=False,
                                                         use_distribution=False,
                                                         metrics=None,
                                                         metric_label=0.0)
    print("done")
    print("=" * 20 + " Starting Prequential Evaluation " + "=" * 20)
    prequential_evaluator.evaluate(termination_buffer=3,
                                   estimator=est,
                                   stream=Kafka_stream,
                                   label_col="label",
                                   prediction_col="prediction",
                                   index_col=offset_col)
    print("Finished Testing Prequential Evaluation Implementation In Pyspark.")
    print("=" * 80)
