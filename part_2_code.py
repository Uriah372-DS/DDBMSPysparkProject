import pyspark
import findspark
import pyspark.sql.functions as f
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, VectorSizeHint, OneHotEncoder, StringIndexer, IndexToString, SQLTransformer, Interaction, UnivariateFeatureSelector
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

findspark.init()


def init_spark(app_name: str):
    spark = SparkSession.builder \
            .appName(app_name) \
            .getOrCreate()
    return spark, spark.sparkContext


if __name__ == '__main__':
    print("---" * 20 + "Spark Project Part 2" + "---" * 20)
    print("Creating Active Global Session...")
    spark, sc = init_spark('Distributed Human Health Association')
    # conf_dict = sc._conf.getAll()
    # for i in range(len(conf_dict)): print(f"{conf_dict[i][0]}: {conf_dict[i][1]}")

    print("Loading The Static Data")
    data = spark.read.json('data.json').drop("Model")
    data.printSchema()
    data.show(5, truncate=False)
    print(f"Loaded {data.count()} Rows and {len(data.columns)} Columns.")
    print("Null Values in columns:")
    data.select([f.count(f.when(f.isnan(c) | f.col(c).isNull(), c)).alias(c) for c in data.columns]).show()

    print("---" * 20 + "Building Our Model" + "---" * 20)
    print("Building Preprocessing Pipeline...")
    userIndexer = StringIndexer(inputCol="User",
                                outputCol="userIndex",
                                handleInvalid="keep")
    userIndexerModel = userIndexer.fit(data)
    userToString = IndexToString(inputCol="userIndex", outputCol="User", labels=userIndexerModel.labels)

    deviceIndexer = StringIndexer(inputCol="Device",
                                  outputCol="DeviceIndex",
                                  handleInvalid="keep")
    deviceIndexerModel = deviceIndexer.fit(data)
    deviceToString = IndexToString(inputCol="DeviceIndex", outputCol="Device", labels=deviceIndexerModel.labels)

    gtIndexer = StringIndexer(inputCol="gt",
                              outputCol="label",
                              handleInvalid="keep")
    gtIndexerModel = gtIndexer.fit(data)
    gtToString = IndexToString(inputCol="label", outputCol="gt", labels=gtIndexerModel.labels)
    PredToString = IndexToString(inputCol="prediction", outputCol="pred", labels=gtIndexerModel.labels)

    features = ['Arrival_Time', 'DeviceIndex', 'userIndex', 'x', 'y', 'z']
    f_statement = str(['Arrival_Time', 'x', 'y', 'z', 'gt']).strip("[]").replace("'", "")

    feature_interactions = [Interaction(inputCols=[f1, f2],
                                        outputCol=str(f1 + ' ' + f2))
                            for i, f1 in enumerate(features)
                            for j, f2 in enumerate(features)
                            if i < j]
    feature_interactions_names = [str(f1 + ' ' + f2)
                                  for i, f1 in enumerate(features)
                                  for j, f2 in enumerate(features)
                                  if i < j]

    preprocessing_pipeline = Pipeline(stages=[
        gtIndexerModel,
        deviceIndexerModel,
        userIndexerModel,
        SQLTransformer(
            statement=f"SELECT {f_statement}, (DeviceIndex + 1) AS DeviceIndex, (userIndex + 1) AS userIndex, (label + 1) AS label FROM __THIS__"),
        *feature_interactions,
        VectorAssembler(inputCols=[*features, *feature_interactions_names],
                        outputCol='features'),
        VectorSizeHint(inputCol='features',
                       size=len(feature_interactions_names) + len(features),
                       handleInvalid='skip'),
        SQLTransformer(statement="SELECT features, label, gt FROM __THIS__")])

    print("Preprocessed Data Example:")
    preprocessed_data = preprocessing_pipeline.fit(data).transform(data).cache()
    preprocessed_data.show(5, truncate=False)

    print("Building Machine-Learning Model...")
    n_trees = 20
    rf = RandomForestClassifier(numTrees=n_trees,
                                featureSubsetStrategy='sqrt',
                                subsamplingRate=1.0,
                                maxDepth=5,
                                impurity='entropy',
                                seed=42)

    learning_model = rf
    print("Building Full Pipline...")

    basic_pipeline = Pipeline(stages=[
        preprocessing_pipeline,
        learning_model,
        PredToString
    ])

    evaluator = MulticlassClassificationEvaluator()

    (train_data, test_data) = data.randomSplit([0.7, 0.3], seed=42)
    train_data = train_data.cache()
    test_data = test_data.cache()

    pred_set = basic_pipeline.fit(train_data).transform(test_data)
    print(f"Accuracy Evaluation Example: {evaluator.evaluate(pred_set)}")

    (trainDF, testDF) = preprocessed_data.randomSplit([0.7, 0.3], seed=42)
    trainDF = trainDF.cache()
    testDF = testDF.cache()

    numTreesList = [25, 25, 30]
    maxDepthList = [20, 30, 30]

    train_scores = []
    test_scores = []
    values = list(zip(numTreesList, maxDepthList))

    for (n, m) in values:
        rf.setNumTrees(n)
        rf.setSubsamplingRate(1 / n)
        rf.setMaxDepth(m)
        train_pred = rf.fit(trainDF).transform(trainDF)
        test_pred = rf.fit(trainDF).transform(testDF)
        train_scores.append(evaluator.evaluate(train_pred))
        test_scores.append(evaluator.evaluate(test_pred))

    # plot the relationship between r and testing accuracy
    plt.plot(values, train_scores, label="train accuracy")
    plt.plot(values, test_scores, label="test accuracy")
    plt.xlabel('Parameters (numTrees, maxDepth)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Parameters for Random Forest Classifier')
    plt.show()

    # preprocessed_data = VectorAssembler(
    #     inputCols=["z"],
    #     outputCol='features').transform(preprocessed_data).select("features", "gtIndex")
    # preprocessed_data.show()
    # print("---" * 20 + "Performing Cross-Validation" + "---" * 20)
    # print("Building Model Evaluator...")
    # evaluator = MulticlassClassificationEvaluator(labelCol="gtIndex")
    #
    # print("Splitting the data to train and test by a 70:30 ratio...")
    # (trainDF, testDF) = preprocessed_data.randomSplit([0.7, 0.3], seed=42)
    #
    # print("Building Cross-Validator estimator and running Cross-Validation...")
    # paramGrid = ParamGridBuilder() \
    #     .addGrid(rf.numTrees, [10, 20]) \
    #     .addGrid(rf.maxDepth, [15, 25]) \
    #     .addGrid(rf.subsamplingRate, [0.7, 0.8]) \
    #     .build()
    #
    # numFolds = 5
    # crossval = TrainValidationSplit(
    #     estimator=rf,
    #     estimatorParamMaps=paramGrid,
    #     evaluator=evaluator,
    #     trainRatio=0.70, seed=42)
    #
    # cv_model = crossval.fit(trainDF)
    # pred_set = cv_model.transform(testDF)
    # print("Cross Validation Results for every parameters permutation", evaluator.evaluate(pred_set), "<-- much better!")
    # best_rf = cv_model.bestModel.stages[1].stages[0]
    # print("best estimator numTrees:", cv_model._java_obj.getNumTrees())
    # print("best estimator maxDepth:", cv_model._java_obj.getMaxDepth())
    # print("best estimator subsamplingRate:", cv_model._java_obj.getSubsamplingRate())
