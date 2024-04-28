
import sys

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def data_cleaning(df): 
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))


if __name__ == "__main__":
    

    spark = SparkSession.builder.appName('rr724_cs643').getOrCreate()


    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

    if len(sys.argv) > 3:
        print("Usage: pyspark_wine_training.py <input_file>  <valid_path> <s3_output_bucketname>", file=sys.stderr)
        sys.exit(-1)
    elif len(sys.argv) == 3:
        input_path = sys.argv[1]
        valid_path = sys.argv[2]
        output_path = sys.argv[3] + "testmodel.model"
    else:
        input_path = "s3://wine-data-rr724/train.csv"
        valid_path = "s3://wine-data-rr724/valid.csv"
        output_path="s3://wine-data-rr724/testmodel.model"

    # read csv file in DataFrame
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(input_path))
    
    train_data_set = data_cleaning(df)

    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(valid_path))
    
    valid_data_set = data_cleaning(df)

    req = ['fixed acidity','volatile acidity','citric acid','chlorides','total sulfur dioxide','density','sulphates','alcohol',]
    
    assembler = VectorAssembler(inputCols=req, outputCol='features')
    
 
    indexer = StringIndexer(inputCol="quality", outputCol="label")


    train_data_set.cache()
    valid_data_set.cache()

    rf = RandomForestClassifier(labelCol='label', featuresCol='features',numTrees=150,maxBins=10, maxDepth=30,seed=100,impurity='gini')
    
    pipeline = Pipeline(stages=[assembler, indexer, rf])
    model = pipeline.fit(train_data_set)


    predictions = model.transform(valid_data_set)

 
    results = predictions.select(['prediction', 'label'])
    evaluator = MulticlassClassificationEvaluator(labelCol='label', 
                                        predictionCol='prediction', 
                                        metricName='accuracy')

    accuracy = evaluator.evaluate(predictions)
    print('Test Accuracy = ', accuracy)
    metrics = MulticlassMetrics(results.rdd.map(tuple))
    print('Weighted f1 score = ', metrics.weightedFMeasure())

    
    # Retrain model on mutiple parameters 
    cvmodel = None
    paramGrid = ParamGridBuilder() \
            .addGrid(rf.maxBins, [9, 8, 4])\
            .addGrid(rf.maxDepth, [25, 6 , 9])\
            .addGrid(rf.numTrees, [500, 50, 150])\
            .addGrid(rf.minInstancesPerNode, [6])\
            .addGrid(rf.seed, [100, 200, 5043, 1000])\
            .addGrid(rf.impurity, ["entropy","gini"])\
            .build()
    pipeline = Pipeline(stages=[assembler, indexer, rf])
    crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=2)

  
    cvmodel = crossval.fit(train_data_set)
    

    model = cvmodel.bestModel
    print(model)

    pred = model.transform(valid_data_set)
    res = pred.select(['prediction', 'label'])
    accuracy = evaluator.evaluate(pred)
    print('Test Accuracy is - ', accuracy)
    metrics = MulticlassMetrics(res.rdd.map(tuple))
    print('Weighted f1 score is - ', metrics.weightedFMeasure())

    model_path = output_path
    model.write().overwrite().save(model_path)
    sys.exit(0)