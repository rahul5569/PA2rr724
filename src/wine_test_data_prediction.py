import os
import sys

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col

def data_cleaning(df):
     
    return df.select(*(col(c).cast("double").alias(c.strip("\"")) for c in df.columns))

    
if __name__ == "__main__":
    
    
    spark = SparkSession.builder.appName('rr724_cs643').getOrCreate()

  
    sc = spark.sparkContext
    sc.setLogLevel('ERROR')

   
    if len(sys.argv) > 3:
        print("Usage: wine_test_data_prediction.py <input_data_file> <model_path>", file=sys.stderr)
        sys.exit(-1)
    elif len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if not("/" in input_path):
            input_path = "data/csv/" + input_path
        model_path="/code/data/model/testdata.model"
        print("----Input file for test data is---")
        print(input_path)
    else:
        current_dir = os.getcwd() 
        print("-----------------------")
        print(current_dir)
        input_path = os.path.join(current_dir, "data/csv/testdata.csv")
        model_path= os.path.join(current_dir, "data/model/testdata.model")

    # read csv file in DataFram 
    df = (spark.read
          .format("csv")
          .option('header', 'true')
          .option("sep", ";")
          .option("inferschema",'true')
          .load(input_path))
    
    df1 = data_cleaning(df)

    req = ['fixed acidity','volatile acidity','citric acid','chlorides','total sulfur dioxide','density','sulphates','alcohol',]
    
   
    rf = PipelineModel.load(model_path)
    
    predictions = rf.transform(df1)
    print(predictions.show(5))
    res = predictions.select(['prediction', 'label'])
    eval = MulticlassClassificationEvaluator(
                                            labelCol='label', 
                                            predictionCol='prediction', 
                                            metricName='accuracy')


    acc = eval.evaluate(predictions)
    print('Test Accuracy is ', acc)
    metrics = MulticlassMetrics(acc.rdd.map(tuple))
    print('Weighted f1 score is ', metrics.weightedFMeasure())
    sys.exit(0)
