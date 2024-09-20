from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F, SparkSession

################### SPARK ##################
spark = SparkSession.builder.appName("movie Recommendation with Hadoop").getOrCreate()

########### Load the prepped data ##############
# als_data = spark.read.parquet("../datasets/als.parquet")
als_data = spark.read.parquet("hdfs:///user/mahyar/datasets/als.parquet")

als_data.show(5)
# cache
als_data.cache()

##### Split #########
(training, test) = als_data.randomSplit([0.8,0.2]) #TODO: tweakable

##### Train ####
als = ALS(maxIter= 10, regParam = 0.1, userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative= True, coldStartStrategy="drop")

als_model = als.fit(training)

##### Eval #####
predictions = als_model.transform(test)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"root main sqaure error = {rmse}")

# predictions.show(50)
# we got root main sqaure error = 0.8149086762553105. which is not bad but not great! let's start fine tuning this