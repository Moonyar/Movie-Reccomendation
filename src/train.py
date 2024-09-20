from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

################### SPARK ##################
spark = SparkSession.builder.appName("movie Recommendation").getOrCreate()

########### Load the prepped data ##############
als_data = spark.read.parquet("../datasets/als.parquet")

# I will first train the data with userID, movieID, and rating for ALS
als_data_filtered = als_data.select("userId", "movieId", "rating")
als_data_filtered.cache()

##### Split #########
(training, test) = als_data_filtered.randomSplit([0.8,0.2]) #TODO: tweakable

##### Train ####
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative= True, coldStartStrategy="drop")

als_model = als.fit(training)

##### Eval #####
predictions = als_model.transform(test)

evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print(f"root main sqaure error = {rmse}")
