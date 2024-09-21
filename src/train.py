from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col

################### SPARK ##################
spark = SparkSession.builder.appName("movie Recommendation with Hadoop").getOrCreate()

########### Load the prepped data ##############
# als_data = spark.read.parquet("../datasets/als.parquet")
als_data = spark.read.parquet("hdfs:///user/mahyar/datasets/als.parquet")

# als_data.show(5)
# cache
als_data.cache()

###### fine tuning ##########
# filter out users with less than 5 ratings
user_ratings_count = als_data.groupby("userID").count()
active_users = user_ratings_count.filter(col("count") >=5 )
als_data_filtered = als_data.join(active_users, "userID")

als_data_filtered.cache()

# filter out movies with less than x ratings TODO: let's see how the top one does alone first
# movie_ratings_count = als_data_filtered.groupBy("movieId").count()
# popular_movies = movie_ratings_count.filter(col("count") >= 5)
# als_data_filtered = als_data_filtered.join(popular_movies, "movieId")

##### Split #########
(training, test) = als_data_filtered.randomSplit([0.8,0.2], seed=42) #TODO: tweakable

##### Train ####
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative= True, coldStartStrategy="drop")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

param_grid = (ParamGridBuilder().addGrid(als.rank, [10,20,50])
              .addGrid(als.maxIter, [5,10,20])
              .addGrid(als.regParam, [0.01,0.1,0.5])
              .build())

cross_validator = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5) # TODO: Tweakable
cv_model = cross_validator.fit(training)

best_model = cv_model.bestModel

##### Eval #####
predictions = best_model.transform(test)


rmse = evaluator.evaluate(predictions)
print(f"root main sqaure error = {rmse}")

predictions.show(50)
# we got root main sqaure error = 0.8149086762553105. which is not bad but not great! let's start fine tuning this