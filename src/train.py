from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import col

################### SPARK ##################
spark = (SparkSession.builder.appName("movie Recommendation with Hadoop").config("spark.sql.shuffle.partitions", "200") # increase shuffle partitions
         .config("spark.executor.memory","4g")
         .config("spark.executor.cores", "4")
         .config("spark.executor.instances","5")
         .getOrCreate())

########### Load the prepped data ##############
# als_data = spark.read.parquet("../datasets/als.parquet")
als_data = spark.read.parquet("hdfs:///user/mahyar/datasets/als.parquet")

# als_data.show(5)

# repartition the data for speed
als_data = als_data.repartition(50) #TODO: Tweakable based on performance
# cache
als_data.cache()

###### fine tuning ##########
# filter out users with less than 5 ratings
user_ratings_count = als_data.groupby("userID").count()
active_users = user_ratings_count.filter(col("count") >=5 )
als_data_filtered = als_data.join(active_users, "userID")
# TODO: drop the counts column
# TODO: make use of the genres column

als_data_filtered.cache()

# filter out movies with less than x ratings TODO: let's see how the top one does alone first
# movie_ratings_count = als_data_filtered.groupBy("movieId").count()
# popular_movies = movie_ratings_count.filter(col("count") >= 5)
# als_data_filtered = als_data_filtered.join(popular_movies, "movieId")

##### Split #########
(training, test) = als_data_filtered.randomSplit([0.8,0.2], seed=42) #TODO: tweakable

##### Train ####
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", nonnegative= False, coldStartStrategy="drop")
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")

param_grid = (ParamGridBuilder().addGrid(als.rank, [10])
              .addGrid(als.maxIter, [20])
              .addGrid(als.regParam, [0.5])
              .build())

cross_validator = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5) # TODO: Tweakable

# checkpoint to improve stability
spark.sparkContext.setCheckpointDir("hdfs:///user/mahyar/checkpoints")

cv_model = cross_validator.fit(training)

best_model = cv_model.bestModel

##### Eval #####
predictions = best_model.transform(test)


rmse = evaluator.evaluate(predictions)
print(f"root main sqaure error = {rmse}")

predictions.show(50)

# Print the best model parameters
print(f"Best Rank: {best_model.rank}")
print(f"Best MaxIter: {best_model._java_obj.parent().getMaxIter()}")
print(f"Best RegParam: {best_model._java_obj.parent().getRegParam()}")

# we got root main square error = 0.8149086762553105. which is not bad but not great! let's start fine-tuning this

spark.stop()