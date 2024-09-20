from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, collect_list, concat_ws
import matplotlib.pyplot as plt

##################### Spark ######################################
spark = SparkSession.builder.appName("Movie Recommendation").getOrCreate()



########################### Loading Data ##############################################
ratings = spark.read.csv("../datasets/ratings.csv", header=True, inferSchema=True)
movies = spark.read.csv("../datasets/movies.csv", header=True, inferSchema=True)
tags = spark.read.csv("../datasets/tags.csv", header=True, inferSchema=True)
genome_scores = spark.read.csv("../datasets/genome-scores.csv", header=True, inferSchema=True)
genome_tags = spark.read.csv("../datasets/genome-tags.csv", header=True, inferSchema=True)

############################ exploring the data a little bit here ###########################
# ratings.show(5)
# ratings.printSchema()
#
# movies.show(20)
# movies.printSchema()
#
# tags.show(20)
# tags.printSchema()
#
#
# let's see what the average rating is for some of the movies
# ratings.groupby("movieId").avg("rating").show(20)
#
# I want to see how many ratings are there
# ratings.groupby("movieID").count().orderBy("count", ascending=False).show(20)
# this is good we have lots of data for each movie to work with!
#
# now let's see how users are rating the movies.
# ratings.groupby("rating").count().orderBy("count", ascending=False).show()
# The top ratings in order of count is as follows: 4.0, 3.0, 5.0, 3.5, 4.5


#################### Preparing data ###################
#
# join the data sets
ratings_with_movies = ratings.join(movies, "movieID")
# ratings_with_movies.show(200) # just making sure all the data is what I want them to be

ratings_movies_tags = ratings_with_movies.join(tags, ["userID", "movieID"], how = "left")
# ratings_movies_tags.show(200)
# I have a suspicion that we lost some data there. Let's check!
# user_1 = ratings_movies_tags.filter(ratings_movies_tags["userID"] == 1)
# user_1.show(200)
# no we didn't it was just the ordering lmao.

# now let's do some clean up.
# I will get rid of the two timestamp columns and rename the tag column to be UDG which stands for
# User Defined Genres. Also let's put No UDG instead of the null values for better readibility.

ratings_movies_tags = ratings_movies_tags.drop("timestamp")
ratings_movies_tags = ratings_movies_tags.withColumnRenamed("tag", "UDG")
ratings_movies_tags = ratings_movies_tags.na.fill({"UDG": "No UDG"})
#ratings_movies_tags.show(200, truncate=False)
#ratings_movies_tags.printSchema()

# Count the number of ratings for each movie
movie_counts = ratings_movies_tags.groupBy("movieId", "title").count().orderBy("count", ascending=False)
# movie_counts.show(100)

# now I want to see if I can get rid of the movies that have less than 20 ratings.
# first I want to check if 20 is a good approximation for this task or if we should even do this!
#
# how many movies have fewer than 20 ratings
# movies_less_than_20_ratings = movie_counts.filter(col("count") < 20)
# num_movies_less_than_20 = movies_less_than_20_ratings.count()
# print(f"Number of movies with less than 20 ratings: {num_movies_less_than_20}")

# I want a histogram to better understand
# movie_counts_pandas = movie_counts.toPandas()
# Plot a histogram
# plt.figure(figsize=(10, 6))
# plt.hist(movie_counts_pandas['count'], bins=50, color='blue', edgecolor='black')
# plt.title('Histogram of Movie Ratings Counts')
# plt.xlabel('Number of Ratings')
# plt.ylabel('Number of Movies')
# plt.grid(True)
# plt.show()
################################ a lot of our data lies between 0 and 1000 so this I will continue with the data as is
# and see how accuarte the MLA will be ################# might come back to this!

###################### final prep ##########################
# ratings_movies_tags = (ratings_movies_tags.groupBy("userId","movieId","rating","title","genres").
#                       agg(concat_ws(",",collect_list("UDG")).alias("all_tags")))

# ratings_movies_tags.show(100)
# okay the UDG has way too many missing values. we get rid of it for now and see how accurate the MLA is. we might come back to this too!

ratings_movies_tags = ratings_movies_tags.drop("UDG")
# ratings_movies_tags.show(100)

als_data = ratings_movies_tags

als_data.write.parquet("../datasets/als.parquet", mode = "overwrite")
print("Data prep finished and saved in als.parquet")