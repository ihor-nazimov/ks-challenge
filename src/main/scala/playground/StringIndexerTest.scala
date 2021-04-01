package playground

import explore_data.ExploreData._
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

object ExploreData extends App {
  val spark = SparkSession.builder()
    .appName("Explore Data")
    .config("spark.master", "local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  val df = spark.read
    .format("csv")
    .option("header", "true")
    .option("sep", ",")
    .load(s"src/main/resources/data/hashed_feature.csv")

//  val df = spark.createDataFrame(
//    Seq((0, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), (1, "b"), (2, "c"), (3, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), (4, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"), (5, "c"))
//  ).toDF("id", "category").drop("id")

  val indexer = new StringIndexer()
    .setInputCol("feature_50")
    .setOutputCol("feature_50_index")

  val indexed = indexer.fit(df).transform(df)
  indexed.show()
  println(indexed.count)
  println(indexed.select("feature_50").distinct().count)
  println(indexed.select("feature_50_index").distinct().count)
  indexed.where(col("feature_50").isNull).show

}