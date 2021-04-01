package explore_data

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.StringIndexer

object ExploreData extends App {
  val spark = SparkSession.builder()
    .appName("Explore Data")
    .config("spark.master", "local")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  def readCsv(tableName: String) =
    spark.read
      .format("csv")
      .option("header", "true")
      .option("sep", ",")
      .load(s"src/main/resources/data/$tableName.csv")

  def printTableReport(df: DataFrame) = {
    df.printSchema()
    println(s"Total rows count: ${df.count}")
    df.columns.foreach( colName =>
      println(s"Unique $colName count: ${df.select(colName).distinct().count}")
    )
  }

  //  input dataset (hashed_feature.csv)
  //  |-- id: string (nullable = true) - mobile subscriber id
  //  |-- feature_50: string (nullable = true) - hashed feature (string, length 40)
  val hashedFeatureDF = readCsv("hashed_feature")
  println("*** hashed_feature.csv report ***")
  printTableReport(hashedFeatureDF)

  //Hash-string to index
  val indexer = new StringIndexer()
    .setInputCol("feature_50")
    .setOutputCol("feature_50_index")

  val indexedFeatureDF = indexer.fit(hashedFeatureDF).transform(hashedFeatureDF)
  indexedFeatureDF.show(100)
  printTableReport(indexedFeatureDF)


  //target/train data (train.csv)
  //  |-- id: string (nullable = true) - mobile subscriber id
  //  |-- target: string (nullable = true) - target value
  val trainDF = readCsv("train")
  println("*** train.csv report ***")
  printTableReport(trainDF)

  //join train data
  val targetedHashFeatureDF = hashedFeatureDF
    .join(trainDF, hashedFeatureDF.col("id") === trainDF.col("id"), "left_outer")

//  targetedHashFeatureDF.show(100)



}
