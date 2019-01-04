import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.classification.LogisticRegression

object Evaluating {


  def main(args: Array[String]): Unit = {

    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local[*]")
      .appName("Using trained model")
      .getOrCreate;

    spark.conf.set("spark.debug.maxToStringFields", 50L)
    spark.sparkContext.setLogLevel("ERROR")

    val loadingDF = spark.read
      .format("csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .option("delimiter", ";")
      .option("inferSchema", "true")
      .load("src/main/resources/student-mat.csv")

    val loadedModel: PipelineModel = PipelineModel.load("spark-warehouse")

    println(loadedModel.extractParamMap())

  }

}
