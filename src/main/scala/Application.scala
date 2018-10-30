import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StandardScaler, VectorAssembler}

object Application {

  def main(args: Array[String]): Unit = {

    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Predicting final grade in Math for two Portugal Schools")
      .getOrCreate;

    spark.conf.set("spark.debug.maxToStringFields", 50L)
    spark.sparkContext.setLogLevel("ERROR")

    val loadingDF = spark.read
      .format("csv")
      .option("header", "true")
      .option("mode", "DROPMALFORMED")
      .option("delimiter", ";")
      .option("inferSchema","true")
      .load("src/main/resources/student-mat.csv")

    // removing columns with string type
    val loadingDF2 = loadingDF.withColumnRenamed("G3", "label")
      .drop("school")
      .drop("sex")
      .drop("address")
      .drop("famsize")
      .drop("Pstatus")
      .drop("Mjob")
      .drop("Fjob")
      .drop("reason")
      .drop("guardian")
      .drop("schoolsup")
      .drop("famsup")
      .drop("paid")
      .drop("activities")
      .drop("nursery")
      .drop("higher")
      .drop("internet")
      .drop("romantic")

    val Array(training, test)  = loadingDF2.randomSplit(Array(0.8, 0.2), seed = 50)

    //checking for zeroes
    //    println ("-------------------" + training.filter(training("G2").equalTo(0.0)).count())

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "age",  "Medu", "Fedu",
        "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health",
        "absences",
        "G1", "G2"
      ))
      .setOutputCol("features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(false)
      .setWithMean(true)

    // could be used for reducing the number of columns
    val selector = new ChiSqSelector()
      .setNumTopFeatures(20)
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")
      .setOutputCol("selectedFeatures")

    val lr = new LogisticRegression()
      .setMaxIter(30)
      .setRegParam(0.001)
      .setElasticNetParam(0.001)
      .setFeaturesCol("selectedFeatures")
      .setLabelCol("label")

    //creating pipeline
    val pipeline = new Pipeline().setStages(Array(assembler,  scaler, selector, lr))
    val lrModel = pipeline.fit(training)

    lrModel.write.overwrite().save("spark-warehouse")
    val loadedModel = PipelineModel.load("spark-warehouse")

    val resultDFtest = loadedModel.transform(test)
    val resultDFtraining = loadedModel.transform(training)

    resultDFtest.sample(0.3, 54)
      .select(
        "features",
        "label",
        "prediction"
        //        "scaledFeatures",
        //        "selectedFeatures",
        //        "rawPrediction",
        //        "probability",
      )
      .show(20
        , false
      )

    // evaluation
    val evaluatorRMSE = new RegressionEvaluator
    println("Is larger better: " + evaluatorRMSE.isLargerBetter)
    println(evaluatorRMSE.explainParams())
    val RMSEtraining = evaluatorRMSE.evaluate(resultDFtraining)
    println("Root Mean Squared Error training= " + RMSEtraining)
    val RMSEtest = evaluatorRMSE.evaluate(resultDFtest)
    println("Root Mean Squared Error test = " + RMSEtest)

  }
}
