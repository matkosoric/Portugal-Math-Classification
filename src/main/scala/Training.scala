import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{ChiSqSelector, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Row


object Training {

  def main(args: Array[String]): Unit = {

    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local[*]")
      .appName("Predicting final grade in Math for two Portuguese Schools")
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

    loadingDF.printSchema()

    val indexers = loadingDF.select("school", "sex", "address", "famsize", "Pstatus", "Mjob",
    "Fjob", "reason", "guardian",  "schoolsup", "famsup", "paid","activities", "nursery","higher","internet","romantic")
      .columns.map { colName =>
      new StringIndexer().setInputCol(colName).setOutputCol(colName + "_indexed")
    }

    val pipelineIndex = new Pipeline()
      .setStages(indexers)

    val dfWithIndexedStrings = pipelineIndex.fit(loadingDF).transform(loadingDF).withColumnRenamed("G3", "label")
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
//      .drop("G1")   // too correlated
//      .drop("G2")   // too correlated

    val dfDoubles= dfWithIndexedStrings.select(
      dfWithIndexedStrings.col("age").cast("float"),
      dfWithIndexedStrings.col("Medu").cast("float"),
      dfWithIndexedStrings.col("Fedu").cast("float"),
      dfWithIndexedStrings.col("traveltime").cast("float"),
      dfWithIndexedStrings.col("studytime").cast("float"),
      dfWithIndexedStrings.col("failures").cast("float"),
      dfWithIndexedStrings.col("famrel").cast("float"),
      dfWithIndexedStrings.col("freetime").cast("float"),
      dfWithIndexedStrings.col("goout").cast("float"),
      dfWithIndexedStrings.col("Dalc").cast("float"),
      dfWithIndexedStrings.col("Walc").cast("float"),
      dfWithIndexedStrings.col("health").cast("float"),
      dfWithIndexedStrings.col("absences").cast("float"),

      dfWithIndexedStrings.col("school_indexed"),
      dfWithIndexedStrings.col("sex_indexed"),
      dfWithIndexedStrings.col("address_indexed"),
      dfWithIndexedStrings.col("famsize_indexed"),
      dfWithIndexedStrings.col("Pstatus_indexed"),
      dfWithIndexedStrings.col("Mjob_indexed"),
      dfWithIndexedStrings.col("Fjob_indexed"),
      dfWithIndexedStrings.col("reason_indexed"),
      dfWithIndexedStrings.col("guardian_indexed"),
      dfWithIndexedStrings.col("schoolsup_indexed"),
      dfWithIndexedStrings.col("famsup_indexed"),
      dfWithIndexedStrings.col("paid_indexed"),
      dfWithIndexedStrings.col("activities_indexed"),
      dfWithIndexedStrings.col("nursery_indexed"),
      dfWithIndexedStrings.col("higher_indexed"),
      dfWithIndexedStrings.col("internet_indexed"),
      dfWithIndexedStrings.col("romantic_indexed"),
      dfWithIndexedStrings.col("label")

          , dfWithIndexedStrings.col("G1").cast("float"),
          dfWithIndexedStrings.col("G2").cast("float")

    )

    dfDoubles.show()

    val Array(training, test)  = dfDoubles.randomSplit(Array(0.8, 0.2), seed = 989)

    training.cache()
    test.cache()

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        // numeric features
        "age",  "Medu", "Fedu", "traveltime", "studytime", "failures",
        "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences",

        // indexed string features
        "school_indexed", "sex_indexed", "address_indexed", "famsize_indexed",
        "Pstatus_indexed", "Mjob_indexed", "Fjob_indexed", "reason_indexed",
        "guardian_indexed", "schoolsup_indexed", "famsup_indexed", "paid_indexed",
        "activities_indexed", "nursery_indexed", "higher_indexed", "internet_indexed", "romantic_indexed"
            ,"G1", "G2"
      ))
      .setOutputCol("features")

    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithStd(false)
      .setWithMean(true)

    // select the most valuable features
    val selector = new ChiSqSelector()
      .setNumTopFeatures(10)
      .setFeaturesCol("scaledFeatures")
      .setLabelCol("label")
      .setOutputCol("selectedFeatures")

    val logRegModel = new LogisticRegression()
      .setFeaturesCol("selectedFeatures")
      .setLabelCol("label")
      .setFamily("auto")

    //    val pipeline = new Pipeline().setStages(Array(assembler, logRegModel))
    val pipeline = new Pipeline().setStages(Array(assembler, scaler, selector, logRegModel))

    val paramGrid = new ParamGridBuilder()

      // single model
      .addGrid(logRegModel.aggregationDepth, Array(2))
      .addGrid(logRegModel.elasticNetParam, Array(0.001))
      .addGrid(logRegModel.fitIntercept, Array(true))
      .addGrid(logRegModel.maxIter, Array(10))
      .addGrid(logRegModel.regParam, Array(0.01))
      .addGrid(logRegModel.standardization, Array(true))
      .addGrid(logRegModel.threshold, Array(0.001))
      .addGrid(logRegModel.tol, Array(0.000001))

      // full grid search
//      .addGrid(logRegModel.maxIter, Array(5, 7, 10, 15, 20))
//      .addGrid(logRegModel.elasticNetParam, Array(0.0001, 0.001, 0.01, 0.1, 0.5))
//      .addGrid(logRegModel.regParam, Array(0.001, 0.01, 0.1, 1.0))
//      .addGrid(logRegModel.aggregationDepth, Array(2, 5, 10))
//      .addGrid(logRegModel.fitIntercept, Array(true, false))
//      .addGrid(logRegModel.standardization, Array(true, false))
//      .addGrid(logRegModel.threshold, Array(0.001, 0.01, 0.1, 1.0))
//      .addGrid(logRegModel.tol, Array(0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)

    val bestLogRegModel = cv.fit(training).bestModel.asInstanceOf[PipelineModel]

    println("Model hyper-parameters:\n" + bestLogRegModel.asInstanceOf[PipelineModel].stages(3).extractParamMap())

    bestLogRegModel.write.overwrite().save("spark-warehouse")
    val loadedModel = PipelineModel.load("spark-warehouse")

    val resultDFtest = loadedModel.transform(test)
    val resultDFtraining = loadedModel.transform(training)

      val Row(coeff1: Matrix) = Correlation.corr(resultDFtraining, "features").head
      println("Pearson correlation matrix:\n" + coeff1.toString(32,Int.MaxValue))

    resultDFtest.sample(0.2, 53245)
      .select(
        "features",
        "label",
        "prediction"
      ).show(20, false)

    // evaluation
    val evaluatorRMSE = new RegressionEvaluator()
    println("Is larger better: " + evaluatorRMSE.isLargerBetter)
    println(evaluatorRMSE.explainParams())
    val RMSEtraining = evaluatorRMSE.evaluate(resultDFtraining)
    println("Root Mean Squared Error on a training set = " + RMSEtraining)
    val RMSEtest = evaluatorRMSE.evaluate(resultDFtest)
    println("Root Mean Squared Error on a test set = " + RMSEtest)

  }
}
