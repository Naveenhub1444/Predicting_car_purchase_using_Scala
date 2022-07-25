/*
Topic: Supervised Learning
There is a lack of publicly available datasets on financial services and especially in the emerging mobile money
transactions domain. Financial datasets are important for researchers to detect fraud in the system. Let us assume that
Paysim is a financial mobile money simulator designed to detect fraud.

Tasks: Now, with the data pipeline ready, you are required to develop the model and predict fraud using spark streaming.
Find out money transfer only which is greater than 200,000

Data Source: https://www.kaggle.com/ntnu-testimon/paysim1
 */

/*
root
 |-- step: integer (nullable = true)              - Unit of time 1 means 1 hour
 |-- type: string (nullable = true)               - CASH_IN, CASH_OUT, DEBIT, PAYMENT, and TRANSFER
 |-- amount: double (nullable = true)             - amount of the transaction in local currency
 |-- nameOrig: string (nullable = true)           - customer who started the transaction
 |-- oldbalanceOrg: double (nullable = true)      - initial balance before the transaction
 |-- newbalanceOrig: double (nullable = true)     - customer's balance after the transaction
 |-- nameDest: string (nullable = true)           - recipient ID of the transaction.
 |-- oldbalanceDest: double (nullable = true)     - initial recipient balance before the transaction
 |-- newbalanceDest: double (nullable = true)     - recipient's balance after the transaction
 |-- isFraud: integer (nullable = true)           - identifies a fraudulent transaction (1) and non-fraudulent (0)
 |-- isFlaggedFraud: integer (nullable = true)    - flags illegal attempts to transfer more than 200,000 in a single transaction.
 */




import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object suv_car_rand_reg {
  def main(args: Array[String]): Unit = {
    val spark= SparkSession
      .builder()
      .master("local[*]")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    val dataspath="D:\\data_files\\suv_data.csv"
    val carDF =spark.read.option("header","true")
      .option ("inferSchema",true)
      .option("mode","DROPMALFORMED")
      .csv(dataspath)
    carDF.printSchema()

/*
root
 |-- User_ID: integer (nullable = true)
 |-- Gender: string (nullable = true)
 |-- Age: integer (nullable = true)
 |-- EstimatedSalary: integer (nullable = true)
 |-- Purchased: integer (nullable = true)
 */

    println("carDF dataframe")
    carDF.show(5,false)

/*
+--------+------+---+---------------+---------+
|User_ID |Gender|Age|EstimatedSalary|Purchased|
+--------+------+---+---------------+---------+
|15624510|Male  |19 |19000          |0        |
|15810944|Male  |35 |20000          |0        |
|15668575|Female|26 |43000          |0        |
|15603246|Female|27 |57000          |0        |
|15804002|Male  |19 |76000          |0        |
+--------+------+---+---------------+---------+
 */
    println("gradeDf describe")
    carDF.describe().show()
/*
+-------+-----------------+------+------------------+----------------+------------------+
|summary|          User_ID|Gender|               Age| EstimatedSalary|         Purchased|
+-------+-----------------+------+------------------+----------------+------------------+
|  count|              400|   400|               400|             400|               400|
|   mean|  1.56915397575E7|  null|            37.655|         69742.5|            0.3575|
| stddev|71658.32158119006|  null|10.482876597307927|34096.9602824248|0.4798639635968691|
|    min|         15566689|Female|                18|           15000|                 0|
|    max|         15815236|  Male|                60|          150000|                 1|
+-------+-----------------+------+------------------+----------------+------------------+
 */
  val indexer = new StringIndexer()
  .setInputCol("Gender")
  .setOutputCol("Gender_label")

    val gen_Label = indexer
      .setHandleInvalid("keep")
      .fit(carDF)
      .transform(carDF)
    println("grade_label printSchema")
    gen_Label.printSchema()
/*
root
 |-- User_ID: integer (nullable = true)
 |-- Gender: string (nullable = true)
 |-- Age: integer (nullable = true)
 |-- EstimatedSalary: integer (nullable = true)
 |-- Purchased: integer (nullable = true)
 |-- Gender_label: double (nullable = false)
 */
    gen_Label.show(5,false)
/*
+--------+------+---+---------------+---------+------------+
|User_ID |Gender|Age|EstimatedSalary|Purchased|Gender_label|
+--------+------+---+---------------+---------+------------+
|15624510|Male  |19 |19000          |0        |1.0         |
|15810944|Male  |35 |20000          |0        |1.0         |
|15668575|Female|26 |43000          |0        |0.0         |
|15603246|Female|27 |57000          |0        |0.0         |
|15804002|Male  |19 |76000          |0        |1.0         |
+--------+------+---+---------------+---------+------------+
 */
  val cols = Array("Age","EstimatedSalary","Gender_label")
    val my_Assembler = new VectorAssembler()
      .setInputCols(cols)
      .setOutputCol("features")
    val car_Feature = my_Assembler.transform(gen_Label)
    println("car_Feature Schema")
    car_Feature.printSchema()
/*
root
 |-- User_ID: integer (nullable = true)
 |-- Gender: string (nullable = true)
 |-- Age: integer (nullable = true)
 |-- EstimatedSalary: integer (nullable = true)
 |-- Purchased: integer (nullable = true)
 |-- Gender_label: double (nullable = false)
 |-- features: vector (nullable = true)
 */
    println("car_Feature data frame")
    car_Feature.show(5,false)

/*
  +--------+------+---+---------------+---------+------------+------------------+
|User_ID |Gender|Age|EstimatedSalary|Purchased|Gender_label|features          |
+--------+------+---+---------------+---------+------------+------------------+
|15624510|Male  |19 |19000          |0        |1.0         |[19.0,19000.0,1.0]|
|15810944|Male  |35 |20000          |0        |1.0         |[35.0,20000.0,1.0]|
|15668575|Female|26 |43000          |0        |0.0         |[26.0,43000.0,0.0]|
|15603246|Female|27 |57000          |0        |0.0         |[27.0,57000.0,0.0]|
|15804002|Male  |19 |76000          |0        |1.0         |[19.0,76000.0,1.0]|
+--------+------+---+---------------+---------+------------+------------------+

 */
val indexer_2 = new StringIndexer()
  .setInputCol("Purchased")
  .setOutputCol("label")

    val purch_Label = indexer_2
      .setHandleInvalid("keep")
      .fit(car_Feature)
      .transform(car_Feature)
    println("purch_label printSchema")
    purch_Label.printSchema()
/*
root
 |-- User_ID: integer (nullable = true)
 |-- Gender: string (nullable = true)
 |-- Age: integer (nullable = true)
 |-- EstimatedSalary: integer (nullable = true)
 |-- Purchased: integer (nullable = true)
 |-- Gender_label: double (nullable = false)
 |-- features: vector (nullable = true)
 |-- label: double (nullable = false)

 */
    purch_Label.createOrReplaceTempView("purch_view")

    println("grade_Label dataframe")
    purch_Label.show(10,false)
/*
   +--------+------+---+---------------+---------+------------+-------------------+-----+
|User_ID |Gender|Age|EstimatedSalary|Purchased|Gender_label|features           |label|
+--------+------+---+---------------+---------+------------+-------------------+-----+
|15624510|Male  |19 |19000          |0        |1.0         |[19.0,19000.0,1.0] |0.0  |
|15810944|Male  |35 |20000          |0        |1.0         |[35.0,20000.0,1.0] |0.0  |
|15668575|Female|26 |43000          |0        |0.0         |[26.0,43000.0,0.0] |0.0  |
|15603246|Female|27 |57000          |0        |0.0         |[27.0,57000.0,0.0] |0.0  |
|15804002|Male  |19 |76000          |0        |1.0         |[19.0,76000.0,1.0] |0.0  |
|15728773|Male  |27 |58000          |0        |1.0         |[27.0,58000.0,1.0] |0.0  |
|15598044|Female|27 |84000          |0        |0.0         |[27.0,84000.0,0.0] |0.0  |
|15694829|Female|32 |150000         |1        |0.0         |[32.0,150000.0,0.0]|1.0  |
|15600575|Male  |25 |33000          |0        |1.0         |[25.0,33000.0,1.0] |0.0  |
|15727311|Female|35 |65000          |0        |0.0         |[35.0,65000.0,0.0] |0.0  |
+--------+------+---+---------------+---------+------------+-------------------+-----+
 */

    purch_Label
val seed = 5043
    val Array(trainData,testData)=purch_Label.randomSplit(Array(0.7,0.3),seed)

    val classifier = new RandomForestClassifier()
      .setImpurity("gini")
      .setMaxDepth(10)
      .setNumTrees(20)
      .setFeatureSubsetStrategy("auto")
      .setSeed(5043)

    val model = classifier.fit(trainData)

    val prediction_df = model.transform(testData)
    println("predictionDF dataframe")
    prediction_df.show(10,false)
/*
+--------+------+---+---------------+---------+------------+-------------------+-----+---------------------------------------------+----------------------------------------------+----------+
|User_ID |Gender|Age|EstimatedSalary|Purchased|Gender_label|features           |label|rawPrediction                                |probability                                   |prediction|
+--------+------+---+---------------+---------+------------+-------------------+-----+---------------------------------------------+----------------------------------------------+----------+
|15571059|Female|33 |41000          |0        |0.0         |[33.0,41000.0,0.0] |0.0  |[19.992537313432834,0.007462686567164179,0.0]|[0.9996268656716418,3.73134328358209E-4,0.0]  |0.0       |
|15575247|Female|48 |131000         |1        |0.0         |[48.0,131000.0,0.0]|1.0  |[3.1409090909090907,16.85909090909091,0.0]   |[0.15704545454545454,0.8429545454545455,0.0]  |1.0       |
|15577514|Male  |43 |129000         |1        |1.0         |[43.0,129000.0,1.0]|1.0  |[7.3,12.7,0.0]                               |[0.365,0.635,0.0]                             |1.0       |
|15578006|Male  |23 |63000          |0        |1.0         |[23.0,63000.0,1.0] |0.0  |[19.992537313432834,0.007462686567164179,0.0]|[0.9996268656716418,3.73134328358209E-4,0.0]  |0.0       |
|15578738|Female|18 |86000          |0        |0.0         |[18.0,86000.0,0.0] |0.0  |[19.973669388904533,0.026330611095466066,0.0]|[0.9986834694452267,0.0013165305547733033,0.0]|0.0       |
|15579212|Male  |39 |77000          |0        |1.0         |[39.0,77000.0,1.0] |0.0  |[13.59142088600062,6.408579113999382,0.0]    |[0.679571044300031,0.3204289556999691,0.0]    |0.0       |
|15581198|Male  |31 |74000          |0        |1.0         |[31.0,74000.0,1.0] |0.0  |[19.973669388904533,0.026330611095466066,0.0]|[0.9986834694452267,0.0013165305547733033,0.0]|0.0       |
|15584114|Male  |24 |23000          |0        |1.0         |[24.0,23000.0,1.0] |0.0  |[19.992537313432834,0.007462686567164179,0.0]|[0.9996268656716418,3.73134328358209E-4,0.0]  |0.0       |
|15584545|Female|32 |86000          |0        |0.0         |[32.0,86000.0,0.0] |0.0  |[19.973669388904533,0.026330611095466066,0.0]|[0.9986834694452267,0.0013165305547733033,0.0]|0.0       |
|15587419|Female|42 |90000          |1        |0.0         |[42.0,90000.0,0.0] |1.0  |[8.452380952380953,11.54761904761905,0.0]    |[0.4226190476190476,0.5773809523809524,0.0]   |1.0       |
+--------+------+---+---------------+---------+------------+-------------------+-----+---------------------------------------------+----------------------------------------------+----------+
 */

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
    val accuracy = evaluator.evaluate(prediction_df)
    println("accuracy % = " + accuracy * 100 )
/*
accuracy % = 88.10606060606061
 */
val dataset="D:\\example\\car_suv.cs"
    val df1 =spark.read.option("header","true")
      .option ("inferSchema",true)
      .option("mode","DROPMALFORMED")
      .csv(dataset)
    println("df1 details...")
    df1.show(10,false)
/*
+--------+------+---+---------------+
|User_ID |Gender|Age|EstimatedSalary|
+--------+------+---+---------------+
|15624510|Male  |19 |19000          |
|15810944|Male  |35 |20000          |
|15668575|Female|26 |43000          |
|15603246|Female|27 |57000          |
|15804002|Male  |19 |76000          |
|15728773|Male  |27 |58000          |
|15598044|Female|27 |84000          |
|15694829|Female|32 |150000         |
|15600575|Male  |25 |33000          |
|15727311|Female|35 |65000          |
+--------+------+---+---------------+
 */
val gen_Label_2 = indexer
  .setHandleInvalid("keep")
  .fit(df1)
  .transform(df1)
    println("gen_label_2 printSchema")
    gen_Label_2.printSchema()
/*
root
 |-- User_ID: integer (nullable = true)
 |-- Gender: string (nullable = true)
 |-- Age: integer (nullable = true)
 |-- EstimatedSalary: integer (nullable = true)
 |-- Gender_label: double (nullable = false)
 */
    gen_Label_2.createOrReplaceTempView("gen_view")

    println("gen_Label dataframe")
    gen_Label_2.show(10,false)

/*
+--------+------+---+---------------+------------+
|User_ID |Gender|Age|EstimatedSalary|Gender_label|
+--------+------+---+---------------+------------+
|15624510|Male  |19 |19000          |1.0         |
|15810944|Male  |35 |20000          |1.0         |
|15668575|Female|26 |43000          |0.0         |
|15603246|Female|27 |57000          |0.0         |
|15804002|Male  |19 |76000          |1.0         |
|15728773|Male  |27 |58000          |1.0         |
|15598044|Female|27 |84000          |0.0         |
|15694829|Female|32 |150000         |0.0         |
|15600575|Male  |25 |33000          |1.0         |
|15727311|Female|35 |65000          |0.0         |
+--------+------+---+---------------+------------+
 */

    val df2 = my_Assembler.transform(gen_Label_2)
    println("df2 details...")
    df2.show(5,false)
/*
+--------+------+---+---------------+------------+------------------+
|User_ID |Gender|Age|EstimatedSalary|Gender_label|features          |
+--------+------+---+---------------+------------+------------------+
|15624510|Male  |19 |19000          |1.0         |[19.0,19000.0,1.0]|
|15810944|Male  |35 |20000          |1.0         |[35.0,20000.0,1.0]|
|15668575|Female|26 |43000          |0.0         |[26.0,43000.0,0.0]|
|15603246|Female|27 |57000          |0.0         |[27.0,57000.0,0.0]|
|15804002|Male  |19 |76000          |1.0         |[19.0,76000.0,1.0]|
+--------+------+---+---------------+------------+------------------+
 */
    println("df3 details...")
    val df3 = model.transform(df2)
    df3.show(10,false)
/*
+--------+------+---+---------------+------------+-------------------+---------------------------------------------+----------------------------------------------+----------+
|User_ID |Gender|Age|EstimatedSalary|Gender_label|features           |rawPrediction                                |probability                                   |prediction|
+--------+------+---+---------------+------------+-------------------+---------------------------------------------+----------------------------------------------+----------+
|15624510|Male  |19 |19000          |1.0         |[19.0,19000.0,1.0] |[19.992537313432834,0.007462686567164179,0.0]|[0.9996268656716418,3.73134328358209E-4,0.0]  |0.0       |
|15810944|Male  |35 |20000          |1.0         |[35.0,20000.0,1.0] |[19.992537313432834,0.007462686567164179,0.0]|[0.9996268656716418,3.73134328358209E-4,0.0]  |0.0       |
|15668575|Female|26 |43000          |0.0         |[26.0,43000.0,0.0] |[19.992537313432834,0.007462686567164179,0.0]|[0.9996268656716418,3.73134328358209E-4,0.0]  |0.0       |
|15603246|Female|27 |57000          |0.0         |[27.0,57000.0,0.0] |[19.992537313432834,0.007462686567164179,0.0]|[0.9996268656716418,3.73134328358209E-4,0.0]  |0.0       |
|15804002|Male  |19 |76000          |1.0         |[19.0,76000.0,1.0] |[19.973669388904533,0.026330611095466066,0.0]|[0.9986834694452267,0.0013165305547733033,0.0]|0.0       |
|15728773|Male  |27 |58000          |1.0         |[27.0,58000.0,1.0] |[19.992537313432834,0.007462686567164179,0.0]|[0.9996268656716418,3.73134328358209E-4,0.0]  |0.0       |
|15598044|Female|27 |84000          |0.0         |[27.0,84000.0,0.0] |[19.973669388904533,0.026330611095466066,0.0]|[0.9986834694452267,0.0013165305547733033,0.0]|0.0       |
|15694829|Female|32 |150000         |0.0         |[32.0,150000.0,0.0]|[0.09090909090909091,19.909090909090907,0.0] |[0.004545454545454546,0.9954545454545455,0.0] |1.0       |
|15600575|Male  |25 |33000          |1.0         |[25.0,33000.0,1.0] |[19.992537313432834,0.007462686567164179,0.0]|[0.9996268656716418,3.73134328358209E-4,0.0]  |0.0       |
|15727311|Female|35 |65000          |0.0         |[35.0,65000.0,0.0] |[19.992537313432834,0.007462686567164179,0.0]|[0.9996268656716418,3.73134328358209E-4,0.0]  |0.0       |
+--------+------+---+---------------+------------+-------------------+---------------------------------------------+----------------------------------------------+----------+
 */
    df3.createOrReplaceTempView("my_View")
    val df4 = spark.sql("select User_ID,Gender,Age,EstimatedSalary,prediction from my_view")
    println("df4 details")
    df4.show(10,false)
/*
+--------+------+---+---------------+----------+
|User_ID |Gender|Age|EstimatedSalary|prediction|
+--------+------+---+---------------+----------+
|15624510|Male  |19 |19000          |0.0       |
|15810944|Male  |35 |20000          |0.0       |
|15668575|Female|26 |43000          |0.0       |
|15603246|Female|27 |57000          |0.0       |
|15804002|Male  |19 |76000          |0.0       |
|15728773|Male  |27 |58000          |0.0       |
|15598044|Female|27 |84000          |0.0       |
|15694829|Female|32 |150000         |1.0       |
|15600575|Male  |25 |33000          |0.0       |
|15727311|Female|35 |65000          |0.0       |
+--------+------+---+---------------+----------+
 */
    df4.createOrReplaceTempView("df4_view")
    println("df4 describe")
    df4.describe().show()
/*
+-------+-----------------+------+------------------+----------------+------------------+
|summary|          User_ID|Gender|               Age| EstimatedSalary|        prediction|
+-------+-----------------+------+------------------+----------------+------------------+
|  count|              400|   400|               400|             400|               400|
|   mean|  1.56915397575E7|  null|            37.655|         69742.5|             0.355|
| stddev|71658.32158119006|  null|10.482876597307927|34096.9602824248|0.4791125882091297|
|    min|         15566689|Female|                18|           15000|               0.0|
|    max|         15815236|  Male|                60|          150000|               1.0|
+-------+-----------------+------+------------------+----------------+------------------+
 */
val final_out = spark.sql("SELECT purch_view.User_ID,purch_view.Gender,purch_view.Age," +
  "purch_view.EstimatedSalary,purch_view.label,df4_view.prediction FROM purch_view  " +
  "JOIN df4_view ON purch_view.User_ID = df4_view.User_ID GROUP BY purch_view.User_ID,purch_view.Gender,purch_view.Age," +
  "purch_view.EstimatedSalary,purch_view.label,df4_view.prediction")

    println("final_out describe")
    final_out.describe().show()
/*
+-------+-----------------+------+-----------------+------------------+-----------------+------------------+
|summary|          User_ID|Gender|              Age|   EstimatedSalary|            label|        prediction|
+-------+-----------------+------+-----------------+------------------+-----------------+------------------+
|  count|              400|   400|              400|               400|              400|               400|
|   mean|  1.56915397575E7|  null|           37.655|           69742.5|           0.3575|             0.355|
| stddev|71658.32158119035|  null|10.48287659730791|34096.960282424785|0.479863963596869|0.4791125882091298|
|    min|         15566689|Female|               18|             15000|              0.0|               0.0|
|    max|         15815236|  Male|               60|            150000|              1.0|               1.0|
+-------+-----------------+------+-----------------+------------------+-----------------+------------------+
 */
    println("final_out dataframe")
    final_out.show(25,false)
/*
+--------+------+---+---------------+-----+----------+
|User_ID |Gender|Age|EstimatedSalary|label|prediction|
+--------+------+---+---------------+-----+----------+
|15705113|Male  |46 |23000          |1.0  |1.0       |
|15798659|Female|30 |62000          |0.0  |0.0       |
|15753861|Female|49 |141000         |1.0  |1.0       |
|15594041|Female|49 |36000          |1.0  |1.0       |
|15768816|Male  |26 |81000          |0.0  |0.0       |
|15766289|Male  |27 |88000          |0.0  |0.0       |
|15570932|Male  |34 |115000         |0.0  |0.0       |
|15775335|Male  |56 |60000          |1.0  |1.0       |
|15682268|Male  |35 |50000          |0.0  |0.0       |
|15736228|Male  |38 |59000          |0.0  |0.0       |
|15769596|Female|56 |104000         |1.0  |1.0       |
|15596522|Male  |49 |89000          |1.0  |1.0       |
|15769902|Male  |42 |65000          |0.0  |0.0       |
|15749130|Female|41 |30000          |0.0  |0.0       |
|15791373|Male  |60 |42000          |1.0  |1.0       |
|15589715|Female|48 |119000         |1.0  |1.0       |
|15709476|Male  |20 |49000          |0.0  |0.0       |
|15738448|Female|30 |79000          |0.0  |0.0       |
|15727467|Male  |57 |74000          |1.0  |1.0       |
|15654574|Female|23 |82000          |0.0  |0.0       |
|15679651|Female|26 |15000          |0.0  |0.0       |
|15621083|Female|48 |29000          |1.0  |1.0       |
|15672821|Female|55 |125000         |1.0  |1.0       |
|15742204|Male  |45 |32000          |1.0  |1.0       |
|15671655|Female|35 |23000          |0.0  |0.0       |
+--------+------+---+---------------+-----+----------+
 */

    println("final_out,who can buy")
    final_out.filter( "label = 1").show(10,false)
/*
+--------+------+---+---------------+-----+----------+
|User_ID |Gender|Age|EstimatedSalary|label|prediction|
+--------+------+---+---------------+-----+----------+
|15705113|Male  |46 |23000          |1.0  |1.0       |
|15753861|Female|49 |141000         |1.0  |1.0       |
|15594041|Female|49 |36000          |1.0  |1.0       |
|15775335|Male  |56 |60000          |1.0  |1.0       |
|15769596|Female|56 |104000         |1.0  |1.0       |
|15596522|Male  |49 |89000          |1.0  |1.0       |
|15791373|Male  |60 |42000          |1.0  |1.0       |
|15589715|Female|48 |119000         |1.0  |1.0       |
|15727467|Male  |57 |74000          |1.0  |1.0       |
|15621083|Female|48 |29000          |1.0  |1.0       |
+--------+------+---+---------------+-----+----------+
 */
    println("final_out,who can't buy")
    final_out.filter( "label = 0").show(10,false)
/*
+--------+------+---+---------------+-----+----------+
|User_ID |Gender|Age|EstimatedSalary|label|prediction|
+--------+------+---+---------------+-----+----------+
|15798659|Female|30 |62000          |0.0  |0.0       |
|15768816|Male  |26 |81000          |0.0  |0.0       |
|15766289|Male  |27 |88000          |0.0  |0.0       |
|15570932|Male  |34 |115000         |0.0  |0.0       |
|15682268|Male  |35 |50000          |0.0  |0.0       |
|15736228|Male  |38 |59000          |0.0  |0.0       |
|15769902|Male  |42 |65000          |0.0  |0.0       |
|15749130|Female|41 |30000          |0.0  |0.0       |
|15709476|Male  |20 |49000          |0.0  |0.0       |
|15738448|Female|30 |79000          |0.0  |0.0       |
+--------+------+---+---------------+-----+----------+
 */




  }

}
