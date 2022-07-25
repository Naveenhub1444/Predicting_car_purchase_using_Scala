import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

object suv_car_logi_regr {
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
 |-- User ID: integer (nullable = true)
 |-- Gender: string (nullable = true)
 |-- Age: integer (nullable = true)
 |-- EstimatedSalary: integer (nullable = true)
 |-- Purchased: integer (nullable = true)
 */

    println("carDF dataframe")
    carDF.show(5,false)

/*
 +--------+------+---+---------------+---------+
|User ID |Gender|Age|EstimatedSalary|Purchased|
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
|summary|          User ID|Gender|               Age| EstimatedSalary|         Purchased|
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
 |-- User ID: integer (nullable = true)
 |-- Gender: string (nullable = true)
 |-- Age: integer (nullable = true)
 |-- EstimatedSalary: integer (nullable = true)
 |-- Purchased: integer (nullable = true)
 |-- Gender_label: double (nullable = false)
 */
    gen_Label.show(5,false)

/*
+--------+------+---+---------------+---------+------------+
|User ID |Gender|Age|EstimatedSalary|Purchased|Gender_label|
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
 |-- User ID: integer (nullable = true)
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
|User ID |Gender|Age|EstimatedSalary|Purchased|Gender_label|features          |
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
|-- User ID: integer (nullable = true)
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
|User ID |Gender|Age|EstimatedSalary|Purchased|Gender_label|features           |label|
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

    val seed = 5043
    val Array(trainData,testData)=purch_Label.randomSplit(Array(0.7,0.3),seed)
    val logisticRegression=new LogisticRegression()
      .setMaxIter(100)
      .setRegParam(0.02)
      .setElasticNetParam(0.8)
    val logisticRegression_model=logisticRegression.fit(trainData)

    val predictionDf=logisticRegression_model.transform(testData)
    println("predictionDF dataframe")
    predictionDf.show(5,false)
/*
+--------+------+---+---------------+---------+------------+-------------------+-----+----------------------------------------------------------+----------------------------------------------------------------+----------+
|User ID |Gender|Age|EstimatedSalary|Purchased|Gender_label|features           |label|rawPrediction                                             |probability                                                     |prediction|
+--------+------+---+---------------+---------+------------+-------------------+-----+----------------------------------------------------------+----------------------------------------------------------------+----------+
|15571059|Female|33 |41000          |0        |0.0         |[33.0,41000.0,0.0] |0.0  |[9.446527272981552,6.707813645579787,-16.157673195539072] |[0.9392727641547403,0.06072723583813071,7.129002259709574E-12]  |0.0       |
|15575247|Female|48 |131000         |1        |0.0         |[48.0,131000.0,0.0]|1.0  |[6.702780125388089,9.420992219390602,-16.157673195539072] |[0.061907217099410355,0.9380927828932856,7.304198222315238E-12] |1.0       |
|15577514|Male  |43 |129000         |1        |1.0         |[43.0,129000.0,1.0]|1.0  |[7.145153010343791,8.973371228267988,-16.157673195539072] |[0.13845067096461589,0.8615493290248886,1.0495574865259026E-11] |1.0       |
|15578006|Male  |23 |63000          |0        |1.0         |[23.0,63000.0,1.0] |0.0  |[10.002171501633685,6.13512609899457,-16.157673195539072] |[0.9795085933969667,0.020491406598768264,4.2651278215116335E-12]|0.0       |
|15578738|Female|18 |86000          |0        |0.0         |[18.0,86000.0,0.0] |0.0  |[10.186996823488204,5.942950827954812,-16.157673195539072]|[0.9858535763633106,0.01414642363312108,3.568351265846627E-12]  |0.0       |
+--------+------+---+---------------+---------+------------+-------------------+-----+----------------------------------------------------------+----------------------------------------------------------------+----------+
 */
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
    val accuracy = evaluator.evaluate(predictionDf)
    println("accuracy % = " + accuracy * 100 )
/*
    accuracy % = 85.19810544825435
 */
val dataset="D:\\example\\car_suv.csv"
    val df1 =spark.read.option("header","true")
      .option ("inferSchema",true)
      .option("mode","DROPMALFORMED")
      .csv(dataset)
    println("df1 details...")
    df1.show(10,false)
/*
+--------+------+---+---------------+
|User ID |Gender|Age|EstimatedSalary|
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
 |-- User ID: integer (nullable = true)
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
|User ID |Gender|Age|EstimatedSalary|Gender_label|
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
+--------+------+---+---------------+------------
 */
val df2 = my_Assembler.transform(gen_Label_2)
    println("df2 details...")
    df2.show(5,false)
/*
+--------+------+---+---------------+------------+------------------+
|User ID |Gender|Age|EstimatedSalary|Gender_label|features          |
+--------+------+---+---------------+------------+------------------+
|15624510|Male  |19 |19000          |1.0         |[19.0,19000.0,1.0]|
|15810944|Male  |35 |20000          |1.0         |[35.0,20000.0,1.0]|
|15668575|Female|26 |43000          |0.0         |[26.0,43000.0,0.0]|
|15603246|Female|27 |57000          |0.0         |[27.0,57000.0,0.0]|
|15804002|Male  |19 |76000          |1.0         |[19.0,76000.0,1.0]|
+--------+------+---+---------------+------------+------------------+
 */
    println("df3 details...")
    val df3 = logisticRegression_model.transform(df2)
    df3.show(10,false)
/*
+--------+------+---+---------------+------------+-------------------+-----------------------------------------------------------+-----------------------------------------------------------------+----------+
|User ID |Gender|Age|EstimatedSalary|Gender_label|features           |rawPrediction                                              |probability                                                      |prediction|
+--------+------+---+---------------+------------+-------------------+-----------------------------------------------------------+-----------------------------------------------------------------+----------+
|15624510|Male  |19 |19000          |1.0         |[19.0,19000.0,1.0] |[11.030619273893691,5.122972441358229,-16.157673195539072] |[0.9972887939873097,0.0027112060111374797,1.5527285815922155E-12]|0.0       |
|15810944|Male  |35 |20000          |1.0         |[35.0,20000.0,1.0] |[9.51366951441064,6.6459925731441745,-16.157673195539072]  |[0.9462252662191145,0.053774733774170254,6.715402661558917E-12]  |0.0       |
|15668575|Female|26 |43000          |0.0         |[26.0,43000.0,0.0] |[10.074022502807985,6.0766702612990136,-16.157673195539072]|[0.9819669637186359,0.01803303627738465,3.979387068525077E-12]   |0.0       |
|15603246|Female|27 |57000          |0.0         |[27.0,57000.0,0.0] |[9.772393279807721,6.373004581401117,-16.157673195539072]  |[0.9676854251499828,0.03231457484471502,5.302118311077184E-12]   |0.0       |
|15804002|Male  |19 |76000          |1.0         |[19.0,76000.0,1.0] |[10.184790955123704,5.945594649525583,-16.157673195539072] |[0.9857857815132435,0.0142142184831806,3.5759853378652076E-12]   |0.0       |
|15728773|Male  |27 |58000          |1.0         |[27.0,58000.0,1.0] |[9.700839301649632,6.440113296978275,-16.157673195539072]  |[0.9630566295824375,0.0369433704118943,5.66816569721968E-12]     |0.0       |
|15598044|Female|27 |84000          |0.0         |[27.0,84000.0,0.0] |[9.37173776039036,6.7626677326382865,-16.157673195539072]  |[0.9314430349385019,0.06855696505387976,7.61857934817815E-12]    |0.0       |
|15694829|Female|32 |150000         |0.0         |[32.0,150000.0,0.0]|[7.922948018636056,8.186611458891448,-16.157673195539072]  |[0.43446336763087434,0.5655366323539944,1.5131160922113367E-11]  |1.0       |
|15600575|Male  |25 |33000          |1.0         |[25.0,33000.0,1.0] |[10.259580467714896,5.890740562467085,-16.157673195539072] |[0.9874924934237064,0.01250750657296945,3.3240405259543862E-12]  |0.0       |
|15727311|Female|35 |65000          |0.0         |[35.0,65000.0,0.0] |[8.90262520022804,7.2427544115266365,-16.157673195539072]  |[0.8402206572051691,0.15977934278384467,1.0986112615305837E-11]  |0.0       |
+--------+------+---+---------------+------------+-------------------+-----------------------------------------------------------+-----------------------------------------------------------------+----------+
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
+-------+-----------------+------+------------------+----------------+-------------------+
|summary|          User_ID|Gender|               Age| EstimatedSalary|         prediction|
+-------+-----------------+------+------------------+----------------+-------------------+
|  count|              400|   400|               400|             400|                400|
|   mean|  1.56915397575E7|  null|            37.655|         69742.5|             0.3025|
| stddev|71658.32158119006|  null|10.482876597307927|34096.9602824248|0.45991581446062235|
|    min|         15566689|Female|                18|           15000|                0.0|
|    max|         15815236|  Male|                60|          150000|                1.0|
+-------+-----------------+------+------------------+----------------+-------------------+
 */

val final_out = spark.sql("SELECT purch_view.User_ID,purch_view.Gender,purch_view.Age," +
  "purch_view.EstimatedSalary,purch_view.label,df4_view.prediction FROM purch_view  " +
  "JOIN df4_view ON purch_view.User_ID = df4_view.User_ID GROUP BY purch_view.User_ID,purch_view.Gender,purch_view.Age," +
  "purch_view.EstimatedSalary,purch_view.label,df4_view.prediction")



    println("final_out describe")
    final_out.describe().show()

/*
+-------+-----------------+------+------------------+-----------------+------------------+-------------------+
|summary|          User_ID|Gender|               Age|  EstimatedSalary|             label|         prediction|
+-------+-----------------+------+------------------+-----------------+------------------+-------------------+
|  count|              400|   400|               400|              400|               400|                400|
|   mean|  1.56915397575E7|  null|            37.655|          69742.5|            0.3575|             0.3025|
| stddev|71658.32158119016|  null|10.482876597307916|34096.96028242478|0.4798639635968691|0.45991581446062224|
|    min|         15566689|Female|                18|            15000|               0.0|                0.0|
|    max|         15815236|  Male|                60|           150000|               1.0|                1.0|
+-------+-----------------+------+------------------+-----------------+------------------+-------------------+
 */

    println("final_out dataframe")
    final_out.show(25,false)

/*
+--------+------+---+---------------+-----+----------+
|User_ID |Gender|Age|EstimatedSalary|label|prediction|
+--------+------+---+---------------+-----+----------+
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
|15766609|Female|47 |47000          |0.0  |1.0       |
|15791373|Male  |60 |42000          |1.0  |1.0       |
|15589715|Female|48 |119000         |1.0  |1.0       |
|15709476|Male  |20 |49000          |0.0  |0.0       |
|15738448|Female|30 |79000          |0.0  |0.0       |
|15727467|Male  |57 |74000          |1.0  |1.0       |
|15748589|Female|45 |45000          |1.0  |0.0       |
|15654574|Female|23 |82000          |0.0  |0.0       |
|15679651|Female|26 |15000          |0.0  |0.0       |
|15672821|Female|55 |125000         |1.0  |1.0       |
|15671655|Female|35 |23000          |0.0  |0.0       |
|15733964|Female|38 |50000          |0.0  |0.0       |
+--------+------+---+---------------+-----+----------+
 */

    println("final_out,who can buy")
    final_out.filter( "label = 1").show(10,false)
/*
    +--------+------+---+---------------+-----+----------+
|User_ID |Gender|Age|EstimatedSalary|label|prediction|
+--------+------+---+---------------+-----+----------+
|15753861|Female|49 |141000         |1.0  |1.0       |
|15594041|Female|49 |36000          |1.0  |1.0       |
|15775335|Male  |56 |60000          |1.0  |1.0       |
|15769596|Female|56 |104000         |1.0  |1.0       |
|15596522|Male  |49 |89000          |1.0  |1.0       |
|15791373|Male  |60 |42000          |1.0  |1.0       |
|15589715|Female|48 |119000         |1.0  |1.0       |
|15727467|Male  |57 |74000          |1.0  |1.0       |
|15748589|Female|45 |45000          |1.0  |0.0       |
|15672821|Female|55 |125000         |1.0  |1.0       |
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
|15766609|Female|47 |47000          |0.0  |1.0       |
|15709476|Male  |20 |49000          |0.0  |0.0       |
+--------+------+---+---------------+-----+----------+
 */

  }

}
