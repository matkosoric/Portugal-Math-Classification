# Predicting Final Grade in Math in Portuguese Schools

This is a demo of logistic regression in Spark MLlib to predict a final grade.
Although G1 and G2 features are highly correlated with the final grade as a target column (G3) and should be excluded from the model, poor predicting performance forced me to include those features also.
Model is saved in the /spark-warehouse folder.

##### RMSE = 2.3788  

### Tools

* [Spark 2.4.0 MLlib](https://spark.apache.org/releases/spark-release-2-4-0.html) - Big Data Analytics Engine
* [Orange 3.19.0](https://orange.biolab.si/) - Data mining tool

### Dataset

Dataset was downloaded from UC Irvine Machine Learning Repository.
It has 32 attributes and 395 entries about students in secondary education in two Portuguese schools.
Labels on the selected data points show value of the final grade (G3).
[Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/student+performance)
  
Correlation between G1 and G3 features:
![G1 - G3 correlation - Matko Soric](https://raw.githubusercontent.com/matkosoric/Portugal-Math-Classification/master/src/main/resources/G1-G3-correlation.png?raw=true "G1-G3 Correlation")
  
Correlation between G1 and G3 features:
![G2 - G3 correlation - Matko Soric](https://raw.githubusercontent.com/matkosoric/Portugal-Math-Classification/master/src/main/resources/G2-G3-correlation.png?raw=true "G2-G3 Correlation")
  
### Results

<pre><code>
root
 |-- school: string (nullable = true)
 |-- sex: string (nullable = true)
 |-- age: integer (nullable = true)
 |-- address: string (nullable = true)
 |-- famsize: string (nullable = true)
 |-- Pstatus: string (nullable = true)
 |-- Medu: integer (nullable = true)
 |-- Fedu: integer (nullable = true)
 |-- Mjob: string (nullable = true)
 |-- Fjob: string (nullable = true)
 |-- reason: string (nullable = true)
 |-- guardian: string (nullable = true)
 |-- traveltime: integer (nullable = true)
 |-- studytime: integer (nullable = true)
 |-- failures: integer (nullable = true)
 |-- schoolsup: string (nullable = true)
 |-- famsup: string (nullable = true)
 |-- paid: string (nullable = true)
 |-- activities: string (nullable = true)
 |-- nursery: string (nullable = true)
 |-- higher: string (nullable = true)
 |-- internet: string (nullable = true)
 |-- romantic: string (nullable = true)
 |-- famrel: integer (nullable = true)
 |-- freetime: integer (nullable = true)
 |-- goout: integer (nullable = true)
 |-- Dalc: integer (nullable = true)
 |-- Walc: integer (nullable = true)
 |-- health: integer (nullable = true)
 |-- absences: integer (nullable = true)
 |-- G1: integer (nullable = true)
 |-- G2: integer (nullable = true)
 |-- G3: integer (nullable = true)

+----+----+----+----------+---------+--------+------+--------+-----+----+----+------+--------+--------------+-----------+---------------+---------------+---------------+------------+------------+--------------+----------------+-----------------+--------------+------------+------------------+---------------+--------------+----------------+----------------+-----+----+----+
| age|Medu|Fedu|traveltime|studytime|failures|famrel|freetime|goout|Dalc|Walc|health|absences|school_indexed|sex_indexed|address_indexed|famsize_indexed|Pstatus_indexed|Mjob_indexed|Fjob_indexed|reason_indexed|guardian_indexed|schoolsup_indexed|famsup_indexed|paid_indexed|activities_indexed|nursery_indexed|higher_indexed|internet_indexed|romantic_indexed|label|  G1|  G2|
+----+----+----+----------+---------+--------+------+--------+-----+----+----+------+--------+--------------+-----------+---------------+---------------+---------------+------------+------------+--------------+----------------+-----------------+--------------+------------+------------------+---------------+--------------+----------------+----------------+-----+----+----+
|18.0| 4.0| 4.0|       2.0|      2.0|     0.0|   4.0|     3.0|  4.0| 1.0| 1.0|   3.0|     6.0|           0.0|        0.0|            0.0|            0.0|            1.0|         2.0|         2.0|           0.0|             0.0|              1.0|           1.0|         0.0|               1.0|            0.0|           0.0|             1.0|             0.0|    6| 5.0| 6.0|
|17.0| 1.0| 1.0|       1.0|      2.0|     0.0|   5.0|     3.0|  3.0| 1.0| 1.0|   3.0|     4.0|           0.0|        0.0|            0.0|            0.0|            0.0|         2.0|         0.0|           0.0|             1.0|              0.0|           0.0|         0.0|               1.0|            1.0|           0.0|             0.0|             0.0|    6| 5.0| 5.0|
|15.0| 1.0| 1.0|       1.0|      2.0|     3.0|   4.0|     3.0|  2.0| 2.0| 3.0|   3.0|    10.0|           0.0|        0.0|            0.0|            1.0|            0.0|         2.0|         0.0|           3.0|             0.0|              1.0|           1.0|         1.0|               1.0|            0.0|           0.0|             0.0|             0.0|   10| 7.0| 8.0|
|15.0| 4.0| 2.0|       1.0|      3.0|     0.0|   3.0|     2.0|  2.0| 1.0| 1.0|   5.0|     2.0|           0.0|        0.0|            0.0|            0.0|            0.0|         4.0|         1.0|           1.0|             0.0|              0.0|           0.0|         1.0|               0.0|            0.0|           0.0|             0.0|             1.0|   15|15.0|14.0|
|16.0| 3.0| 3.0|       1.0|      2.0|     0.0|   4.0|     3.0|  2.0| 1.0| 2.0|   5.0|     4.0|           0.0|        0.0|            0.0|            0.0|            0.0|         0.0|         0.0|           1.0|             1.0|              0.0|           0.0|         1.0|               1.0|            0.0|           0.0|             1.0|             0.0|   10| 6.0|10.0|
|16.0| 4.0| 3.0|       1.0|      2.0|     0.0|   5.0|     4.0|  2.0| 1.0| 2.0|   5.0|    10.0|           0.0|        1.0|            0.0|            1.0|            0.0|         1.0|         0.0|           2.0|             0.0|              0.0|           0.0|         1.0|               0.0|            0.0|           0.0|             0.0|             0.0|   15|15.0|15.0|
|16.0| 2.0| 2.0|       1.0|      2.0|     0.0|   4.0|     4.0|  4.0| 1.0| 1.0|   3.0|     0.0|           0.0|        1.0|            0.0|            1.0|            0.0|         0.0|         0.0|           1.0|             0.0|              0.0|           1.0|         0.0|               1.0|            0.0|           0.0|             0.0|             0.0|   11|12.0|12.0|
|17.0| 4.0| 4.0|       2.0|      2.0|     0.0|   4.0|     1.0|  4.0| 1.0| 1.0|   1.0|     6.0|           0.0|        0.0|            0.0|            0.0|            1.0|         0.0|         2.0|           1.0|             0.0|              1.0|           0.0|         0.0|               1.0|            0.0|           0.0|             1.0|             0.0|    6| 6.0| 5.0|
|15.0| 3.0| 2.0|       1.0|      2.0|     0.0|   4.0|     2.0|  2.0| 1.0| 1.0|   1.0|     0.0|           0.0|        1.0|            0.0|            1.0|            1.0|         1.0|         0.0|           1.0|             0.0|              0.0|           0.0|         1.0|               1.0|            0.0|           0.0|             0.0|             0.0|   19|16.0|18.0|
|15.0| 3.0| 4.0|       1.0|      2.0|     0.0|   5.0|     5.0|  1.0| 1.0| 1.0|   5.0|     0.0|           0.0|        1.0|            0.0|            0.0|            0.0|         0.0|         0.0|           1.0|             0.0|              0.0|           0.0|         1.0|               0.0|            0.0|           0.0|             0.0|             0.0|   15|14.0|15.0|
|15.0| 4.0| 4.0|       1.0|      2.0|     0.0|   3.0|     3.0|  3.0| 1.0| 2.0|   2.0|     0.0|           0.0|        0.0|            0.0|            0.0|            0.0|         3.0|         4.0|           2.0|             0.0|              0.0|           0.0|         1.0|               1.0|            0.0|           0.0|             0.0|             0.0|    9|10.0| 8.0|
|15.0| 2.0| 1.0|       3.0|      3.0|     0.0|   5.0|     2.0|  2.0| 1.0| 1.0|   4.0|     4.0|           0.0|        0.0|            0.0|            0.0|            0.0|         1.0|         0.0|           2.0|             1.0|              0.0|           0.0|         0.0|               0.0|            0.0|           0.0|             0.0|             0.0|   12|10.0|12.0|
|15.0| 4.0| 4.0|       1.0|      1.0|     0.0|   4.0|     3.0|  3.0| 1.0| 3.0|   5.0|     2.0|           0.0|        1.0|            0.0|            1.0|            0.0|         4.0|         1.0|           0.0|             1.0|              0.0|           0.0|         1.0|               0.0|            0.0|           0.0|             0.0|             0.0|   14|14.0|14.0|
|15.0| 4.0| 3.0|       2.0|      2.0|     0.0|   5.0|     4.0|  3.0| 1.0| 2.0|   3.0|     2.0|           0.0|        1.0|            0.0|            0.0|            0.0|         3.0|         0.0|           0.0|             0.0|              0.0|           0.0|         1.0|               1.0|            0.0|           0.0|             0.0|             0.0|   11|10.0|10.0|
|15.0| 2.0| 2.0|       1.0|      3.0|     0.0|   4.0|     5.0|  2.0| 1.0| 1.0|   3.0|     0.0|           0.0|        1.0|            0.0|            0.0|            1.0|         0.0|         0.0|           1.0|             2.0|              0.0|           0.0|         0.0|               1.0|            0.0|           0.0|             0.0|             1.0|   16|14.0|16.0|
|16.0| 4.0| 4.0|       1.0|      1.0|     0.0|   4.0|     4.0|  4.0| 1.0| 2.0|   2.0|     4.0|           0.0|        0.0|            0.0|            0.0|            0.0|         4.0|         0.0|           1.0|             0.0|              0.0|           0.0|         0.0|               1.0|            0.0|           0.0|             0.0|             0.0|   14|14.0|14.0|
|16.0| 4.0| 4.0|       1.0|      3.0|     0.0|   3.0|     2.0|  3.0| 1.0| 2.0|   2.0|     6.0|           0.0|        0.0|            0.0|            0.0|            0.0|         1.0|         1.0|           2.0|             0.0|              0.0|           0.0|         1.0|               0.0|            0.0|           0.0|             0.0|             0.0|   14|13.0|14.0|
|16.0| 3.0| 3.0|       3.0|      2.0|     0.0|   5.0|     3.0|  2.0| 1.0| 1.0|   4.0|     4.0|           0.0|        0.0|            0.0|            0.0|            0.0|         0.0|         0.0|           2.0|             0.0|              1.0|           0.0|         0.0|               0.0|            0.0|           0.0|             1.0|             0.0|   10| 8.0|10.0|
|17.0| 3.0| 2.0|       1.0|      1.0|     3.0|   5.0|     5.0|  5.0| 2.0| 4.0|   5.0|    16.0|           0.0|        1.0|            0.0|            0.0|            0.0|         1.0|         1.0|           0.0|             0.0|              0.0|           0.0|         0.0|               0.0|            0.0|           0.0|             0.0|             0.0|    5| 6.0| 5.0|
|16.0| 4.0| 3.0|       1.0|      1.0|     0.0|   3.0|     1.0|  3.0| 1.0| 3.0|   5.0|     4.0|           0.0|        1.0|            0.0|            1.0|            0.0|         4.0|         0.0|           1.0|             1.0|              0.0|           1.0|         1.0|               0.0|            0.0|           0.0|             0.0|             0.0|   10| 8.0|10.0|
+----+----+----+----------+---------+--------+------+--------+-----+----+----+------+--------+--------------+-----------+---------------+---------------+---------------+------------+------------+--------------+----------------+-----------------+--------------+------------+------------------+---------------+--------------+----------------+----------------+-----+----+----+
only showing top 20 rows

Model hyper-parameters:
{
	logreg_1079f9d337e0-aggregationDepth: 2,
	logreg_1079f9d337e0-elasticNetParam: 0.001,
	logreg_1079f9d337e0-family: auto,
	logreg_1079f9d337e0-featuresCol: selectedFeatures,
	logreg_1079f9d337e0-fitIntercept: true,
	logreg_1079f9d337e0-labelCol: label,
	logreg_1079f9d337e0-maxIter: 10,
	logreg_1079f9d337e0-predictionCol: prediction,
	logreg_1079f9d337e0-probabilityCol: probability,
	logreg_1079f9d337e0-rawPredictionCol: rawPrediction,
	logreg_1079f9d337e0-regParam: 0.01,
	logreg_1079f9d337e0-standardization: true,
	logreg_1079f9d337e0-threshold: 0.001,
	logreg_1079f9d337e0-tol: 1.0E-6
}
Pearson correlation matrix:
1.0                    -0.14134800560351005    -0.14328091671130977   0.08346300654976704    -0.01295013498336172   0.21423367639916452    0.052380064040017736    3.836698986096511E-4   0.18360617745190172    0.10480740703710131     0.12150145861500565    -0.08573656255838923   0.19093782220662714    0.3532507196608683      -0.08748505845356876   0.12158090252093481    0.033252673121696365   -0.04539406772489167    -0.08676753897701453   -0.045785776987062765  0.053087485473170284    0.2675713743208881     -0.26294888858401444   0.1386309962204759     8.686549425520958E-4    0.10522851673934797    0.07443492868270701     0.22141017223407713    0.12204400880656785    0.20781783408303783     -0.08789052917386155    -0.17481972807957116   
-0.14134800560351005   1.0                     0.6054279471376931     -0.1544164373345926    0.05320042476343546    -0.2048844141946284    -0.0012224269608619796  0.034770680412664234   0.022970651640258556   0.04259815836449059     -0.05640482940092041   -0.01837370614049859   0.08577062120022352    -0.09277750389597127    0.1035816234613191     -0.12132072397872      -0.038947741167282474  0.11924288160988823     0.3875784762628307     0.13582721432456116    0.09632819248537147     -0.0900072111723233    -0.04042398028479715   -0.21121993631155164   0.14034237818695217     -0.12087489245873011   -0.17795831873097437    -0.13037426624252754   -0.15362164395649003   0.03630503060891339     0.21570047315362692     0.22863381592100004    
-0.14328091671130977   0.6054279471376931      1.0                    -0.1456098056500115    -0.019270884292221888  -0.23435350219398485   -0.005407245952085541   -0.04680070386647827   0.05050382965717539    0.029053006937150946    -0.008024214101889195  0.01847321356179954    0.03212601497940499    -0.04269808586901299    0.06059938903943988    -0.06063182526771076   -0.030832593561202125  0.09663251012461159     0.2551728421191196     0.2931140551340112     0.03343978395638027     0.01671870439634169    0.061664970332391274   -0.19939867707886164   0.08308060667159355     -0.0997980591217156    -0.1572706540999979     -0.14282463579631877   -0.12039352746949704   0.008083083079347745    0.2024013260522016      0.1775904560668182     
0.08346300654976704    -0.1544164373345926     -0.1456098056500115    1.0                    -0.072779991950643     0.0992848875605291     -0.020953836567937274   -0.008672003359007622  0.0187444214196289     0.12156919004811932     0.08509060898774264    0.015765148743931805   -0.012459398263291819  0.26420843061131666     0.015512760136775269   0.3412844757601383     0.0793743666688041     -0.015178940358280811   -0.08326333624849494   -0.07440303792883461   -0.09463119473433665    0.0699495127543545     -0.07388629877511282   0.03549841512453246    -0.08203862134768208    0.005044039366879727   0.018614519776247923    0.08611459260875556    0.09588232870621566    0.049423455838040746    -0.05766688267241808    -0.11123983687910002   
-0.01295013498336172   0.05320042476343546     -0.019270884292221888  -0.072779991950643     1.0                    -0.15417103717500494   -0.006908367262858958   -0.11483523994329728   -0.04738088016880163   -0.1735247260832542     -0.20769540909301998   -0.12913516107984008   -0.04262273816494896   -0.05748383635951097    -0.3228100922392402    0.006515624620345845   -0.08010141563214071   -0.00964493898649527    -0.018242463918688688  0.05232499099982435    0.04724682603896347     0.06985012138450888    0.03314281248210835    -0.1578004481466311    0.18237417442241252     -0.045811026108247985  -0.10597852638413524    -0.19746283065003323   -0.047291089000663006  0.058455421104591995    0.1549816324975247      0.1441380343544417     
0.21423367639916452    -0.2048844141946284     -0.23435350219398485   0.0992848875605291     -0.15417103717500494   1.0                    -0.04277206014854146    0.01837003186131827    0.15383247662344307    0.12348058415636964     0.15305776315287425    0.05408519169280742    0.07486475776523499    0.012505598930627892    0.0022526439577380984  0.08412458419044158    0.03461851969133442    0.012505598930627892    -0.07786413825889707   0.0277082153289939     -0.024701650315280082   0.1908296639743931     -0.023197231391020402  0.07770723989210482    -0.14118288130322942    0.07882241499553908    0.060268175046268364    0.2483149325247724     0.05105295576100704    0.1386417014611702      -0.3715313132762741     -0.3717605954324395    
0.052380064040017736   -0.0012224269608619796  -0.005407245952085541  -0.020953836567937274  -0.006908367262858958  -0.04277206014854146   1.0                     0.19688729875594296    0.10940057290353174    -0.06411201644991102    -0.09794543057984291   0.10313379495510348    -0.03804923321367136   -0.07926768943281397    0.08056649885023708    -0.04063959326950652   -0.028505422804962426  -0.03438552737953116    -0.04912533513117055   -0.046831209719556476  -0.026453021296640126   0.0733019588625999     0.014351254497986037   0.02521943415824519    0.003506006413566438    -0.042017395164031746  -0.015979565534698027   -0.008956002061176296  -0.045233329907468305  -0.028692761961412832   0.007311138205253501    -0.03243312263655729   
3.836698986096511E-4   0.034770680412664234    -0.04680070386647827   -0.008672003359007622  -0.11483523994329728   0.01837003186131827    0.19688729875594296     1.0                    0.30353640358068923    0.18245046817237584     0.14305070620110116    0.05021218342712054    -0.0728242734092499    0.029513979747000717    0.22110433436706656    -0.05600249669245977   0.05106140886452288    -0.0420450549468432     0.053540592844244776   -0.022392121634565557  -0.027031759164309124   0.042714476848443185   -0.020546467545938647  0.01053099573607802    -0.04457203286492424    -0.10160919686812968   -0.01294091843841107    0.036167945619441644   -0.08171789565461628   0.02607559785952485     -0.008606350145682333   -0.020685882223513308  
0.18360617745190172    0.022970651640258556    0.05050382965717539    0.0187444214196289     -0.04738088016880163   0.15383247662344307    0.10940057290353174     0.30353640358068923    1.0                    0.263669664512258       0.4163378394159751     -0.025687209242066692  -0.00227375568496698   -0.017056918313437794   0.06983246905818309    -0.07992569188803528   0.05435913429805569    0.0014962209046870036   0.016873650291614407   -0.04992726562954679   -0.015959193897863752   -0.06577640796939752   -0.020904795575094535  -0.004247250666157547  0.011670626490574317    -0.07697631605115299   0.016146885755551375    0.09201379325239942    -0.02053748038599388   0.024872093510299127    -0.1814754917402133     -0.18143957020205678   
0.10480740703710131    0.04259815836449059     0.029053006937150946   0.12156919004811932    -0.1735247260832542    0.12348058415636964    -0.06411201644991102    0.18245046817237584    0.263669664512258      1.0                     0.6322589831748876     0.07758568429844874    0.1062318639234786     0.06636383353758056     0.2543035074758287     0.06570648019288305    0.06589458612000634    -0.0029826442039361492  -0.042385378768444434  0.046199106752006905   0.11635633225944253     0.056239642468606646   -0.006335376939321634  0.014431856827351366   0.07631396763921067     0.05512670196077484    0.11156081323071983     0.09805611744978793    -0.036408508359686516  0.0487806634518908      -0.083332094366119      -0.0541441371779579    
0.12150145861500565    -0.05640482940092041    -0.008024214101889195  0.08509060898774264    -0.20769540909301998   0.15305776315287425    -0.09794543057984291    0.14305070620110116    0.4163378394159751     0.6322589831748876      1.0                    0.12601107877095352    0.11778505990078432    0.02636682073951795     0.2635901318332627     0.08273815406394334    0.09461576833426404    -0.028124608788819096   0.025107171194233727   -0.12073317050220246   0.0659692553150441      -0.03361207931995223   -0.0817854511882001    0.060751811685526555   0.06180132280284384     0.03299692929400295    0.12441141073537912     0.1389990465942951     -0.016376731361636324  0.010052972568951756    -0.14138327955954957    -0.08756681871079501   
-0.08573656255838923   -0.01837370614049859    0.01847321356179954    0.015765148743931805   -0.12913516107984008   0.05408519169280742    0.10313379495510348     0.05021218342712054    -0.025687209242066692  0.07758568429844874     0.12601107877095352    1.0                    0.014772674908168595   -0.056921394112430315   0.15661058104965828    0.045872552429251943   0.009084930095080782   0.023618835731298575    0.018013927741745626   0.03425269023492854    -0.08957383894238145    -0.04663539604537577   -0.012040395058443176  -9.142589793396497E-4  -0.048134224479400516   -0.002750768346223814  0.0069515253281411974   -0.01109729891205435   0.08580755037345458    0.04514844875847757     -0.08718975103336322    -0.10326680422855783   
0.19093782220662714    0.08577062120022352     0.03212601497940499    -0.012459398263291819  -0.04262273816494896   0.07486475776523499    -0.03804923321367136    -0.0728242734092499    -0.00227375568496698   0.1062318639234786      0.11778505990078432    0.014772674908168595   1.0                    -0.07748858755152377    -0.09449048652328552   0.07317890189611302    -0.01502382361908087   0.1397909695024977      -0.07492962066772861   -0.008873728623896033  0.09921922987634603     0.07494626664599657    0.010989245975072191   -0.01751888428352511   0.03674342062298524     0.04240824176564887    0.0263392092243621      0.08636075130223883    -0.08711809440468038   0.1431429985287586      -0.011960503150713092   -0.017146349600279925  
0.3532507196608683     -0.09277750389597127    -0.04269808586901299   0.26420843061131666    -0.05748383635951097   0.012505598930627892   -0.07926768943281397    0.029513979747000717   -0.017056918313437794  0.06636383353758056     0.02636682073951795    -0.056921394112430315  -0.07748858755152377   1.0                     -0.03259510318953112   0.24614142971261313    0.09188663612136072    -0.030649350649350655   -0.005724057259127565  0.0203862620642869     0.015391537273413657    0.06943059179968103    -0.13731430681184992   0.1638844030958943     -0.009600288802948631   0.10982303526167037    0.05096471914376255     0.003610616551805645   0.1716112008008807     0.057500924682621184    -0.0066194540317998695  -0.031141925024272467  
-0.08748505845356876   0.1035816234613191      0.06059938903943988    0.015512760136775269   -0.3228100922392402    0.0022526439577380984  0.08056649885023708     0.22110433436706656    0.06983246905818309    0.2543035074758287      0.2635901318332627     0.15661058104965828    -0.09449048652328552   -0.03259510318953112    1.0                    0.010400192056464677   0.1041798238626917     -0.053008198116308164   0.05713108591192772    -0.07196429404840295   -0.017559106098534408   -0.03425493627144654   -0.1535439136087504    0.1544226762894954     -0.12962301045238767    -0.10303228636187364   -0.054911898128459244   0.16851673963718397    -0.020629021314330188  -0.11753645994508118    0.07101048943748058     0.08944402029916375    
0.12158090252093481    -0.12132072397872       -0.06063182526771076   0.3412844757601383     0.006515624620345845   0.08412458419044158    -0.04063959326950652    -0.05600249669245977   -0.07992569188803528   0.06570648019288305     0.08273815406394334    0.045872552429251943   0.07317890189611302    0.24614142971261313     0.010400192056464677   1.0                    -0.10902159313573673   -0.056052008746436684   -0.08126049284318948   -0.06839584898953198   0.060974347308044206    -0.04980656271523206   -0.006135084781935909  0.03563776842541105    -0.10084353864416695    -0.05111003569426144   0.023909126739875157    0.052170639965328724   0.22508989825326045    0.041674667399190034    -0.08518738467719252    -0.13630253625178793   
0.033252673121696365   -0.038947741167282474   -0.030832593561202125  0.0793743666688041     -0.08010141563214071   0.03461851969133442    -0.028505422804962426   0.05106140886452288    0.05435913429805569    0.06589458612000634     0.09461576833426404    0.009084930095080782   -0.01502382361908087   0.09188663612136072     0.1041798238626917     -0.10902159313573673   1.0                    0.11449366564328278     0.06756863486156924    0.06343815369512122    -0.05617820878473355    0.007319181180911839   -0.007572913288975979  0.09503699054878638    -0.02723306092029181    -0.004062021233329103  -0.06439092829111602    0.005474162252947175   -0.009213166547046339  0.060806714754985364    0.10263100625527856     0.10186240300316772    
-0.04539406772489167   0.11924288160988823     0.09663251012461159    -0.015178940358280811  -0.00964493898649527   0.012505598930627892   -0.03438552737953116    -0.0420450549468432    0.0014962209046870036  -0.0029826442039361492  -0.028124608788819096  0.023618835731298575   0.1397909695024977     -0.030649350649350655   -0.053008198116308164  -0.056052008746436684  0.11449366564328278    1.0                     -0.07515936053289148   0.049112358609418365   -0.03591358697129793    0.005420232073581035   0.07552286874651747    0.01843280607418034    -0.07117455491841214    0.06905041139206816    -0.07644707871564384    -0.04116102869058428   0.11662466617734267    0.07884857336718716     0.01770091041095926     0.043944716423140885   
-0.08676753897701453   0.3875784762628307      0.2551728421191196     -0.08326333624849494   -0.018242463918688688  -0.07786413825889707   -0.04912533513117055    0.053540592844244776   0.016873650291614407   -0.042385378768444434   0.025107171194233727   0.018013927741745626   -0.07492962066772861   -0.005724057259127565   0.05713108591192772    -0.08126049284318948   0.06756863486156924    -0.07515936053289148    1.0                    0.1884366522603547     0.033427701803666665    -0.09043997014732241   -0.09773676393082215   -0.17886312847003935   0.1526014007045367      -0.089538107790263     -0.1269637806716619     -0.050440078715506924  -0.08633665763259667   -0.028471216003865506   0.15102147066021368     0.13352478217083885    
-0.045785776987062765  0.13582721432456116     0.2931140551340112     -0.07440303792883461   0.05232499099982435    0.0277082153289939     -0.046831209719556476   -0.022392121634565557  -0.04992726562954679   0.046199106752006905    -0.12073317050220246   0.03425269023492854    -0.008873728623896033  0.0203862620642869      -0.07196429404840295   -0.06839584898953198   0.06343815369512122    0.049112358609418365    0.1884366522603547     1.0                    0.11043168917953788     0.0956423927772156     0.09301925172767514    -0.07393108420426223   -0.013306302646976551   0.001407679782285773   -0.12575988541855382    0.012623596253447613   0.047776822153969604   0.027392954242047164    0.12038092775124858     0.09139152002996341    
0.053087485473170284   0.09632819248537147     0.03343978395638027    -0.09463119473433665   0.04724682603896347    -0.024701650315280082  -0.026453021296640126   -0.027031759164309124  -0.015959193897863752  0.11635633225944253     0.0659692553150441     -0.08957383894238145   0.09921922987634603    0.015391537273413657    -0.017559106098534408  0.060974347308044206   -0.05617820878473355   -0.03591358697129793    0.033427701803666665   0.11043168917953788    1.0                     0.05047255091460003    0.009686775410893035   -0.009929822761051766  0.1719742886702205      -0.0688453027488522    -0.0016236761780489893  -0.009984604749602444  -0.0797071958526684    0.061890105508350605    0.0663438158290415      0.09927515174871136    
0.2675713743208881     -0.0900072111723233     0.01671870439634169    0.0699495127543545     0.06985012138450888    0.1908296639743931     0.0733019588625999      0.042714476848443185   -0.06577640796939752   0.056239642468606646    -0.03361207931995223   -0.04663539604537577   0.07494626664599657    0.06943059179968103     -0.03425493627144654   -0.04980656271523206   0.007319181180911839   0.005420232073581035    -0.09043997014732241   0.0956423927772156     0.05047255091460003     1.0                    -0.054580196729163456  0.01248876083617142    0.015559898417462399    0.001110926199928064   0.14940012737943548     -0.014136337559598528  -0.03970258394518818   0.07907107069464708     0.011718251085562155    -0.030765499385132376  
-0.26294888858401444   -0.04042398028479715    0.061664970332391274   -0.07388629877511282   0.03314281248210835    -0.023197231391020402  0.014351254497986037    -0.020546467545938647  -0.020904795575094535  -0.006335376939321634   -0.0817854511882001    -0.012040395058443176  0.010989245975072191   -0.13731430681184992    -0.1535439136087504    -0.006135084781935909  -0.007572913288975979  0.07552286874651747     -0.09773676393082215   0.09301925172767514    0.009686775410893035    -0.054580196729163456  1.0                    -0.06137888372532828   -0.013125748599197231   -0.04097409765787012   -0.04811252243246886    -0.09271260077824707   0.010884199774917513   -0.10206491940937294    -0.1830263245185165     -0.09146384784919027   
0.1386309962204759     -0.21121993631155164    -0.19939867707886164   0.03549841512453246    -0.1578004481466311    0.07770723989210482    0.02521943415824519     0.01053099573607802    -0.004247250666157547  0.014431856827351366    0.060751811685526555   -9.142589793396497E-4  -0.01751888428352511   0.1638844030958943      0.1544226762894954     0.03563776842541105    0.09503699054878638    0.01843280607418034     -0.17886312847003935   -0.07393108420426223   -0.009929822761051766   0.01248876083617142    -0.06137888372532828   1.0                    -0.3342413219983456     0.0012727928425323141  0.06575959492214284     0.11973027667907865    0.14962213668381386    -0.013105991066563247   0.035461206950764       0.024388432570382052   
8.686549425520958E-4   0.14034237818695217     0.08308060667159355    -0.08203862134768208   0.18237417442241252    -0.14118288130322942   0.003506006413566438    -0.04457203286492424   0.011670626490574317   0.07631396763921067     0.06180132280284384    -0.048134224479400516  0.03674342062298524    -0.009600288802948631   -0.12962301045238767   -0.10084353864416695   -0.02723306092029181   -0.07117455491841214    0.1526014007045367     -0.013306302646976551  0.1719742886702205      0.015559898417462399   -0.013125748599197231  -0.3342413219983456    1.0                     0.013745731263822754   -0.08769007335028715    -0.18582096578007293   -0.14960139252726018   -0.0023258476127543046  0.02527865470413117     0.0882794893094        
0.10522851673934797    -0.12087489245873011    -0.0997980591217156    0.005044039366879727   -0.045811026108247985  0.07882241499553908    -0.042017395164031746   -0.10160919686812968   -0.07697631605115299   0.05512670196077484     0.03299692929400295    -0.002750768346223814  0.04240824176564887    0.10982303526167037     -0.10303228636187364   -0.05111003569426144   -0.004062021233329103  0.06905041139206816     -0.089538107790263     0.001407679782285773   -0.0688453027488522     0.001110926199928064   -0.04097409765787012   0.0012727928425323141  0.013745731263822754    1.0                    0.012903494352312713    0.12597033581420927    0.023240352525126164   0.0020050439395291157   -0.029670441168488836   -0.03514150655738285   
0.07443492868270701    -0.17795831873097437    -0.1572706540999979    0.018614519776247923   -0.10597852638413524   0.060268175046268364   -0.015979565534698027   -0.01294091843841107   0.016146885755551375   0.11156081323071983     0.12441141073537912    0.0069515253281411974  0.0263392092243621     0.05096471914376255     -0.054911898128459244  0.023909126739875157   -0.06439092829111602   -0.07644707871564384    -0.1269637806716619    -0.12575988541855382   -0.0016236761780489893  0.14940012737943548    -0.04811252243246886   0.06575959492214284    -0.08769007335028715    0.012903494352312713   1.0                     0.021253623756587695   -0.026102751244458773  0.0033779928551742923   -0.1034254052132962     -0.10099277010072365   
0.22141017223407713    -0.13037426624252754    -0.14282463579631877   0.08611459260875556    -0.19746283065003323   0.2483149325247724     -0.008956002061176296   0.036167945619441644   0.09201379325239942    0.09805611744978793     0.1389990465942951     -0.01109729891205435   0.08636075130223883    0.003610616551805645    0.16851673963718397    0.052170639965328724   0.005474162252947175   -0.04116102869058428    -0.050440078715506924  0.012623596253447613   -0.009984604749602444   -0.014136337559598528  -0.09271260077824707   0.11973027667907865    -0.18582096578007293    0.12597033581420927    0.021253623756587695    1.0                    -0.030451150961888315  0.14904556717155346     -0.19705097677695935    -0.20113013480663997   
0.12204400880656785    -0.15362164395649003    -0.12039352746949704   0.09588232870621566    -0.047291089000663006  0.05105295576100704    -0.045233329907468305   -0.08171789565461628   -0.02053748038599388   -0.036408508359686516   -0.016376731361636324  0.08580755037345458    -0.08711809440468038   0.1716112008008807      -0.020629021314330188  0.22508989825326045    -0.009213166547046339  0.11662466617734267     -0.08633665763259667   0.047776822153969604   -0.0797071958526684     -0.03970258394518818   0.010884199774917513   0.14962213668381386    -0.14960139252726018    0.023240352525126164   -0.026102751244458773   -0.030451150961888315  1.0                    -0.05343399376572982    -0.06198817631150185    -0.1095679963256042    
0.20781783408303783    0.03630503060891339     0.008083083079347745   0.049423455838040746   0.058455421104591995   0.1386417014611702     -0.028692761961412832   0.02607559785952485    0.024872093510299127   0.0487806634518908      0.010052972568951756   0.04514844875847757    0.1431429985287586     0.057500924682621184    -0.11753645994508118   0.041674667399190034   0.060806714754985364   0.07884857336718716     -0.028471216003865506  0.027392954242047164   0.061890105508350605    0.07907107069464708    -0.10206491940937294   -0.013105991066563247  -0.0023258476127543046  0.0020050439395291157  0.0033779928551742923   0.14904556717155346    -0.05343399376572982   1.0                     -0.10520107868720752    -0.17890170432666078   
-0.08789052917386155   0.21570047315362692     0.2024013260522016     -0.05766688267241808   0.1549816324975247     -0.3715313132762741    0.007311138205253501    -0.008606350145682333  -0.1814754917402133    -0.083332094366119      -0.14138327955954957   -0.08718975103336322   -0.011960503150713092  -0.0066194540317998695  0.07101048943748058    -0.08518738467719252   0.10263100625527856    0.01770091041095926     0.15102147066021368    0.12038092775124858    0.0663438158290415      0.011718251085562155   -0.1830263245185165    0.035461206950764      0.02527865470413117     -0.029670441168488836  -0.1034254052132962     -0.19705097677695935   -0.06198817631150185   -0.10520107868720752    1.0                     0.862521494253055      
-0.17481972807957116   0.22863381592100004     0.1775904560668182     -0.11123983687910002   0.1441380343544417     -0.3717605954324395    -0.03243312263655729    -0.020685882223513308  -0.18143957020205678   -0.0541441371779579     -0.08756681871079501   -0.10326680422855783   -0.017146349600279925  -0.031141925024272467   0.08944402029916375    -0.13630253625178793   0.10186240300316772    0.043944716423140885    0.13352478217083885    0.09139152002996341    0.09927515174871136     -0.030765499385132376  -0.09146384784919027   0.024388432570382052   0.0882794893094         -0.03514150655738285   -0.10099277010072365    -0.20113013480663997   -0.1095679963256042    -0.17890170432666078    0.862521494253055       1.0                    
+----------------------------------------------------------------------------------------------------------------------------------------------+-----+----------+
|features                                                                                                                                      |label|prediction|
+----------------------------------------------------------------------------------------------------------------------------------------------+-----+----------+
|(32,[0,1,2,3,4,6,7,8,9,10,11,12,16,18,20,24,25,30,31],[15.0,3.0,2.0,1.0,2.0,4.0,4.0,4.0,1.0,1.0,5.0,10.0,1.0,1.0,2.0,1.0,1.0,7.0,6.0])        |6    |8.0       |
|(32,[0,1,2,3,4,6,7,8,9,10,11,14,20,24,30,31],[15.0,3.0,4.0,1.0,2.0,5.0,5.0,1.0,1.0,1.0,5.0,1.0,1.0,1.0,14.0,15.0])                            |15   |15.0      |
|(32,[0,1,2,3,4,6,7,8,9,10,11,18,19,20,24,25,30,31],[15.0,4.0,4.0,1.0,2.0,3.0,3.0,3.0,1.0,2.0,2.0,3.0,4.0,2.0,1.0,1.0,10.0,8.0])               |9    |10.0      |
|[16.0,3.0,3.0,3.0,1.0,0.0,3.0,3.0,4.0,3.0,5.0,3.0,8.0,0.0,1.0,1.0,1.0,0.0,3.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,9.0,9.0]            |10   |10.0      |
|(32,[0,1,2,3,4,6,7,8,9,10,11,19,22,24,25,30,31],[16.0,4.0,3.0,1.0,3.0,5.0,3.0,5.0,1.0,1.0,3.0,3.0,1.0,1.0,1.0,7.0,9.0])                       |8    |8.0       |
|(32,[0,1,2,3,4,6,7,8,9,10,11,12,18,20,25,30,31],[16.0,4.0,4.0,1.0,1.0,4.0,4.0,4.0,1.0,2.0,2.0,4.0,4.0,1.0,1.0,14.0,14.0])                     |14   |15.0      |
|(32,[0,1,2,3,4,6,7,8,9,10,11,18,19,20,26,30,31],[16.0,4.0,4.0,1.0,3.0,5.0,3.0,2.0,1.0,1.0,5.0,3.0,1.0,1.0,1.0,13.0,13.0])                     |14   |15.0      |
|[17.0,1.0,1.0,3.0,1.0,1.0,5.0,2.0,1.0,1.0,2.0,1.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,2.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,1.0,7.0,6.0]            |0    |0.0       |
|[17.0,3.0,2.0,1.0,1.0,1.0,4.0,4.0,4.0,3.0,4.0,3.0,19.0,0.0,1.0,0.0,1.0,1.0,3.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,0.0,11.0,9.0]          |10   |9.0       |
|(32,[0,1,2,3,4,6,7,8,9,10,11,18,19,20,21,24,26,30,31],[17.0,3.0,2.0,1.0,4.0,5.0,2.0,2.0,1.0,2.0,5.0,4.0,4.0,2.0,1.0,1.0,1.0,17.0,17.0])       |18   |15.0      |
|[17.0,4.0,3.0,2.0,2.0,0.0,4.0,5.0,5.0,1.0,3.0,2.0,4.0,1.0,1.0,1.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,1.0,0.0,0.0,1.0,13.0,11.0]          |11   |14.0      |
|[17.0,4.0,4.0,1.0,1.0,0.0,5.0,2.0,1.0,1.0,2.0,3.0,12.0,0.0,0.0,1.0,1.0,0.0,1.0,0.0,3.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,1.0,0.0,8.0,10.0]          |10   |11.0      |
|(32,[0,1,2,3,4,6,7,8,9,10,11,12,14,18,19,20,23,24,30,31],[18.0,2.0,1.0,1.0,3.0,4.0,2.0,4.0,1.0,3.0,2.0,6.0,1.0,1.0,1.0,2.0,1.0,1.0,15.0,14.0])|14   |13.0      |
|(32,[0,1,2,3,4,6,7,8,9,10,11,12,14,18,24,25,26,30,31],[18.0,4.0,3.0,1.0,2.0,4.0,3.0,2.0,1.0,1.0,3.0,2.0,1.0,3.0,1.0,1.0,1.0,8.0,8.0])         |8    |10.0      |
|(32,[0,1,2,3,4,6,7,8,9,10,11,12,14,18,19,20,23,24,30,31],[18.0,4.0,4.0,2.0,1.0,3.0,2.0,4.0,1.0,4.0,3.0,22.0,1.0,3.0,1.0,1.0,1.0,1.0,9.0,9.0]) |9    |8.0       |
|(32,[0,1,2,3,4,5,6,7,8,9,10,11,13,15,18,23,26,30,31],[19.0,2.0,3.0,1.0,3.0,1.0,5.0,4.0,2.0,1.0,2.0,5.0,1.0,1.0,1.0,1.0,1.0,7.0,5.0])          |0    |0.0       |
+----------------------------------------------------------------------------------------------------------------------------------------------+-----+----------+

Is larger better: false
labelCol: label column name (default: label)
metricName: metric name in evaluation (mse|rmse|r2|mae) (default: rmse)
predictionCol: prediction column name (default: prediction)
Root Mean Squared Error training= 2.6359793430097653
Root Mean Squared Error test = 2.3788281840880745

</code></pre>