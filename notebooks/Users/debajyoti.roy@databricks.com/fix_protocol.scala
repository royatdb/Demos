// Databricks notebook source
// MAGIC %md 
// MAGIC # Toy ETL Use case
// MAGIC * Ingest a complicated dataset and Transform it into tables
// MAGIC * Use Spark SQL to do data analysis on the table
// MAGIC 
// MAGIC _Date: 7/10/2017_

// COMMAND ----------

// MAGIC %md #FIX protocol
// MAGIC http://www.fixtradingcommunity.org/

// COMMAND ----------

dbutils.fs.rm("/mnt/roy/sample_fix")

// COMMAND ----------

// MAGIC %md #Ingested Data

// COMMAND ----------

dbutils.fs.put("/mnt/roy/sample_fix", """
8=FIX.4.19=11235=049=BRKR56=INVMGR34=23552=19980604-07:58:28112=19980604-07:58:2810=157
8=FIX.4.19=15435=649=BRKR56=INVMGR34=23652=19980604-07:58:4823=11568528=N55=SPMI.MI54=227=20000044=10100.00000025=H10=159
8=FIX.4.19=9035=049=INVMGR56=BRKR34=23652=19980604-07:59:3010=225
8=FIX.4.19=11235=049=BRKR56=INVMGR34=23752=19980604-07:59:48112=19980604-07:59:4810=225
8=FIX.4.19=15435=649=BRKR56=INVMGR34=23852=19980604-07:59:5623=11568628=N55=FIA.MI54=227=25000044=7900.00000025=H10=231
8=FIX.4.19=9035=049=INVMGR56=BRKR34=23752=19980604-08:00:3110=203
8=FIX.4.19=15435=649=BRKR56=INVMGR34=23952=19980604-08:00:3623=11568728=N55=PIRI.MI54=127=30000044=5950.00000025=H10=168
8=FIX.4.19=9035=049=INVMGR56=BRKR34=23852=19980604-08:01:3110=026
8=FIX.4.19=11235=049=BRKR56=INVMGR34=24052=19980604-08:01:36112=19980604-08:01:3610=190
8=FIX.4.19=9035=049=INVMGR56=BRKR34=23952=19980604-08:02:3110=026
8=FIX.4.19=11235=049=BRKR56=INVMGR34=24152=19980604-08:02:36112=19980604-08:02:3610=018
8=FIX.4.19=9035=049=INVMGR56=BRKR34=24052=19980604-08:03:3110=220
8=FIX.4.19=6135=A34=149=EXEC52=20121105-23:24:0656=BANZAI98=0108=3010=003
8=FIX.4.19=6135=A34=149=BANZAI52=20121105-23:24:0656=EXEC98=0108=3010=003
8=FIX.4.19=4935=034=249=BANZAI52=20121105-23:24:3756=EXEC10=228
8=FIX.4.19=4935=034=249=EXEC52=20121105-23:24:3756=BANZAI10=228
8=FIX.4.19=10335=D34=349=BANZAI52=20121105-23:24:4256=EXEC11=135215788257721=138=1000040=154=155=MSFT59=010=062
8=FIX.4.19=13935=834=349=EXEC52=20121105-23:24:4256=BANZAI6=011=135215788257714=017=120=031=032=037=138=1000039=054=155=MSFT150=2151=010=059
8=FIX.4.19=15335=834=449=EXEC52=20121105-23:24:4256=BANZAI6=12.311=135215788257714=1000017=220=031=12.332=1000037=238=1000039=254=155=MSFT150=2151=010=230
8=FIX.4.19=10335=D34=449=BANZAI52=20121105-23:24:5556=EXEC11=135215789503221=138=1000040=154=155=ORCL59=010=047
8=FIX.4.19=13935=834=549=EXEC52=20121105-23:24:5556=BANZAI6=011=135215789503214=017=320=031=032=037=338=1000039=054=155=ORCL150=2151=010=049
8=FIX.4.19=15335=834=649=EXEC52=20121105-23:24:5556=BANZAI6=12.311=135215789503214=1000017=420=031=12.332=1000037=438=1000039=254=155=ORCL150=2151=010=220
8=FIX.4.19=10835=D34=549=BANZAI52=20121105-23:25:1256=EXEC11=135215791235721=138=1000040=244=1054=155=SPY59=010=003
8=FIX.4.19=13835=834=749=EXEC52=20121105-23:25:1256=BANZAI6=011=135215791235714=017=520=031=032=037=538=1000039=054=155=SPY150=2151=010=252
8=FIX.4.19=10435=F34=649=BANZAI52=20121105-23:25:1656=EXEC11=135215791643738=1000041=135215791235754=155=SPY10=198
8=FIX.4.19=8235=334=849=EXEC52=20121105-23:25:1656=BANZAI45=658=Unsupported message type10=000
8=FIX.4.19=10435=F34=749=BANZAI52=20121105-23:25:2556=EXEC11=135215792530938=1000041=135215791235754=155=SPY10=197
8=FIX.4.19=8235=334=949=EXEC52=20121105-23:25:2556=BANZAI45=758=Unsupported message type10=002
""")

// COMMAND ----------

val fixDS = spark.read.textFile("/mnt/roy/sample_fix").map(_.split("\u0001"))
display(fixDS)

// COMMAND ----------

// MAGIC %md 
// MAGIC # Transformation

// COMMAND ----------

import scala.util.Try

val tagDS = fixDS.map{tagArray => 
  val body = tagArray.map{tag => 
      val parts = tag.split("=")
      val k = Try(parts(0).toInt).getOrElse(0)
      val fieldName = k match{
        case 10 => "check_sum"
        case 98 => "encrypt"
        case 108 => "heart_beat"
        case 112 => "test_req_id"
        case 21 => "handl_inst"
        case 23 => "unique_id"
        case 25 => "relative_quality"
        case 27 => "shares"
        case 28 => "transaction_type"
        case 20 => "exec_trans"
        case 31 => "last_px"
        case 32 => "last_shares"
        case 37 => "order_id"
        case 34 => "seq_num"
        case 35 => "msg_type"
        case 37 => "order_id"
        case 38 => "order_qty"
        case 39 => "ord_status"
        case 40 => "order_type"
        case 41 => "orig_ci_id"
        case 44 => "price"
        case 45 => "ref_seq"
        case 49 => "sender_company"
        case 52 => "sending_time"
        case 54 => "side"
        case 55 => "ticker"
        case 56 => "receiving_firm"
        case 58 => "free_text"
        case 59 => "time_in_force"
        case 6 => "avg_px"
        case 8 => "begin"
        case 9 => "length"
        case 11 => "ci_ord_id"
        case 14 => "cum_qty"
        case 17 => "exec_id"
        case 150 => "execution"
        case 151 => "leaves_qty"
        case _ => "unknown"
      }
      "\""+fieldName+"\""+":\""+Try(parts(1)).getOrElse("")+"\""
    }.mkString(", ")
    s"{ $body }"
}.filter(data => !data.contains("unknown"))

display(tagDS)

// COMMAND ----------

val fixJsonDS = spark.read.json(tagDS.rdd).na.fill("")

display(fixJsonDS)

// COMMAND ----------

// MAGIC %python
// MAGIC from pyspark.sql.types import DateType
// MAGIC from pyspark.sql.functions import udf
// MAGIC import time
// MAGIC 
// MAGIC # toTime function
// MAGIC #   Input: 20121105-23:25:12
// MAGIC #   Output: 2012-11-05 23:25:12 GMT
// MAGIC def toTime(fix_timestr):
// MAGIC   fix_time = time.strptime(fix_timestr, "%Y%m%d-%H:%M:%S")
// MAGIC   fix_t = time.mktime(fix_time)
// MAGIC   return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(fix_t))
// MAGIC 
// MAGIC print toTime("20121105-23:25:12")
// MAGIC 
// MAGIC # Register the UDF
// MAGIC sqlContext.udf.register("toTime", toTime)

// COMMAND ----------

fixJsonDS.selectExpr("toTime(sending_time) gmt_time", "cast (price as int) price", "sender_company", "cast (shares as int) shares", "ticker", "msg_type").na.fill(0).createOrReplaceTempView("fix_table")

// COMMAND ----------

// MAGIC %md #Curated Data

// COMMAND ----------

// MAGIC %sql select * from fix_table

// COMMAND ----------

// MAGIC %md
// MAGIC # Data Analysis

// COMMAND ----------

// MAGIC %sql select count(1), sender_company from fix_table group by sender_company

// COMMAND ----------

// MAGIC %sql select sum(price), ticker, sender_company from fix_table group by ticker, sender_company

// COMMAND ----------

// MAGIC %sql select shares, ticker, year(gmt_time) from fix_table

// COMMAND ----------

