// Databricks notebook source
// MAGIC %md #Loading FIX and Creating Tables

// COMMAND ----------

dbutils.fs.rm("/databricks/sample_fix")

// COMMAND ----------

dbutils.fs.put("/databricks/sample_fix", """
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
""")

// COMMAND ----------

// MAGIC %sh head /dbfs/databricks/sample_fix

// COMMAND ----------

val fixDS = spark.read.textFile("/databricks/sample_fix").map(_.split("\u0001"))
display(fixDS)

// COMMAND ----------

import scala.util.Try

val tagDS = fixDS.map{tagArray => 
  val body = tagArray.map{tag => 
      val parts = tag.split("=")
      val k = Try(parts(0).toInt).getOrElse(0)
      val fieldName = k match{
        case 10 => "check_sum"
        case 112 => "test_req_id"
        case 23 => "unique_id"
        case 25 => "relative_quality"
        case 27 => "shares"
        case 28 => "transaction_type"
        case 34 => "seq_num"
        case 35 => "msg_type"
        case 44 => "price"
        case 49 => "sender_company"
        case 52 => "sending_time"
        case 54 => "side"
        case 55 => "ticker"
        case 56 => "receiving_firm"
        case 8 => "begin"
        case 9 => "length"
        case _ => "unknown"
      }
      "\""+fieldName+"\""+":\""+Try(parts(1)).getOrElse("")+"\""
    }.mkString(", ")
    s"{ $body }"
}
display(tagDS)

// COMMAND ----------

val fixJsonDS = spark.read.json(tagDS.rdd).na.fill("")
display(fixJsonDS)

// COMMAND ----------

fixJsonDS.createOrReplaceTempView("fix_table")

// COMMAND ----------

// MAGIC %sql select sending_time, price, sender_company, shares, ticker from fix_table where msg_type="6"

// COMMAND ----------

// MAGIC %sql select count(1), sender_company from fix_table group by sender_company having sender_company <> ''

// COMMAND ----------

