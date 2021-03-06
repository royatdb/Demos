# Databricks notebook source
# MAGIC %md 
# MAGIC # Toy Analytics Use case
# MAGIC * Explore a dataset and Train a ML model
# MAGIC * Deploy the trained ML Model and make Predictions on live data
# MAGIC 
# MAGIC _Date: 7/10/2017_

# COMMAND ----------

# MAGIC %md #Training Data

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/amazon/data20K

# COMMAND ----------

data = spark.read.parquet("/databricks-datasets/amazon/data20K")
data.createOrReplaceTempView("reviews")
display(data)

# COMMAND ----------

# MAGIC %sql SELECT count(1), rating FROM reviews GROUP BY rating ORDER BY rating

# COMMAND ----------

# MAGIC %sql SELECT count(1), rating FROM reviews WHERE review LIKE '%great%' GROUP BY rating ORDER BY rating

# COMMAND ----------

# MAGIC %sql SELECT count(1), rating FROM reviews WHERE review LIKE '%poor%' GROUP BY rating ORDER BY rating

# COMMAND ----------

# MAGIC %md #NLP Pipeline

# COMMAND ----------

from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer

tokenizer = RegexTokenizer()    \
  .setInputCol("review")        \
  .setOutputCol("tokens")       \
  .setPattern("\\W+")

remover = StopWordsRemover()    \
  .setInputCol("tokens")        \
  .setOutputCol("stopWordFree") \

counts = CountVectorizer()      \
  .setInputCol("stopWordFree")  \
  .setOutputCol("features")     \
  .setVocabSize(1000)

# COMMAND ----------

from pyspark.ml.feature import Binarizer

binarizer = Binarizer()  \
  .setInputCol("rating") \
  .setOutputCol("label") \
  .setThreshold(3.5)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

lr = LogisticRegression()

p = Pipeline().setStages([tokenizer, remover, counts, binarizer, lr])

# COMMAND ----------

# MAGIC %md #Model Training

# COMMAND ----------

splits = data.randomSplit([0.8, 0.2], 42)
train = splits[0].cache()
test = splits[1].cache()

# COMMAND ----------

model = p.fit(train)
model.stages[-1].summary.areaUnderROC

# COMMAND ----------

result = model.transform(test)

# COMMAND ----------

display(result)

# COMMAND ----------

display(result.select("rating", "label", "prediction", "review"))

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator()
print "AUC: %(result)s" % {"result": evaluator.evaluate(result)}

# COMMAND ----------

# MAGIC %md #ROC

# COMMAND ----------

partialPipeline = Pipeline().setStages([tokenizer, remover, counts, binarizer])
preppedData = partialPipeline.fit(train).transform(train)
lrModel = LogisticRegression().fit(preppedData)
display(lrModel, preppedData, "ROC")

# COMMAND ----------

# MAGIC %md #Model Tuning

# COMMAND ----------

print lr.explainParams()

# COMMAND ----------

lr.setRegParam(0.01)
lr.setElasticNetParam(0.1)
counts.setVocabSize(1000)
model = p.fit(train)
result = model.transform(test)
print "AUC %(result)s" % {"result": BinaryClassificationEvaluator().evaluate(result)}

# COMMAND ----------

# MAGIC %md #Model Deployment

# COMMAND ----------

model.write().overwrite().save("/mnt/roy/amazon-model")

# COMMAND ----------

# MAGIC %fs ls dbfs:/mnt/roy/amazon-model/stages

# COMMAND ----------

# MAGIC %scala 
# MAGIC val modelDir = dbutils.fs.ls("/mnt/roy/amazon-model/stages/").last.path
# MAGIC display(spark.read.parquet(modelDir+"/data"))

# COMMAND ----------

# MAGIC %md #Model Serving

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.ml.PipelineModel
# MAGIC val model = PipelineModel.load("/mnt/roy/amazon-model")

# COMMAND ----------

# MAGIC %md #Live Data

# COMMAND ----------

# MAGIC %fs rm -r /mnt/roy/uapdemo/amazon-stream-input/

# COMMAND ----------

# MAGIC %scala
# MAGIC val testData = spark.sql("select * from amazon where time > ((select max(time) from amazon) - 14*24*60*60)")
# MAGIC println(s"Rows writen: ${testData.count()}")
# MAGIC testData.coalesce(48).write.json("/mnt/roy/uapdemo/amazon-stream-input")

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.types._
# MAGIC 
# MAGIC val streamSchema = new StructType()
# MAGIC   .add(StructField("rating",DoubleType,true))
# MAGIC   .add(StructField("review",StringType,true))
# MAGIC   .add(StructField("time",LongType,true))
# MAGIC   .add(StructField("title",StringType,true))
# MAGIC   .add(StructField("user",StringType,true))
# MAGIC 
# MAGIC spark.conf.set("spark.sql.shuffle.partitions", "4")
# MAGIC 
# MAGIC val inputStream = spark
# MAGIC   .readStream
# MAGIC   .schema(streamSchema)
# MAGIC   .option("maxFilesPerTrigger", 1)
# MAGIC   .json("/mnt/roy/uapdemo/amazon-stream-input")

# COMMAND ----------

# MAGIC %md #Model Evaluation

# COMMAND ----------

# MAGIC %scala
# MAGIC val scoredStream = model.transform(inputStream)

# COMMAND ----------

# MAGIC %scala
# MAGIC scoredStream.writeStream
# MAGIC   .format("memory")
# MAGIC   .queryName("stream")
# MAGIC   .start()

# COMMAND ----------

# MAGIC %sql select * from stream

# COMMAND ----------

# MAGIC %md #Monitoring

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.sql.functions._
# MAGIC val time = (text: String) => {
# MAGIC   new java.sql.Timestamp(text.toLong*1000)
# MAGIC }
# MAGIC val timeUdf = udf(time)
# MAGIC 
# MAGIC val isMatch = (pred: Double, label: Double) => if(label==pred) "yes" else "no"
# MAGIC val isMatchUdf = udf(isMatch)
# MAGIC 
# MAGIC display(scoredStream
# MAGIC   .withColumn("time_col", timeUdf($"time") )
# MAGIC   .withColumn("is_match", isMatchUdf($"prediction", $"label") )
# MAGIC   .groupBy($"is_match", window($"time_col", "1 day"))
# MAGIC   .count()
# MAGIC )

# COMMAND ----------

# MAGIC %md #TL;DR 
# MAGIC ![ml](https://s3.us-east-2.amazonaws.com/databricks-roy/MLDB.jpeg)

# COMMAND ----------

