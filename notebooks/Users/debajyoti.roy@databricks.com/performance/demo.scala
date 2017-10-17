// Databricks notebook source
// MAGIC %run ./setup.scala

// COMMAND ----------

table("stockprice")

// COMMAND ----------

// MAGIC %sql select count(*) from stockprice

// COMMAND ----------

// MAGIC %md # Run the query, with data skipping index on

// COMMAND ----------

enableSkipping()

// COMMAND ----------

// MAGIC %sql select count(distinct cast(time as date)) from stockprice where price > 1000

// COMMAND ----------

// MAGIC %md # Run the query again, with data skipping index off

// COMMAND ----------

disableSkipping()

// COMMAND ----------

// MAGIC %sql select count(distinct cast(time as date)) from stockprice where price > 1000

// COMMAND ----------

// MAGIC %md # Run the query with caching on

// COMMAND ----------

enableSkipping()
enableCaching()

// COMMAND ----------

// MAGIC %sql select date, avg(price) from stockprice group by date order by date

// COMMAND ----------

// MAGIC %sql select date, avg(price) from stockprice group by date having avg(price) > 11

// COMMAND ----------

// MAGIC %md # More complex queries

// COMMAND ----------

table("stockvolumes")

// COMMAND ----------

// MAGIC %sql select * from stockvolumes where volume > 998

// COMMAND ----------

// MAGIC %sql select avg(p.price) from stockprice p, stockvolumes v where p.date = v.date and v.volume > 998 group by p.date

// COMMAND ----------

enableAggPushdown()
enablePruning()

// COMMAND ----------

// MAGIC %sql select avg(p.price) from stockprice p, stockvolumes v where p.date = v.date and v.volume > 998 group by p.date

// COMMAND ----------

