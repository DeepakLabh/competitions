from pyspark import SparkContext

from pyspark.sql import SQLContext

sc =SparkContext()
sqlContext = SQLContext(sc)

sparkDF = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('/home/arya/deepak/competitions/kaggle/grocery/data/train.csv')

k = sparkDF.groupby('date', 'store_nbr', 'item_nbr').agg({'unit_sales':'mean'})

