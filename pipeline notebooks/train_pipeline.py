# Databricks notebook source
dbutils.widgets.text("mode", "train")
dbutils.widgets.text("root_dir", "/mnt/data")
dbutils.widgets.text("config_dir", "/mnt/config")
dbutils.widgets.text("account", "storageaccount")
dbutils.widgets.text("account_key", "key")

# COMMAND ----------

mode = dbutils.widgets.get("mode")
root_dir = dbutils.widgets.get("root_dir")
config_dir = dbutils.widgets.get("config_dir")
account = dbutils.widgets.get("account")
account_key = dbutils.widgets.get("account_key")

spark.conf.set(f"fs.azure.account.key.{account}.dfs.core.windows.net",
                account_key)


# COMMAND ----------

from model import train_model
from pipeline_utils import PipelineUtils


# COMMAND ----------

plu = PipelineUtils(spark, mode, root_dir, config_dir)
identifier = plu.manifest()
config = plu.read_yaml()

gold = plu.read("gold", identifier)

train_model(gold, config)